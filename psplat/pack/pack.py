"""``psplat pack``: write an ordered scene in shippable formats (P1.3, #22).

Input is the ordered PLY and manifest from ``psplat order``. Output, in one
folder:

- ``<stem>.ordered.ply`` — the ordered PLY itself; any prefix is the best
  splats for that budget;
- ``<stem>.sog`` — PlayCanvas SOG in a *banded* order (decision D7): splats
  stay inside their cut-point band and are Morton-sorted only within it, so
  every cut point is an exact prefix while most of SOG's compression stays;
- ``<stem>.spz`` (SPZ v4) and ``<stem>.glb`` (glTF ``KHR_gaussian_splatting``),
  both in exact psplat order;
- ``<stem>.psplat.json`` — the manifest, with ``bytes_<format>`` added to
  each cut: the size of that budget's prefix packed on its own, which is the
  x-axis of a rate–quality curve.

SOG clusters spherical harmonics into a palette of up to 64K entries.
splat-transform does that on a GPU through WebGPU, which psplat cannot drive
yet, and on a CPU it is only practical for small scenes (``docs/formats.md``).
So by default SOG keeps the higher SH bands only up to
:data:`SOG_CPU_LIMIT` splats and otherwise drops them, and says so. The
other formats always keep them.
"""

from __future__ import annotations

import math
import shutil
import tempfile
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from psplat.io import manifest as mf
from psplat.io.ply import read_ply, read_ply_header, write_ply
from psplat.pack import SPLAT_TRANSFORM_VERSION, splat_transform
from psplat.pack.layout import banded_order_at
from psplat.provenance import sha256_file

FORMATS = ("ply", "sog", "spz", "gltf")
EXTENSIONS = {"sog": ".sog", "spz": ".spz", "gltf": ".glb"}
# Manifest key suffix per format: bytes_ply, bytes_sog, bytes_spz, bytes_glb.
SIZE_KEYS = {"ply": "bytes_ply", "sog": "bytes_sog", "spz": "bytes_spz", "gltf": "bytes_glb"}
# Largest scene whose SOG spherical-harmonic palette psplat clusters on the
# CPU by default: 32,768 splats took 4.7 s on 4 threads; 100,000 ran past
# 25 minutes (docs/formats.md).
SOG_CPU_LIMIT = 32_768
SOG_SH_POLICIES = ("auto", "keep", "drop")
ALLOCATIONS = ("uniform",)


class PackError(ValueError):
    """The inputs to ``pack`` are inconsistent or the request cannot be met."""


@dataclass
class PackResult:
    outputs: dict[str, Path] = field(default_factory=dict)
    skipped: dict[str, str] = field(default_factory=dict)  # format -> reason
    notices: list[str] = field(default_factory=list)
    manifest_path: Path | None = None
    manifest: dict[str, Any] = field(default_factory=dict)


def output_stem(ordered_ply: Path) -> str:
    name = ordered_ply.name
    for suffix in (".ordered.ply", ".ply"):
        if name.lower().endswith(suffix):
            return name[: -len(suffix)]
    return ordered_ply.stem


def ply_prefix_bytes(path: Path, counts: Sequence[int]) -> dict[int, int]:
    """Exact size of a PLY holding only the first ``k`` splats, for each ``k``.

    Only the vertex count in the header and the number of rows change, so no
    file needs writing.
    """
    header = read_ply_header(path)
    if [el.name for el in header.elements] != ["vertex"] or header.format == "ascii":
        raise PackError(f"{path}: expected a binary PLY with only a vertex element")
    n = header.element("vertex").count
    body = path.stat().st_size - header.size
    if n and body % n:
        raise PackError(f"{path}: body of {body} bytes does not divide into {n} rows")
    stride = body // n if n else 0
    digits = len(str(n))
    return {k: header.size + len(str(k)) - digits + k * stride for k in counts}


def _bounds(manifest: dict[str, Any]) -> list[int]:
    count = int(manifest["count"])
    return sorted({0, count, *(int(c["index"]) for c in manifest["cuts"])})


def _check_inputs(ordered_ply: Path, manifest: dict[str, Any], count: int) -> None:
    if manifest["count"] != count:
        raise PackError(
            f"the manifest is for {manifest['count']:,} splats but {ordered_ply.name} has "
            f"{count:,}; were they written by the same psplat order run?"
        )
    recorded = manifest.get("provenance", {}).get("output", {}).get("ply_sha256")
    if recorded is not None and recorded != sha256_file(ordered_ply):
        raise PackError(
            f"{ordered_ply.name} is not the file this manifest was written with "
            "(SHA-256 differs); rerun psplat order"
        )


def pack(
    ordered_ply: Path,
    manifest_path: Path,
    formats: Sequence[str],
    out_dir: Path,
    *,
    alloc: str = "uniform",
    cut_sizes: bool = True,
    sog_sh: str = "auto",
    log: Callable[[str], None] = lambda message: None,
) -> PackResult:
    """Write ``formats`` of ``ordered_ply`` into ``out_dir`` with an updated manifest."""
    unknown = [f for f in formats if f not in FORMATS]
    if unknown or not formats:
        raise PackError(f"choose one or more formats from {', '.join(FORMATS)}")
    if alloc not in ALLOCATIONS:
        raise PackError(
            f"--alloc {alloc} is not available yet; only uniform. Perceptual attribute "
            "allocation is tracked in https://github.com/zach/psplat/issues/38"
        )
    if sog_sh not in SOG_SH_POLICIES:
        raise PackError(f"--sog-sh must be one of {', '.join(SOG_SH_POLICIES)}")

    manifest = mf.read(manifest_path)
    scene = read_ply(ordered_ply)
    _check_inputs(ordered_ply, manifest, scene.count)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = output_stem(ordered_ply)
    bounds = _bounds(manifest)
    budgets = [b for b in bounds if b > 0]
    sizes: dict[str, dict[int, int]] = {}
    result = PackResult()
    pack_info: dict[str, Any] = {"formats": [], "alloc": alloc, "cut_sizes": cut_sizes}

    if "ply" in formats:
        target = out_dir / f"{stem}.ordered.ply"
        if target.resolve() != ordered_ply.resolve():
            shutil.copyfile(ordered_ply, target)
        result.outputs["ply"] = target
        sizes["ply"] = ply_prefix_bytes(target, budgets)
        pack_info["formats"].append("ply")
        log(f"  ply: {target.name}")

    others = [f for f in formats if f != "ply"]
    if others:
        try:
            splat_transform.find_node()
            splat_transform.package_dir()
        except splat_transform.ToolUnavailable as exc:
            for f in others:
                result.skipped[f] = str(exc)
            others = []
        else:
            pack_info["splat_transform"] = SPLAT_TRANSFORM_VERSION

    degree = math.isqrt(scene.shN.shape[1] + 1) - 1
    sog_keeps_sh = True
    if "sog" in others:
        keep_sh = (
            degree == 0 or sog_sh == "keep" or (sog_sh == "auto" and scene.count <= SOG_CPU_LIMIT)
        )
        sog_keeps_sh = keep_sh
        pack_info["sog_sh_degree"] = degree if keep_sh else 0
        if not keep_sh:
            reason = (
                "--sog-sh drop" if sog_sh == "drop"
                else f"{scene.count:,} splats is more than the {SOG_CPU_LIMIT:,} psplat "
                "clusters on the CPU, and GPU clustering is not wired up yet"
            )  # fmt: skip
            result.notices.append(
                f"SOG written without spherical-harmonic bands 1-{degree} ({reason}); "
                "view-dependent colour is lost in the SOG only. PLY, SPZ and glTF keep it. "
                "--sog-sh keep clusters on the CPU instead, which takes far longer."
            )
        elif degree > 0 and scene.count > SOG_CPU_LIMIT:
            result.notices.append(
                f"clustering SOG spherical harmonics for {scene.count:,} splats on the CPU; "
                "expect this to take a long time"
            )

    with tempfile.TemporaryDirectory(prefix=".psplat-pack-", dir=out_dir) as tmp:
        work = Path(tmp)
        prefix_files: dict[int, Path] = {}

        def prefix_ply(k: int, sh: bool) -> Path:
            if k == scene.count and sh:
                return ordered_ply
            key = k if sh else -k - 1
            if key not in prefix_files:
                part = scene.take(np.arange(k))
                if not sh:
                    part.shN = np.zeros((k, 0, 3), np.float32)
                path = work / f"prefix-{k}{'' if sh else '-sh0'}.ply"
                write_ply(part, path, layout="source" if sh else "canonical")
                prefix_files[key] = path
            return prefix_files[key]

        for fmt in others:
            start = time.perf_counter()
            sh = fmt != "sog" or sog_keeps_sh
            target = out_dir / f"{stem}{EXTENSIONS[fmt]}"
            sizes[fmt] = {}
            wanted = budgets if cut_sizes else [scene.count]
            for k in sorted(set(wanted) | {scene.count}):
                dest = target if k == scene.count else work / f"cut-{k}{EXTENSIONS[fmt]}"
                source = prefix_ply(k, sh)
                if fmt == "sog":
                    order_file = work / f"order-{k}.u32"
                    edges = [b for b in bounds if b <= k]
                    order = banded_order_at(scene.means[:k], edges)
                    order_file.write_bytes(order.astype("<u4").tobytes())
                    splat_transform.write_sog(source, dest, order_file)
                else:
                    splat_transform.convert(source, dest)
                sizes[fmt][k] = dest.stat().st_size
            result.outputs[fmt] = target
            pack_info["formats"].append(fmt)
            log(f"  {fmt}: {target.name} ({time.perf_counter() - start:.1f} s)")

    for cut in manifest["cuts"]:
        for fmt, by_count in sizes.items():
            if int(cut["index"]) in by_count:
                cut[SIZE_KEYS[fmt]] = by_count[int(cut["index"])]
            else:
                cut.pop(SIZE_KEYS[fmt], None)
    manifest.setdefault("provenance", {})["pack"] = pack_info
    result.manifest = manifest
    result.manifest_path = out_dir / f"{stem}.psplat.json"
    mf.write(manifest, result.manifest_path)
    return result
