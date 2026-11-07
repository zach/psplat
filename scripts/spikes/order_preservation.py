"""P1.0 spike (#19): which packed formats preserve splat order?

psplat's whole product is "the first N splats in the file are the best N".
That only survives packing if a format keeps splats in the order it was given.
This script measures it for every format PlayCanvas splat-transform writes.

Method: N splats sit on a jittered 3D grid and are written in a random order.
Each splat's grid cell identifies it even after lossy quantisation, so after a
round trip through a format the output order can be mapped back to input
indices exactly. A writer that sorts spatially (Morton, chunking, LOD) shows up
as a permutation far from the identity.

    python scripts/spikes/order_preservation.py --n 4096 100000 --version 3.10.3

Needs Node's npx. Writes a JSON summary next to its working files and prints a
Markdown table.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import struct
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from psplat.io.ply import read_ply, write_ply
from psplat.io.scene import Scene

# (label, output name, how to read it back)
FORMATS = [
    ("PLY", "out.ply", "direct"),
    ("compressed PLY", "out.compressed.ply", "via-ply"),
    ("SOG (bundled)", "out.sog", "via-ply"),
    ("SOG (unbundled)", "sog/meta.json", "via-ply"),
    ("SOG, library writer, identity indices", "ordered.sog", "via-ply"),
    ("Streamed SOG (LOD)", "lod/lod-meta.json", "via-ply"),
    ("SPZ v4", "out.spz", "via-ply"),
    ("glTF KHR_gaussian_splatting", "out.glb", "glb"),
]


def grid_scene(n: int, seed: int = 0) -> tuple[Scene, np.ndarray, float]:
    """N splats on a jittered grid, in random order; returns scene, cell keys, spacing."""
    rng = np.random.default_rng(seed)
    side = math.ceil(n ** (1 / 3))
    spacing = 2.0 / side
    cells = np.stack(np.meshgrid(*[np.arange(side)] * 3, indexing="ij"), -1).reshape(-1, 3)
    cells = cells[rng.permutation(len(cells))[:n]]  # random file order
    jitter = rng.uniform(-0.15, 0.15, (n, 3)) * spacing
    means = (cells - (side - 1) / 2) * spacing + jitter
    f = np.float32
    scene = Scene(
        means=means.astype(f),
        scales=np.full((n, 3), math.log(spacing * 0.2), f) + rng.normal(0, 0.2, (n, 3)).astype(f),
        quats=rng.normal(size=(n, 4)).astype(f),
        opacities=rng.normal(1.0, 1.5, n).astype(f),
        sh0=rng.normal(0, 1, (n, 3)).astype(f),
        shN=(rng.normal(0, 0.1, (n, 15, 3))).astype(f),
    )
    return scene, cells, spacing


def read_glb_positions(path: Path) -> np.ndarray:
    """POSITION of every primitive in a .glb, concatenated in document order."""
    data = path.read_bytes()
    magic, version, _ = struct.unpack_from("<4sII", data, 0)
    if magic != b"glTF" or version != 2:
        raise ValueError("not a glTF 2 binary")
    offset, doc, binary = 12, None, b""
    while offset < len(data):
        length, kind = struct.unpack_from("<I4s", data, offset)
        chunk = data[offset + 8 : offset + 8 + length]
        if kind == b"JSON":
            doc = json.loads(chunk)
        elif kind == b"BIN\x00":
            binary = chunk
        offset += 8 + length
    assert doc is not None
    out = []
    for mesh in doc.get("meshes", []):
        for prim in mesh["primitives"]:
            acc = doc["accessors"][prim["attributes"]["POSITION"]]
            view = doc["bufferViews"][acc["bufferView"]]
            if acc.get("componentType") != 5126 or acc.get("type") != "VEC3":
                raise ValueError(f"unexpected POSITION accessor {acc}")
            start = view.get("byteOffset", 0) + acc.get("byteOffset", 0)
            stride = view.get("byteStride", 12)
            raw = np.frombuffer(binary, np.uint8, count=stride * acc["count"], offset=start)
            out.append(raw.reshape(acc["count"], stride)[:, :12].copy().view("<f4").reshape(-1, 3))
    return np.concatenate(out) if out else np.zeros((0, 3), np.float32)


def permutation(
    positions: np.ndarray, cells: np.ndarray, spacing: float, side: int
) -> tuple[np.ndarray, float]:
    """Input index of each output splat, by grid cell; and the worst position error in cells."""
    keys = {tuple(c): i for i, c in enumerate(cells.tolist())}
    cell_of = np.rint(positions / spacing + (side - 1) / 2).astype(int)
    err = float(np.abs(positions / spacing + (side - 1) / 2 - cell_of).max()) if len(cell_of) else 0
    perm = np.array([keys.get(tuple(c), -1) for c in cell_of.tolist()])
    return perm, err


def summarise(perm: np.ndarray, n: int) -> dict[str, Any]:
    ident = np.arange(len(perm))
    matched = perm >= 0
    prefix_ok = 0
    for k in (n // 100, n // 10, n // 4, n // 2, n):
        if len(perm) >= k and set(perm[:k].tolist()) == set(range(k)):
            prefix_ok = k
    if matched.sum() > 1:
        rank = np.argsort(np.argsort(perm[matched]))
        spearman = float(np.corrcoef(rank, np.arange(matched.sum()))[0, 1])
    else:
        spearman = float("nan")
    return {
        "count_in": n,
        "count_out": len(perm),
        "unmatched": int((~matched).sum()),
        "identity": bool(len(perm) == n and np.array_equal(perm, ident)),
        "fixed_points": float((perm == ident[: len(perm)]).mean()) if len(perm) else 0.0,
        "spearman": spearman,
        "largest_valid_prefix": prefix_ok,
    }


def run(n: int, version: str, work: Path, gpu: str) -> list[dict[str, Any]]:
    work.mkdir(parents=True, exist_ok=True)
    scene, cells, spacing = grid_scene(n)
    side = math.ceil(n ** (1 / 3))
    src = work / "in.ply"
    write_ply(scene, src)
    lib = install(version, work.parent / f"node-{version}")
    node = shutil.which("node")
    assert node, "Node is required"
    cli = (
        str(lib / "bin" / "cli.mjs")
        if (lib / "bin" / "cli.mjs").exists()
        else str(lib / "dist" / "cli.mjs")
    )
    # SOG compression needs WebGPU; without a GPU it fails unless told "-g cpu".
    tool = [node, cli, "-w", "-q", "-g", gpu]
    ordered_writer = Path(__file__).with_name("write_sog_ordered.mjs")
    rows = []
    for label, name, mode in FORMATS:
        out = work / name
        out.parent.mkdir(parents=True, exist_ok=True)
        start = time.perf_counter()
        if name == "ordered.sog":
            cmd = [node, str(ordered_writer), str(lib), str(src), str(out)]
        else:
            cmd = [*tool, str(src), str(out)]
        made = subprocess.run(cmd, capture_output=True, text=True)
        secs = time.perf_counter() - start
        row: dict[str, Any] = {"format": label, "n": n, "write_seconds": round(secs, 1)}
        if made.returncode != 0:
            row["error"] = (made.stderr or made.stdout).strip().splitlines()[-1:]
            rows.append(row)
            continue
        if out.is_file():
            row["bytes"] = out.stat().st_size
        else:
            row["bytes"] = sum(p.stat().st_size for p in out.parent.rglob("*") if p.is_file())
        if out.suffix == ".json":
            row["bytes"] = sum(p.stat().st_size for p in out.parent.rglob("*") if p.is_file())
        try:
            if mode == "direct":
                positions = read_ply(out).means
            elif mode == "glb":
                positions = read_glb_positions(out)
            else:
                back = (
                    work / f"back-{label.replace(' ', '_').replace('(', '').replace(')', '')}.ply"
                )
                subprocess.run([*tool, str(out), str(back)], check=True, capture_output=True)
                positions = read_ply(back).means
        except Exception as exc:
            row["error"] = [f"read back failed: {exc}"]
            rows.append(row)
            continue
        perm, err = permutation(positions.astype(np.float64), cells, spacing, side)
        row.update(summarise(perm, n))
        row["max_position_error_cells"] = round(err, 4)
        rows.append(row)
        print(f"  n={n:>7} {label:28s} identity={row['identity']} spearman={row['spearman']:.3f}",
              flush=True)  # fmt: skip
    return rows


def install(version: str, prefix: Path) -> Path:
    """Install splat-transform once into ``prefix``; return the package directory."""
    pkg = prefix / "node_modules" / "@playcanvas" / "splat-transform"
    if not (pkg / "package.json").exists():
        prefix.mkdir(parents=True, exist_ok=True)
        npm = shutil.which("npm")
        assert npm, "Node's npm is required"
        subprocess.run([npm, "install", "--prefix", str(prefix), "--silent",
                        f"@playcanvas/splat-transform@{version}"], check=True)  # fmt: skip
    return pkg


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--n", type=int, nargs="+", default=[4096, 100_000])
    parser.add_argument("--version", default="3.10.3", help="splat-transform version")
    parser.add_argument("--work", type=Path, default=Path("order-spike"))
    parser.add_argument("--gpu", default="cpu", help="splat-transform -g value: cpu or an index")
    args = parser.parse_args()
    rows = []
    for n in args.n:
        rows += run(n, args.version, args.work / f"n{n}", args.gpu)
    (args.work / "summary.json").write_text(json.dumps({"splat_transform": args.version,
                                                        "rows": rows}, indent=2))  # fmt: skip
    print(f"\nsplat-transform {args.version}\n")
    print("| Format | N | Order kept | Spearman | Largest valid prefix | Bytes | Write s |")
    print("|---|---|---|---|---|---|---|")
    for r in rows:
        if "error" in r:
            print(f"| {r['format']} | {r['n']} | error: {' '.join(r['error'])} | | | | |")
            continue
        print(f"| {r['format']} | {r['n']} | {'yes' if r['identity'] else 'no'} | "
              f"{r['spearman']:.3f} | {r['largest_valid_prefix']} | {r['bytes']:,} | "
              f"{r['write_seconds']} |")  # fmt: skip
    return 0


if __name__ == "__main__":
    sys.exit(main())
