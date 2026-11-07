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
import itertools
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

OFFSET = np.array([3.0, 5.0, 7.0])

# (label, output name, how to read it back)
FORMATS = [
    ("PLY", "out.ply", "direct"),
    ("compressed PLY", "out.compressed.ply", "via-ply"),
    ("SOG (bundled)", "out.sog", "via-ply"),
    ("SOG (unbundled)", "sog/meta.json", "via-ply"),
    ("SOG, library writer, identity indices", "ordered.sog", "via-ply"),
    ("SOG, library writer, banded Morton", "banded.sog", "via-ply"),
    ("Streamed SOG (LOD)", "lod/lod-meta.json", "via-ply"),
    ("SPZ v4", "out.spz", "via-ply"),
    ("glTF KHR_gaussian_splatting", "out.glb", "glb"),
]


# Cut points, in percent, for the banded order. They include 1 so the prefix
# checks in summarise() line up with band boundaries.
CUTS = (1, 5, 10, 25, 50, 75, 100)


def morton_codes(points: np.ndarray, bits: int = 10) -> np.ndarray:
    """3D Morton (Z-order) codes after quantising each axis to ``bits`` bits."""
    lo, hi = points.min(0), points.max(0)
    q = ((points - lo) / np.maximum(hi - lo, 1e-12) * (2**bits - 1)).astype(np.uint64)
    code = np.zeros(len(points), np.uint64)
    for b in range(bits):
        for axis in range(3):
            code |= ((q[:, axis] >> np.uint64(b)) & np.uint64(1)) << np.uint64(3 * b + axis)
    return code


def banded_order(means: np.ndarray, cuts: tuple[int, ...] = CUTS) -> np.ndarray:
    """Keep each cut-point band's membership, Morton-sort within each band.

    Prefixes at the cut points then hold exactly the same splats as the
    perceptual order, while splats inside a band sit next to their spatial
    neighbours, which is what SOG's image compression rewards.
    """
    n = len(means)
    bounds = [0, *sorted({max(1, round(n * c / 100)) for c in cuts})]
    out = []
    for s, e in itertools.pairwise(bounds):
        idx = np.arange(s, e)
        out.append(idx[np.argsort(morton_codes(means[idx]), kind="stable")])
    return np.concatenate(out).astype("<u4")


def grid_scene(n: int, seed: int = 0, sh_degree: int = 3) -> tuple[Scene, np.ndarray, float]:
    """N splats on a jittered grid, in random order; returns scene, cell keys, spacing."""
    rng = np.random.default_rng(seed)
    side = math.ceil(n ** (1 / 3))
    spacing = 2.0 / side
    cells = np.stack(np.meshgrid(*[np.arange(side)] * 3, indexing="ij"), -1).reshape(-1, 3)
    cells = cells[rng.permutation(len(cells))[:n]]  # random file order
    jitter = rng.uniform(-0.15, 0.15, (n, 3)) * spacing
    # The grid is deliberately off-centre (OFFSET cells), so a writer that flips
    # or rotates axes produces unmatched splats rather than a plausible-looking
    # permutation of a symmetric grid.
    means = (cells + OFFSET) * spacing + jitter
    f = np.float32
    scene = Scene(
        means=means.astype(f),
        scales=np.full((n, 3), math.log(spacing * 0.2), f) + rng.normal(0, 0.2, (n, 3)).astype(f),
        quats=rng.normal(size=(n, 4)).astype(f),
        opacities=rng.normal(1.0, 1.5, n).astype(f),
        sh0=rng.normal(0, 1, (n, 3)).astype(f),
        shN=(rng.normal(0, 0.1, (n, (sh_degree + 1) ** 2 - 1, 3))).astype(f),
    )
    return scene, cells, spacing


def coherent_scene(n: int, seed: int = 1, sh_degree: int = 0) -> Scene:
    """Splats on surfaces whose attributes vary smoothly in space, like a real
    capture, listed in a random "importance" order.

    Random attributes compress equally badly in any order; spatially coherent
    ones are where ordering changes SOG's file size.
    """
    rng = np.random.default_rng(seed)
    k = n // 2
    d = rng.normal(size=(k, 3))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    sphere = d * 1.0
    plane = np.column_stack([rng.uniform(-3, 3, n - k), rng.uniform(-3, 3, n - k),
                             np.full(n - k, -1.2)])  # fmt: skip
    p = np.concatenate([sphere, plane])
    f = np.float32
    colour = np.column_stack([np.sin(3 * p[:, 0]), np.cos(2 * p[:, 1]), np.sin(p[:, 2] + p[:, 0])])
    scene = Scene(
        means=p.astype(f),
        scales=(np.log(0.02 + 0.01 * (1 + np.sin(4 * p))) + rng.normal(0, 0.05, (n, 3))).astype(f),
        quats=np.column_stack(
            [np.cos(p[:, 0]), np.sin(p[:, 1]), np.cos(p[:, 2]) * 0.3, np.full(n, 0.2)]
        ).astype(f),
        opacities=(2.0 + np.sin(5 * p[:, 1]) + rng.normal(0, 0.1, n)).astype(f),
        sh0=(colour + rng.normal(0, 0.02, (n, 3))).astype(f),
        shN=np.zeros((n, (sh_degree + 1) ** 2 - 1, 3), f),
    )
    return scene.take(rng.permutation(n))


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
    exact = positions / spacing - OFFSET
    cell_of = np.rint(exact).astype(int)
    err = float(np.abs(exact - cell_of).max()) if len(cell_of) else 0.0
    perm = np.array([keys.get(tuple(c), -1) for c in cell_of.tolist()])
    return perm, err


def summarise(perm: np.ndarray, n: int) -> dict[str, Any]:
    ident = np.arange(len(perm))
    matched = perm >= 0
    # Which proper prefixes hold exactly the first k input splats (as a set)?
    prefixes = [
        pct for pct in (1, 10, 25, 50)
        if len(perm) >= (k := round(n * pct / 100)) and set(perm[:k].tolist()) == set(range(k))
    ]  # fmt: skip
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
        "prefixes_valid": prefixes,
    }


def run(
    n: int,
    version: str,
    work: Path,
    gpu: str,
    only: list[str] | None = None,
    sh_degree: int = 3,
) -> list[dict[str, Any]]:
    work.mkdir(parents=True, exist_ok=True)
    scene, cells, spacing = grid_scene(n, sh_degree=sh_degree)
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
        if only and not any(o.lower() in label.lower() for o in only):
            continue
        out = work / name
        out.parent.mkdir(parents=True, exist_ok=True)
        start = time.perf_counter()
        if name == "ordered.sog":
            cmd = [node, str(ordered_writer), str(lib), str(src), str(out)]
        elif name == "banded.sog":
            order_file = work / "banded.u32"
            order_file.write_bytes(banded_order(scene.means).tobytes())
            cmd = [node, str(ordered_writer), str(lib), str(src), str(out), str(order_file)]
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
                # splat-transform writes glTF in glTF space, which is PLY space
                # rotated 180 degrees about z: x and y negated.
                positions = read_glb_positions(out) * np.array([-1.0, -1.0, 1.0])
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


def size_experiment(
    n: int, version: str, work: Path, gpu: str, sh_degree: int
) -> list[dict[str, Any]]:
    """SOG bytes for one coherent scene under three texel orders."""
    work.mkdir(parents=True, exist_ok=True)
    scene = coherent_scene(n, sh_degree=sh_degree)
    src = work / "in.ply"
    write_ply(scene, src)
    lib = install(version, work.parent / f"node-{version}")
    node = shutil.which("node")
    assert node
    writer = Path(__file__).with_name("write_sog_ordered.mjs")
    cli = str(lib / "bin" / "cli.mjs")
    (work / "banded.u32").write_bytes(banded_order(scene.means).tobytes())
    variants = {
        "Morton (command line)": [node, cli, "-w", "-q", "-g", gpu, str(src), str(work / "m.sog")],
        "psplat order kept exactly": [node, str(writer), str(lib), str(src), str(work / "i.sog")],
        "banded Morton at cut points": [node, str(writer), str(lib), str(src),
                                        str(work / "b.sog"), str(work / "banded.u32")],
    }  # fmt: skip
    rows = []
    for label, cmd in variants.items():
        start = time.perf_counter()
        subprocess.run(cmd, check=True, capture_output=True)
        out = Path(cmd[-1] if cmd[-1].endswith(".sog") else cmd[-2])
        rows.append({"variant": label, "n": n, "bytes": out.stat().st_size,
                     "write_seconds": round(time.perf_counter() - start, 1)})  # fmt: skip
        print(f"  size n={n:>7} {label:30s} {rows[-1]['bytes']:>12,} bytes", flush=True)
    base = rows[0]["bytes"]
    for r in rows:
        r["vs_morton"] = round(r["bytes"] / base, 3)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--n", type=int, nargs="+", default=[4096, 100_000])
    parser.add_argument("--version", default="3.10.3", help="splat-transform version")
    parser.add_argument("--work", type=Path, default=Path("order-spike"))
    parser.add_argument("--gpu", default="cpu", help="splat-transform -g value: cpu or an index")
    parser.add_argument("--only", nargs="*", help="run formats whose label contains these words")
    parser.add_argument("--sh-degree", type=int, default=3, help="SH degree of the test scenes")
    parser.add_argument("--size", type=int, nargs="*", default=[],
                        help="also run the SOG size experiment at these splat counts")  # fmt: skip
    args = parser.parse_args()
    rows = []
    for n in args.n:
        rows += run(n, args.version, args.work / f"n{n}", args.gpu, args.only, args.sh_degree)
    sizes = []
    for n in args.size:
        sizes += size_experiment(n, args.version, args.work / f"size{n}", args.gpu, args.sh_degree)
    args.work.mkdir(parents=True, exist_ok=True)
    summary = {"splat_transform": args.version, "sh_degree": args.sh_degree,
               "order": rows, "size": sizes}  # fmt: skip
    (args.work / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nsplat-transform {args.version}, SH degree {args.sh_degree}\n")
    print("| Format | N | Order kept | Spearman | Prefixes kept (%) | Bytes | Write s |")
    print("|---|---|---|---|---|---|---|")
    for r in rows:
        if "error" in r:
            print(f"| {r['format']} | {r['n']} | error: {' '.join(r['error'])} | | | | |")
            continue
        kept = ", ".join(str(p) for p in r["prefixes_valid"]) or "none"
        same = "yes" if r["identity"] else "no"
        print(f"| {r['format']} | {r['n']} | {same} | {r['spearman']:.3f} | {kept} | "
              f"{r['bytes']:,} | {r['write_seconds']} |")  # fmt: skip
    if sizes:
        print("\n| SOG texel order | N | Bytes | vs Morton | Write s |")
        print("|---|---|---|---|---|")
        for r in sizes:
            print(f"| {r['variant']} | {r['n']} | {r['bytes']:,} | {r['vs_morton']} | "
                  f"{r['write_seconds']} |")  # fmt: skip
    return 0


if __name__ == "__main__":
    sys.exit(main())
