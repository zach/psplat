"""Interop with PlayCanvas splat-transform, an independent reader and writer.

Opt-in, because it installs splat-transform from npm: set PSPLAT_INTEROP=1.
CI runs it on Linux. Locally::

    PSPLAT_INTEROP=1 pytest tests/test_interop_splat_transform.py

Besides PLY agreement, it pins down which formats keep splat order (decision
D7, docs/formats.md). If splat-transform changes how a writer orders splats,
these tests fail before psplat's truncation silently breaks.
"""

from __future__ import annotations

import csv
import json
import os
import shutil
import struct
import subprocess
from pathlib import Path

import numpy as np
import pytest

from psplat.io.ply import read_ply, write_ply
from psplat.io.scene import Scene
from psplat.pack import SPLAT_TRANSFORM_VERSION
from psplat.pack.layout import band_bounds, banded_order

ORDERED_WRITER = (
    Path(__file__).resolve().parents[1] / "scripts" / "spikes" / "write_sog_ordered.mjs"
)

pytestmark = pytest.mark.skipif(
    os.environ.get("PSPLAT_INTEROP") != "1" or shutil.which("npm") is None,
    reason="set PSPLAT_INTEROP=1 with Node's npm on PATH to run",
)


@pytest.fixture(scope="session")
def splat_transform(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The pinned splat-transform package directory, installed once per session."""
    prefix = tmp_path_factory.mktemp("splat-transform")
    npm = shutil.which("npm")
    assert npm is not None
    subprocess.run([npm, "install", "--prefix", str(prefix), "--silent",
                    f"@playcanvas/splat-transform@{SPLAT_TRANSFORM_VERSION}"],
                   check=True, capture_output=True)  # fmt: skip
    return prefix / "node_modules" / "@playcanvas" / "splat-transform"


def _node() -> str:
    node = shutil.which("node")
    assert node is not None
    return node


def _cli(pkg: Path, *args: str) -> None:
    # "-g cpu": CI has no GPU, and SOG fails without an adapter unless told so.
    cmd = [_node(), str(pkg / "bin" / "cli.mjs"), "-w", "-q", "-g", "cpu", *args]
    subprocess.run(cmd, check=True, capture_output=True)


def _scene(n: int = 64, degree: int = 3) -> Scene:
    rng = np.random.default_rng(11)
    f = np.float32
    return Scene(
        means=rng.normal(size=(n, 3)).astype(f),
        scales=(rng.normal(size=(n, 3)) - 3).astype(f),
        quats=np.abs(rng.normal(size=(n, 4))).astype(f) + 0.1,
        opacities=rng.normal(size=n).astype(f),
        sh0=rng.normal(size=(n, 3)).astype(f),
        shN=(rng.normal(size=(n, (degree + 1) ** 2 - 1, 3)) * 0.1).astype(f),
    )


# PLY agreement -------------------------------------------------------------------------


def test_their_ply_matches_ours_bit_for_bit_and_keeps_order(
    tmp_path: Path, splat_transform: Path
) -> None:
    ours = _scene()
    write_ply(ours, tmp_path / "ours.ply")
    _cli(splat_transform, str(tmp_path / "ours.ply"), str(tmp_path / "theirs.ply"))
    theirs = read_ply(tmp_path / "theirs.ply")
    # splat-transform writes a different property order, so this also checks
    # that the reader goes by name, not position.
    names = [n for n, _ in theirs.ply_layout.properties] if theirs.ply_layout else []
    assert names[:4] != ["x", "y", "z", "nx"]
    for field in ("means", "scales", "quats", "opacities", "sh0", "shN"):
        np.testing.assert_array_equal(getattr(theirs, field), getattr(ours, field), err_msg=field)


def test_their_reading_of_our_file_matches(tmp_path: Path, splat_transform: Path) -> None:
    ours = _scene()
    write_ply(ours, tmp_path / "ours.ply")
    _cli(splat_transform, str(tmp_path / "ours.ply"), str(tmp_path / "theirs.csv"))
    with (tmp_path / "theirs.csv").open(newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == ours.count
    col = {k: np.array([float(r[k]) for r in rows], np.float32) for k in rows[0]}
    np.testing.assert_allclose(col["x"], ours.means[:, 0], rtol=1e-6)
    np.testing.assert_allclose(col["opacity"], ours.opacities, rtol=1e-6)
    # f_rest is channel-major: green (channel 1) coefficient 2 is f_rest_17.
    np.testing.assert_allclose(col["f_rest_17"], ours.shN[:, 2, 1], rtol=1e-5, atol=1e-7)


# Order preservation (D7) ---------------------------------------------------------------

GRID = 8  # 8^3 = 512 splats
OFFSET = np.array([3.0, 5.0, 7.0])  # off-centre, so an axis flip cannot pass


def _grid_scene() -> tuple[Scene, np.ndarray]:
    """512 splats on an off-centre jittered grid, written in random order."""
    rng = np.random.default_rng(3)
    cells = np.stack(np.meshgrid(*[np.arange(GRID)] * 3, indexing="ij"), -1).reshape(-1, 3)
    cells = cells[rng.permutation(len(cells))]
    n = len(cells)
    f = np.float32
    scene = Scene(
        means=((cells + OFFSET) * 0.25 + rng.uniform(-0.03, 0.03, (n, 3))).astype(f),
        scales=np.full((n, 3), -3.0, f),
        quats=np.tile(np.array([1, 0, 0, 0], f), (n, 1)),
        opacities=np.zeros(n, f),
        sh0=rng.normal(size=(n, 3)).astype(f),
        shN=np.zeros((n, 0, 3), f),
    )
    return scene, cells


def _input_index(positions: np.ndarray, cells: np.ndarray) -> np.ndarray:
    """Input index of each output splat, recovered from its grid cell."""
    lookup = {tuple(c): i for i, c in enumerate(cells.tolist())}
    found = np.rint(positions / 0.25 - OFFSET).astype(int)
    return np.array([lookup.get(tuple(c), -1) for c in found.tolist()])


def _glb_positions(path: Path) -> np.ndarray:
    data = path.read_bytes()
    length = struct.unpack_from("<I", data, 12)[0]
    doc = json.loads(data[20 : 20 + length])
    binary = data[20 + length + 8 :]
    acc = doc["accessors"][doc["meshes"][0]["primitives"][0]["attributes"]["POSITION"]]
    view = doc["bufferViews"][acc["bufferView"]]
    start = view.get("byteOffset", 0) + acc.get("byteOffset", 0)
    xyz = np.frombuffer(binary, "<f4", count=3 * acc["count"], offset=start).reshape(-1, 3)
    # splat-transform's glTF space is PLY space with x and y negated.
    return xyz * np.array([-1.0, -1.0, 1.0])


@pytest.mark.parametrize("suffix", [".ply", ".spz", ".glb"])
def test_these_formats_keep_splat_order(tmp_path: Path, splat_transform: Path, suffix: str) -> None:
    scene, cells = _grid_scene()
    write_ply(scene, tmp_path / "in.ply")
    out = tmp_path / f"out{suffix}"
    _cli(splat_transform, str(tmp_path / "in.ply"), str(out))
    if suffix == ".glb":
        positions = _glb_positions(out)
    else:
        _cli(splat_transform, str(out), str(tmp_path / "back.ply"))
        positions = read_ply(tmp_path / "back.ply").means.astype(np.float64)
    np.testing.assert_array_equal(_input_index(positions, cells), np.arange(len(cells)))


@pytest.mark.parametrize("output", ["out.sog", "out.compressed.ply"])
def test_command_line_sog_and_compressed_ply_reorder(
    tmp_path: Path, splat_transform: Path, output: str
) -> None:
    # Documented upstream behaviour (Morton sort). If this starts failing,
    # splat-transform changed its writers: revisit D7 in docs/plan.md.
    scene, cells = _grid_scene()
    write_ply(scene, tmp_path / "in.ply")
    _cli(splat_transform, str(tmp_path / "in.ply"), str(tmp_path / output))
    _cli(splat_transform, str(tmp_path / output), str(tmp_path / "back.ply"))
    perm = _input_index(read_ply(tmp_path / "back.ply").means.astype(np.float64), cells)
    assert sorted(perm.tolist()) == list(range(len(cells)))  # same splats...
    assert np.mean(perm == np.arange(len(cells))) < 0.1  # ...in a different order


@pytest.mark.parametrize("cuts", [None, [5, 10, 25, 50, 75, 100]])
def test_library_sog_writer_honours_a_given_order(
    tmp_path: Path, splat_transform: Path, cuts: list[int] | None
) -> None:
    scene, cells = _grid_scene()
    write_ply(scene, tmp_path / "in.ply")
    cmd = [_node(), str(ORDERED_WRITER), str(splat_transform), str(tmp_path / "in.ply"),
           str(tmp_path / "out.sog")]  # fmt: skip
    if cuts is not None:
        (tmp_path / "order.u32").write_bytes(banded_order(scene.means, cuts).tobytes())
        cmd.append(str(tmp_path / "order.u32"))
    subprocess.run(cmd, check=True, capture_output=True)
    _cli(splat_transform, str(tmp_path / "out.sog"), str(tmp_path / "back.ply"))
    perm = _input_index(read_ply(tmp_path / "back.ply").means.astype(np.float64), cells)
    n = len(cells)
    if cuts is None:
        np.testing.assert_array_equal(perm, np.arange(n))
    else:
        for k in band_bounds(n, cuts):
            assert set(perm[:k].tolist()) == set(range(k)), f"prefix {k} is not exact"
        # The identity also passes the prefix check; make sure the order file was used.
        np.testing.assert_array_equal(perm, banded_order(scene.means, cuts))


# psplat pack (P1.3) --------------------------------------------------------------------


def _packed(
    tmp_path: Path,
    splat_transform: Path,
    monkeypatch: pytest.MonkeyPatch,
    degree: int,
    *formats: str,
    sog_sh: str = "auto",
) -> tuple[Scene, np.ndarray, Path]:
    """Pack the grid scene, treating its file order as psplat's order."""
    from psplat.io import manifest as mf
    from psplat.pack.pack import pack

    monkeypatch.setenv("PSPLAT_SPLAT_TRANSFORM_DIR", str(splat_transform))
    scene, cells = _grid_scene()
    rng = np.random.default_rng(5)
    scene.shN = (rng.normal(size=(scene.count, (degree + 1) ** 2 - 1, 3)) * 0.2).astype(np.float32)
    write_ply(scene, tmp_path / "grid.ordered.ply")
    manifest = mf.build(
        source_sha256="0" * 64, count=scene.count, method="contribution", terms=["contribution"],
        cuts_percent=[5, 10, 25, 50, 75, 100], provenance={"tool": "psplat test"},
    )  # fmt: skip
    mf.write(manifest, tmp_path / "grid.psplat.json")
    pack(tmp_path / "grid.ordered.ply", tmp_path / "grid.psplat.json", formats,
         tmp_path / "dist", sog_sh=sog_sh)  # fmt: skip
    return scene, cells, tmp_path / "dist"


def test_pack_spz_and_gltf_keep_exact_order(
    tmp_path: Path, splat_transform: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scene, cells, dist = _packed(tmp_path, splat_transform, monkeypatch, 0, "spz", "gltf")
    _cli(splat_transform, str(dist / "grid.spz"), str(tmp_path / "spz.ply"))
    spz = _input_index(read_ply(tmp_path / "spz.ply").means.astype(np.float64), cells)
    glb = _input_index(_glb_positions(dist / "grid.glb"), cells)
    np.testing.assert_array_equal(spz, np.arange(scene.count))
    np.testing.assert_array_equal(glb, np.arange(scene.count))


@pytest.mark.parametrize(("degree", "policy", "bands"), [(0, "auto", 0), (1, "auto", 3),
                                                          (1, "drop", 0)])  # fmt: skip
def test_pack_sog_cuts_are_exact_prefixes(
    tmp_path: Path, splat_transform: Path, monkeypatch: pytest.MonkeyPatch,
    degree: int, policy: str, bands: int,
) -> None:  # fmt: skip
    from psplat.io import manifest as mf
    from psplat.pack.layout import banded_order_at

    scene, cells, dist = _packed(tmp_path, splat_transform, monkeypatch, degree, "sog",
                                 sog_sh=policy)  # fmt: skip
    _cli(splat_transform, str(dist / "grid.sog"), str(tmp_path / "back.ply"))
    back = read_ply(tmp_path / "back.ply")
    perm = _input_index(back.means.astype(np.float64), cells)
    cuts = mf.read(dist / "grid.psplat.json")["cuts"]
    edges = [0] + [c["index"] for c in cuts]
    np.testing.assert_array_equal(perm, banded_order_at(scene.means, edges))
    for cut in cuts:
        k = cut["index"]
        # The first k texels are exactly the splats a PLY truncated at k holds.
        assert set(perm[:k].tolist()) == set(range(k)), f"cut {k}"
        assert cut["bytes_sog"] > 0
    assert back.shN.shape[1] == bands
