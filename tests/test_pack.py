"""P1.3: ``psplat pack``, without Node.

splat-transform is replaced by fakes here, so these run on every platform;
``tests/test_interop_splat_transform.py`` packs for real when
``PSPLAT_INTEROP=1``.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from typer.testing import CliRunner

import psplat.pack.pack as pk
from psplat.cli import app
from psplat.io import manifest as mf
from psplat.io.ply import read_ply, write_ply
from psplat.pack import splat_transform as st
from psplat.pack.layout import band_bounds, banded_order, banded_order_at

runner = CliRunner()


@pytest.fixture(scope="module")
def ordered(tmp_path_factory: pytest.TempPathFactory, synthetic_dir: Path) -> Path:
    """The fixture, analysed and ordered: scene.ordered.ply + scene.psplat.json."""
    work = tmp_path_factory.mktemp("ordered")
    for name in ("scene.ply", "cams.json"):
        shutil.copy(synthetic_dir / name, work / name)
    for args in (
        ["analyze", str(work / "scene.ply"), "--cameras", str(work / "cams.json"),
         "--backend", "torch", "--device", "cpu"],
        ["order", str(work / "scene.ply")],
    ):  # fmt: skip
        result = runner.invoke(app, args)
        assert result.exit_code == 0, result.output
    return work


@pytest.fixture
def workdir(tmp_path: Path, ordered: Path) -> Path:
    for name in ("scene.ordered.ply", "scene.psplat.json"):
        shutil.copy(ordered / name, tmp_path / name)
    return tmp_path


@pytest.fixture
def fake_tools(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Fake splat-transform: each output file holds its source PLY, and calls are recorded."""
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(st, "find_node", lambda: ("node", "npm"))
    monkeypatch.setattr(st, "package_dir", lambda **_: Path("fake"))

    def convert(source: Path, target: Path) -> None:
        calls.append({"fmt": target.suffix, "source": read_ply(source), "target": target})
        shutil.copyfile(source, target)

    def write_sog(source: Path, target: Path, order_file: Path) -> None:
        order = np.frombuffer(order_file.read_bytes(), "<u4").copy()
        calls.append({"fmt": ".sog", "source": read_ply(source), "target": target, "order": order})
        shutil.copyfile(source, target)

    monkeypatch.setattr(st, "convert", convert)
    monkeypatch.setattr(st, "write_sog", write_sog)
    return calls


def run(*args: str) -> Any:
    return runner.invoke(app, ["pack", *args])


# PLY ----------------------------------------------------------------------------------------


def test_ply_prefix_sizes_are_exact(tmp_path: Path, ordered: Path) -> None:
    scene = read_ply(ordered / "scene.ordered.ply")
    counts = [1, 9, 10, 99, 100, 150, 200]
    got = pk.ply_prefix_bytes(ordered / "scene.ordered.ply", counts)
    for k in counts:
        write_ply(scene.take(np.arange(k)), tmp_path / f"{k}.ply")
        assert got[k] == (tmp_path / f"{k}.ply").stat().st_size, k


def test_ply_only_needs_no_node(workdir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(shutil, "which", lambda name: None)  # no node anywhere
    result = run(str(workdir / "scene.ordered.ply"), "--ply")
    assert result.exit_code == 0, result.output
    dist = workdir / "dist"
    assert (dist / "scene.ordered.ply").read_bytes() == (workdir / "scene.ordered.ply").read_bytes()
    manifest = mf.read(dist / "scene.psplat.json")
    sizes = pk.ply_prefix_bytes(dist / "scene.ordered.ply", [c["index"] for c in manifest["cuts"]])
    for cut in manifest["cuts"]:
        assert cut["bytes_ply"] == sizes[cut["index"]]
        assert "bytes_sog" not in cut
    assert manifest["cuts"][-1]["bytes_ply"] == (dist / "scene.ordered.ply").stat().st_size
    assert manifest["provenance"]["pack"]["formats"] == ["ply"]
    # The input manifest is untouched.
    assert "bytes_ply" not in mf.read(workdir / "scene.psplat.json")["cuts"][0]


def test_missing_node_skips_other_formats_with_a_clear_message(
    workdir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(shutil, "which", lambda name: None)
    result = run(str(workdir / "scene.ordered.ply"), "--ply", "--sog", "--out", str(workdir / "o"))
    assert result.exit_code == 3
    assert "skipped sog: Node.js was not found" in result.output and "nodejs.org" in result.output
    assert (workdir / "o" / "scene.ordered.ply").exists()
    assert not (workdir / "o" / "scene.sog").exists()
    assert "bytes_sog" not in mf.read(workdir / "o" / "scene.psplat.json")["cuts"][0]


# SOG, SPZ, glTF with fake tools ----------------------------------------------------------


def test_every_format_and_cut_is_packed_from_the_right_prefix(
    workdir: Path, fake_tools: list[dict[str, Any]]
) -> None:
    result = run(str(workdir / "scene.ordered.ply"), "--sog", "--spz", "--gltf")
    assert result.exit_code == 0, result.output
    dist = workdir / "dist"
    scene = read_ply(workdir / "scene.ordered.ply")
    manifest = mf.read(dist / "scene.psplat.json")
    indices = [c["index"] for c in manifest["cuts"]]
    assert sorted({len(c["source"].means) for c in fake_tools}) == indices
    for call in fake_tools:
        k = call["source"].count
        # Every prefix is the first k splats, attributes unchanged (SH kept: 200 ≤ limit).
        assert call["source"].equals(scene.take(np.arange(k)))
        if call["fmt"] == ".sog":
            edges = [b for b in band_bounds(scene.count, [5, 10, 25, 50, 75, 100]) if b <= k]
            np.testing.assert_array_equal(call["order"], banded_order_at(scene.means[:k], edges))
    for cut in manifest["cuts"]:
        for key in ("bytes_sog", "bytes_spz", "bytes_glb"):
            assert cut[key] > 0
    for name in ("scene.sog", "scene.spz", "scene.glb"):
        assert (dist / name).stat().st_size == manifest["cuts"][-1]["bytes_" + name[-3:]]
    assert manifest["provenance"]["pack"]["sog_sh_degree"] == 3
    assert not list(dist.glob(".psplat-pack-*")), "temporary folder left behind"


def test_no_cut_sizes_packs_only_the_whole_scene(
    workdir: Path, fake_tools: list[dict[str, Any]]
) -> None:
    result = run(str(workdir / "scene.ordered.ply"), "--spz", "--no-cut-sizes")
    assert result.exit_code == 0, result.output
    assert [c["source"].count for c in fake_tools] == [200]
    cuts = mf.read(workdir / "dist" / "scene.psplat.json")["cuts"]
    assert "bytes_spz" not in cuts[0] and cuts[-1]["bytes_spz"] > 0


@pytest.mark.parametrize(("policy", "limit", "kept"), [
    ("auto", 1000, True), ("auto", 100, False), ("drop", 1000, False), ("keep", 100, True),
])  # fmt: skip
def test_sog_spherical_harmonics_policy(
    workdir: Path,
    fake_tools: list[dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
    policy: str,
    limit: int,
    kept: bool,
) -> None:
    monkeypatch.setattr(pk, "SOG_CPU_LIMIT", limit)
    result = run(str(workdir / "scene.ordered.ply"), "--sog", "--spz", "--sog-sh", policy)
    assert result.exit_code == 0, result.output
    for call in fake_tools:
        bands = call["source"].shN.shape[1]
        assert bands == (15 if kept or call["fmt"] != ".sog" else 0), call["fmt"]
    dropped = "without spherical-harmonic bands 1-3" in result.output
    assert dropped == (not kept)
    assert ("on the CPU; expect" in result.output) == (policy == "keep")
    manifest = mf.read(workdir / "dist" / "scene.psplat.json")
    assert manifest["provenance"]["pack"]["sog_sh_degree"] == (3 if kept else 0)


# Refusals ------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("args", "message"),
    [
        ([], "choose at least one format"),
        (["--ply", "--alloc", "perceptual"], "issues/38"),
        (["--sog", "--sog-sh", "maybe"], "--sog-sh must be one of"),
        (["--ply", "--manifest", "nope.json"], "nope.json not found"),
    ],
)
def test_pack_refuses_bad_requests(workdir: Path, args: list[str], message: str) -> None:
    result = run(str(workdir / "scene.ordered.ply"), *args)
    assert result.exit_code == 1
    assert message in result.output


def test_pack_refuses_a_ply_the_manifest_does_not_describe(workdir: Path) -> None:
    scene = read_ply(workdir / "scene.ordered.ply")
    scene.opacities[0] += 1
    write_ply(scene, workdir / "scene.ordered.ply")
    result = run(str(workdir / "scene.ordered.ply"), "--ply")
    assert result.exit_code == 1 and "SHA-256 differs" in result.output
    write_ply(scene.take(np.arange(150)), workdir / "scene.ordered.ply")
    result = run(str(workdir / "scene.ordered.ply"), "--ply")
    assert result.exit_code == 1 and "for 200 splats" in result.output


# Pieces ------------------------------------------------------------------------------------


def test_banded_order_at_matches_banded_order_and_checks_edges() -> None:
    rng = np.random.default_rng(0)
    means = rng.normal(size=(97, 3))
    cuts = [5, 10, 25, 50, 75, 100]
    np.testing.assert_array_equal(
        banded_order_at(means, band_bounds(97, cuts)), banded_order(means, cuts)
    )
    for bad in ([], [1, 97], [0, 50], [0, 60, 50, 97]):
        with pytest.raises(ValueError, match="band edges"):
            banded_order_at(means, bad)


def test_output_stem() -> None:
    assert pk.output_stem(Path("a/garden.ordered.ply")) == "garden"
    assert pk.output_stem(Path("garden.PLY")) == "garden"
    assert pk.output_stem(Path("garden.splat")) == "garden"


def test_cache_dir_and_override(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("PSPLAT_CACHE", str(tmp_path / "c"))
    assert st.cache_dir() == tmp_path / "c"
    monkeypatch.setenv("PSPLAT_SPLAT_TRANSFORM_DIR", str(tmp_path))
    (tmp_path / "package.json").write_text('{"version": "1.0.0"}', encoding="utf-8")
    with pytest.raises(st.ToolUnavailable, match=r"holds splat-transform 1\.0\.0"):
        st.package_dir()
