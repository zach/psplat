"""P1.2: ``psplat order``, the sort, and the ``scene.psplat.json`` manifest."""

from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from typer.testing import CliRunner

from psplat.analyze import scores_baseline as sb
from psplat.cli import app
from psplat.io import manifest as mf
from psplat.io.ply import read_ply, write_ply
from psplat.io.weights import Weights
from psplat.order.order import apply_order, sort_order
from psplat.pack.layout import band_bounds

runner = CliRunner()
SHA = "0" * 64


# The sort ------------------------------------------------------------------------------------


def test_sort_is_descending_with_ties_in_input_order() -> None:
    scores = np.array([1.0, 3.0, 2.0, 3.0, -np.inf, 2.0, 0.0])
    np.testing.assert_array_equal(sort_order(scores), [1, 3, 2, 5, 0, 6, 4])
    assert sort_order(scores).dtype == np.int64
    assert sort_order(np.zeros(0)).shape == (0,)


def test_many_ties_match_an_independent_reference() -> None:
    rng = np.random.default_rng(0)
    scores = rng.integers(0, 50, 200_000).astype(np.float64)
    scores[rng.random(scores.size) < 0.1] = -np.inf
    want = np.lexsort((np.arange(scores.size), -scores))  # last key is primary
    np.testing.assert_array_equal(sort_order(scores), want)


def test_all_equal_scores_keep_the_file_order() -> None:
    np.testing.assert_array_equal(sort_order(np.full(1000, 0.5)), np.arange(1000))


@pytest.mark.parametrize(
    ("bad", "message"),
    [
        (np.array([1.0, np.nan]), "NaN"),
        (np.array([np.inf]), r"\+inf"),
        (np.ones((2, 2)), "one per"),
    ],
)
def test_sort_refuses_scores_with_no_order(bad: np.ndarray, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        sort_order(bad)


def test_apply_order_needs_a_permutation(synthetic_dir: Path) -> None:
    scene = read_ply(synthetic_dir / "scene.ply")
    with pytest.raises(ValueError, match="permutation"):
        apply_order(scene, np.zeros(scene.count, np.int64))
    with pytest.raises(ValueError, match="permutation"):
        apply_order(scene, np.arange(scene.count - 1))


# Cut points ----------------------------------------------------------------------------------


@pytest.mark.parametrize("count", [0, 1, 3, 7, 200, 3_512_000])
def test_cuts_are_monotone_end_at_count_and_match_sog_bands(count: int) -> None:
    percents = [50, 5, 10, 25, 75]  # unsorted, and 100 left out
    cuts = mf.cuts_for(count, percents)
    assert [c["frac"] for c in cuts] == [0.05, 0.1, 0.25, 0.5, 0.75, 1.0]
    indices = [c["index"] for c in cuts]
    assert indices == sorted(indices) and indices[-1] == count
    if count:
        assert indices[0] >= 1
        # Every cut is a band edge of the banded SOG layout, so an exact prefix there too.
        assert set(indices) <= set(band_bounds(count, percents))


def test_handoff_example_indices() -> None:
    cuts = mf.cuts_for(3_512_000, [5])
    assert cuts[0] == {"frac": 0.05, "index": 175_600}


# The manifest --------------------------------------------------------------------------------


def handoff_example() -> dict[str, Any]:
    """The example of handoff §3.4, filled in."""
    return {
        "version": 1,
        "source_sha256": SHA,
        "count": 3512000,
        "ordering": {
            "method": "perceptual-v1",
            "terms": ["contribution", "csf", "masking"],
            "ppd": 60,
        },
        "cuts": [
            {"frac": 0.05, "index": 175600, "bytes_sog": 812345, "flip": 0.11, "lpips": 0.21},
            {"frac": 1.0, "index": 3512000},
        ],
        "lod": {"projected_sigma_px_min": [0.5, 1.0, 2.0]},
        "provenance": {"tool": "psplat 0.1.0", "backend": "gsplat@abc123", "date": "2026-11-08"},
    }


def test_handoff_example_validates() -> None:
    mf.validate(handoff_example())


def break_it(path: str, value: Any) -> dict[str, Any]:
    m = copy.deepcopy(handoff_example())
    *parents, last = path.split(".")
    node: Any = m
    for p in parents:
        node = node[int(p)] if p.isdigit() else node[p]
    if value is KeyError:
        del node[int(last) if last.isdigit() else last]
    else:
        node[int(last) if last.isdigit() else last] = value
    return m


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        ("version", 2, "version 2"),
        ("source_sha256", "abc", "source_sha256"),
        ("count", -1, "count"),
        ("ordering.method", KeyError, "method"),
        ("ordering.colour", "red", "colour"),
        ("cuts", [], "cuts"),
        ("cuts.0.frac", 0, "frac"),
        ("cuts.0.index", 1.5, "index"),
        ("cuts.0.bytes_sog", -3, "bytes_sog"),
        ("cuts.0.sharpness", 1.0, "sharpness"),
        ("cuts.1.index", 4_000_000, "exceeds count"),
        ("cuts.1.frac", 0.05, "frac must increase"),
        ("cuts.1.index", 10, "must not decrease"),
        ("provenance", KeyError, "provenance"),
        ("provenance.tool", "other 1.0", "tool"),
        ("extra", 1, "extra"),
    ],
)
def test_invalid_manifests_are_rejected(path: str, value: Any, message: str) -> None:
    with pytest.raises(mf.ManifestError, match=message):
        mf.validate(break_it(path, value))


def test_write_read_round_trip(tmp_path: Path) -> None:
    m = handoff_example()
    mf.write(m, tmp_path / "m.json")
    raw = (tmp_path / "m.json").read_bytes()
    assert b"\r\n" not in raw and raw.endswith(b"}\n")
    assert mf.read(tmp_path / "m.json") == m
    (tmp_path / "bad.json").write_text("{", encoding="utf-8")
    with pytest.raises(mf.ManifestError, match="not JSON"):
        mf.read(tmp_path / "bad.json")
    (tmp_path / "old.json").write_text(json.dumps(break_it("version", 0)), encoding="utf-8")
    with pytest.raises(mf.ManifestError, match=r"old\.json: manifest version 0"):
        mf.read(tmp_path / "old.json")


def test_invalid_manifest_is_not_written(tmp_path: Path) -> None:
    with pytest.raises(mf.ManifestError):
        mf.write(break_it("count", -1), tmp_path / "m.json")
    assert not (tmp_path / "m.json").exists()


# psplat order --------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def analysed(tmp_path_factory: pytest.TempPathFactory, synthetic_dir: Path) -> Path:
    """A directory with the fixture and its weights (all baseline terms)."""
    work = tmp_path_factory.mktemp("analysed")
    for name in ("scene.ply", "cams.json"):
        shutil.copy(synthetic_dir / name, work / name)
    result = runner.invoke(app, [
        "analyze", str(work / "scene.ply"), "--cameras", str(work / "cams.json"),
        "--backend", "torch", "--device", "cpu", "--views", "all", "--terms", "baselines",
    ])  # fmt: skip
    assert result.exit_code == 0, result.output
    return work


@pytest.fixture
def workdir(tmp_path: Path, analysed: Path) -> Path:
    for name in ("scene.ply", "cams.json", "scene.weights.npz"):
        shutil.copy(analysed / name, tmp_path / name)
    return tmp_path


@pytest.mark.parametrize("method", list(sb.BASELINES))
def test_order_writes_the_ordered_scene_and_a_valid_manifest(
    workdir: Path, method: str, oracles: dict[str, Any]
) -> None:
    result = runner.invoke(app, ["order", str(workdir / "scene.ply"), "--method", method])
    assert result.exit_code == 0, result.output
    scene = read_ply(workdir / "scene.ply")
    weights = Weights.load(workdir / "scene.weights.npz")
    want = sort_order(sb.score(method, scene, weights))
    ordered = read_ply(workdir / "scene.ordered.ply")
    # Every attribute, extras included, is the input's, permuted.
    assert ordered.equals(scene.take(want))
    assert not ordered.equals(scene)

    manifest = mf.read(workdir / "scene.psplat.json")
    assert manifest["count"] == scene.count == 200
    assert manifest["ordering"]["method"] == method
    assert manifest["ordering"]["terms"] == list(sb.BASELINES[method].needs)
    assert ("seed" in manifest["ordering"]) == sb.BASELINES[method].sampled
    indices = [c["index"] for c in manifest["cuts"]]
    assert indices == [10, 20, 50, 100, 150, 200]
    from psplat.provenance import sha256_file

    assert manifest["source_sha256"] == sha256_file(workdir / "scene.ply")
    if method != "opacity":
        # Off-screen splats are never inside a budget short of everything.
        off = set(oracles["offscreen"]["splats"])
        assert off <= set(want[-len(off) :].tolist())
        assert manifest["provenance"]["backend"] == "torch on cpu"
        assert manifest["provenance"]["analysis"]["weights_sha256"] == sha256_file(
            workdir / "scene.weights.npz"
        )


def test_order_is_deterministic(workdir: Path) -> None:
    for name in ("a.ply", "b.ply"):
        result = runner.invoke(app, [
            "order", str(workdir / "scene.ply"), str(workdir / "scene.weights.npz"),
            "--method", "coreset-simple", "--seed", "3", "--out", str(workdir / name),
        ])  # fmt: skip
        assert result.exit_code == 0, result.output
    assert (workdir / "a.ply").read_bytes() == (workdir / "b.ply").read_bytes()
    manifest = mf.read(workdir / "scene.psplat.json")
    assert manifest["ordering"]["seed"] == 3


def test_order_custom_cuts_and_paths(workdir: Path) -> None:
    # Missing output folders are created.
    result = runner.invoke(app, [
        "order", str(workdir / "scene.ply"), "--cuts", "1,33", "--out",
        str(workdir / "out" / "x.ply"), "--manifest", str(workdir / "meta" / "m.json"),
    ])  # fmt: skip
    assert result.exit_code == 0, result.output
    cuts = mf.read(workdir / "meta" / "m.json")["cuts"]
    assert [(c["frac"], c["index"]) for c in cuts] == [(0.01, 2), (0.33, 66), (1.0, 200)]
    assert "1% 2, 33% 66, 100% 200" in result.output


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (["--method", "random"], "unknown --method"),
        (["--cuts", "0,50"], "--cuts"),
        (["--cuts", "5,abc"], "--cuts"),
        (["--out", "{scene}"], "must differ"),
    ],
)
def test_order_rejects_bad_options(workdir: Path, args: list[str], message: str) -> None:
    args = [a.replace("{scene}", str(workdir / "scene.ply")) for a in args]
    result = runner.invoke(app, ["order", str(workdir / "scene.ply"), *args])
    assert result.exit_code == 1
    assert message in result.output


def test_order_explains_missing_weights_and_terms(workdir: Path, synthetic_dir: Path) -> None:
    (workdir / "scene.weights.npz").rename(workdir / "elsewhere.npz")
    result = runner.invoke(app, ["order", str(workdir / "scene.ply")])
    assert result.exit_code == 1 and "run it first" in result.output
    # opacity needs no analysis at all.
    result = runner.invoke(app, ["order", str(workdir / "scene.ply"), "--method", "opacity"])
    assert result.exit_code == 0, result.output

    plain = workdir / "plain.npz"
    result = runner.invoke(app, [
        "analyze", str(workdir / "scene.ply"), "--cameras", str(workdir / "cams.json"),
        "--backend", "torch", "--device", "cpu", "--out", str(plain),
    ])  # fmt: skip
    assert result.exit_code == 0, result.output
    result = runner.invoke(
        app, ["order", str(workdir / "scene.ply"), str(plain), "--method", "pup-simple"]
    )
    assert result.exit_code == 1 and "--terms contribution,fisher" in result.output


def test_order_refuses_weights_from_another_scene(workdir: Path) -> None:
    scene = read_ply(workdir / "scene.ply")
    scene.opacities[0] += 1.0  # same splat count, different file
    write_ply(scene, workdir / "other.ply")
    result = runner.invoke(
        app, ["order", str(workdir / "other.ply"), str(workdir / "scene.weights.npz")]
    )
    assert result.exit_code == 1 and "different PLY" in result.output
