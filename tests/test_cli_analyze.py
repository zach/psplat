"""P0.7: ``psplat analyze``, the weights file, and backend selection."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from typer.testing import CliRunner

import psplat.render.registry as registry
from psplat.cli import app
from psplat.io.cameras import read_cameras_json, write_cameras
from psplat.io.ply import read_ply
from psplat.io.weights import Weights, WeightsFormatError
from psplat.render.torch_backend import TorchRenderer

runner = CliRunner()


@pytest.fixture
def workdir(tmp_path: Path, synthetic_dir: Path) -> Path:
    for name in ("scene.ply", "cams.json"):
        shutil.copy(synthetic_dir / name, tmp_path / name)
    return tmp_path


def run(*args: str) -> Any:
    return runner.invoke(app, list(args))


def test_analyze_writes_weights_matching_the_backend(
    workdir: Path, oracles: dict[str, Any]
) -> None:
    out = workdir / "w.npz"
    result = run(
        "analyze", str(workdir / "scene.ply"), "--cameras", str(workdir / "cams.json"),
        "--backend", "torch", "--views", "all", "--out", str(out), "--device", "cpu",
    )  # fmt: skip
    assert result.exit_code == 0, result.output
    assert "torch on cpu" in result.output and "200 splats x 8 views" in result.output
    w = Weights.load(out)
    assert w.contribution.shape == (200, 8)
    assert w.view_names == [f"view_{k:02d}" for k in range(8)]
    direct = TorchRenderer("cpu").contributions(
        read_ply(workdir / "scene.ply"), read_cameras_json(workdir / "cams.json")
    )
    np.testing.assert_array_equal(w.contribution, direct)
    off = oracles["offscreen"]["splats"]
    assert not w.visible[off].any() and np.all(w.sigma_px[off] == 0)
    target = oracles["occluded"]["splat"]
    hidden = w.view_names.index(oracles["occluded"]["hidden_in"])
    assert w.visible[target, hidden] and w.contribution[target, hidden] == 0
    # Footprint axes are consistent: sigma is the geometric mean of the two.
    vis = w.visible
    np.testing.assert_allclose(
        w.sigma_px[vis].astype(np.float32),
        np.sqrt(w.sigma_px_minor[vis].astype(np.float32) * w.sigma_px_major[vis]),
        rtol=2e-3,
    )
    assert w.meta["terms"] == ["contribution"] and w.meta["backend"] == "torch"
    assert w.meta["provenance"]["source"]["sha256"]
    assert w.meta["provenance"]["hardware"]["python"]


def test_default_output_and_view_selection(workdir: Path) -> None:
    result = run("analyze", str(workdir / "scene.ply"), "-c", str(workdir / "cams.json"))
    assert result.exit_code == 0, result.output
    w = Weights.load(workdir / "scene.weights.npz")
    assert len(w.view_names) == 6 and "view_00" not in w.view_names  # train only
    run("analyze", str(workdir / "scene.ply"), "-c", str(workdir / "cams.json"),
        "--views", "3", "-o", str(workdir / "three.npz"))  # fmt: skip
    assert len(Weights.load(workdir / "three.npz").view_names) == 3


def test_analyze_accepts_a_colmap_model(tmp_path: Path, synthetic_dir: Path) -> None:
    colmap = Path(__file__).parent / "fixtures" / "colmap_tiny" / "bin"
    result = run("analyze", str(synthetic_dir / "scene.ply"), "-c", str(colmap),
                 "--views", "all", "-o", str(tmp_path / "w.npz"))  # fmt: skip
    assert result.exit_code == 0, result.output
    assert Weights.load(tmp_path / "w.npz").contribution.shape == (200, 10)


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        (["--terms", "contribution,csf"], "issues/32"),
        (["--terms", "sharpness"], "unknown term"),
        (["--backend", "gsplat"], "issues/23"),
        (["--backend", "msplat"], "issues/24"),
        (["--backend", "vulkan"], "unknown backend"),
        (["--backend", "null"], "--backend torch"),
        (["--views", "some"], "views must be"),
    ],
)
def test_analyze_errors_are_actionable(workdir: Path, extra: list[str], message: str) -> None:
    result = run("analyze", str(workdir / "scene.ply"), "-c", str(workdir / "cams.json"), *extra)
    assert result.exit_code == 1
    assert message in result.output


def test_analyze_reports_missing_inputs(workdir: Path) -> None:
    result = run("analyze", str(workdir / "nope.ply"), "-c", str(workdir / "cams.json"))
    assert result.exit_code == 1 and "nope.ply" in result.output
    unlabelled = [c.__class__(**{**c.__dict__, "split": None}) for c in
                  read_cameras_json(workdir / "cams.json")]  # fmt: skip
    write_cameras(unlabelled, workdir / "plain.json")
    result = run("analyze", str(workdir / "scene.ply"), "-c", str(workdir / "plain.json"),
                 "--views", "eval")  # fmt: skip
    assert result.exit_code == 1 and "selects no cameras" in result.output


# Backend selection ---------------------------------------------------------------------------


def test_auto_selects_torch_without_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(registry, "_cuda_available", lambda: False)
    selection = registry.select_backend(device="cpu")
    assert selection.renderer.name == "torch"


def test_auto_explains_an_installed_but_unimplemented_gsplat(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(registry, "_cuda_available", lambda: True)
    monkeypatch.setattr(registry, "_importable", lambda module: module == "gsplat")
    selection = registry.select_backend(device="cpu")
    assert selection.renderer.name == "torch"
    assert "gsplat is installed" in selection.reason and "issues/23" in selection.reason


# Weights file ---------------------------------------------------------------------------------


def test_weights_round_trip_and_validation(tmp_path: Path) -> None:
    n, v = 4, 3
    w = Weights(
        contribution=np.arange(n * v, dtype=np.float32).reshape(n, v),
        sigma_px=np.ones((n, v), np.float16),
        sigma_px_minor=np.ones((n, v), np.float16),
        sigma_px_major=np.ones((n, v), np.float16),
        visible=np.ones((n, v), bool),
        view_names=["a", "b", "c"],
        meta={"terms": ["contribution"]},
    )
    w.save(tmp_path / "w.npz")
    back = Weights.load(tmp_path / "w.npz")
    np.testing.assert_array_equal(back.contribution, w.contribution)
    assert back.view_names == ["a", "b", "c"] and back.meta["format"] == "psplat-weights"
    assert not list(tmp_path.glob("*.tmp*"))
    with pytest.raises(WeightsFormatError, match="view_names"):
        Weights(w.contribution, w.sigma_px, w.sigma_px_minor, w.sigma_px_major, w.visible, ["a"])
    np.savez(tmp_path / "other.npz", meta=np.array(json.dumps({"format": "x"})))
    with pytest.raises(WeightsFormatError, match="not a psplat weights"):
        Weights.load(tmp_path / "other.npz")
