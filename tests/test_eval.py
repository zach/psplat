"""P1.6: ``psplat eval`` on the synthetic fixture with the torch backend."""

from __future__ import annotations

import math
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import pytest
from typer.testing import CliRunner

from psplat.cli import app
from psplat.eval import evaluate as ev
from psplat.eval import metrics as m
from psplat.io import manifest as mf
from psplat.io.cameras import read_cameras_json, select_views
from psplat.io.ply import read_ply
from psplat.pack.pack import ply_prefix_bytes
from psplat.render.torch_backend import TorchRenderer

runner = CliRunner()
CHEAP = "psnr,ssim,flip"  # no downloads


@pytest.fixture(scope="module")
def ordered(tmp_path_factory: pytest.TempPathFactory, synthetic_dir: Path) -> Path:
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
    for name in ("scene.ordered.ply", "scene.psplat.json", "cams.json"):
        shutil.copy(ordered / name, tmp_path / name)
    return tmp_path


def run_eval(workdir: Path, *args: str) -> Any:
    return runner.invoke(app, [
        "eval", str(workdir / "scene.ordered.ply"), "--cameras", str(workdir / "cams.json"),
        "--backend", "torch", "--device", "cpu", *args,
    ])  # fmt: skip


def test_eval_writes_one_row_per_budget_and_view(workdir: Path) -> None:
    result = run_eval(workdir, "--metrics", CHEAP, "--trainer", "synthetic")
    assert result.exit_code == 0, result.output
    table = pq.read_table(workdir / "scene.results.parquet").to_pylist()
    manifest = mf.read(workdir / "scene.psplat.json")
    cuts = [(c["frac"], c["index"]) for c in manifest["cuts"]]
    views = ["view_00", "view_04"]  # the fixture's eval split
    assert len(table) == len(cuts) * len(views)
    assert {(r["budget_frac"], r["budget_splats"]) for r in table} == set(cuts)
    assert {r["view"] for r in table} == set(views)
    first = table[0]
    assert first["scene"] == "scene" and first["trainer"] == "synthetic"
    assert first["ordering"] == "contribution" and first["terms"] == "contribution"
    assert first["format"] == "ply" and first["reference"] == "full"
    assert first["source_sha256"] == manifest["source_sha256"]
    assert first["lpips"] is None and first["cvvdp"] is None and first["fps_web"] is None
    sizes = ply_prefix_bytes(workdir / "scene.ordered.ply", [k for _, k in cuts])
    assert all(r["bytes"] == sizes[r["budget_splats"]] for r in table)
    # Against the whole scene, the full budget is perfect and less is worse.
    for view in views:
        rows = sorted((r for r in table if r["view"] == view), key=lambda r: r["budget_splats"])
        full = rows[-1]
        assert full["psnr"] == math.inf and full["ssim"] == 1.0 and full["flip"] == 0.0
        assert all(r["psnr"] < 60 and r["flip"] > 0 for r in rows[:-1])
        assert rows[0]["flip"] > rows[-2]["flip"]
    assert "100%" in result.output and "wrote" in result.output


def test_rows_match_metrics_on_direct_renders(workdir: Path) -> None:
    result = run_eval(workdir, "--metrics", "psnr,flip", "--budgets", "25", "--views", "1")
    assert result.exit_code == 0, result.output
    row = pq.read_table(workdir / "scene.results.parquet").to_pylist()[0]
    scene = read_ply(workdir / "scene.ordered.ply")
    cam = select_views(read_cameras_json(workdir / "cams.json"), 1)[0]
    oracle = TorchRenderer("cpu")
    full = oracle.render(scene, cam).image
    part = oracle.render(scene.take(np.arange(50)), cam).image
    assert row["budget_splats"] == 50 and row["view"] == cam.name
    assert row["psnr"] == pytest.approx(m.psnr(part, full), rel=1e-6)
    assert row["flip"] == pytest.approx(m.flip(part, full), rel=1e-6)


def test_write_manifest_stores_finite_means(workdir: Path) -> None:
    result = run_eval(workdir, "--metrics", CHEAP, "--write-manifest")
    assert result.exit_code == 0, result.output
    manifest = mf.read(workdir / "scene.psplat.json")
    table = pq.read_table(workdir / "scene.results.parquet").to_pylist()
    for cut in manifest["cuts"]:
        rows = [r for r in table if r["budget_splats"] == cut["index"]]
        assert cut["flip"] == pytest.approx(np.mean([r["flip"] for r in rows]), abs=1e-6)
        if cut["frac"] == 1.0:
            assert "psnr" not in cut  # inf has no JSON form
        else:
            assert cut["psnr"] == pytest.approx(np.mean([r["psnr"] for r in rows]), abs=1e-5)
    assert manifest["provenance"]["eval"]["reference"] == "full"


def test_compare_with_photographs(workdir: Path) -> None:
    # Photographs here are the full renders saved as 8-bit PNG, so the 100%
    # budget is close to, but not exactly, the reference.
    from PIL import Image

    photos = workdir / "photos"
    photos.mkdir()
    scene = read_ply(workdir / "scene.ordered.ply")
    oracle = TorchRenderer("cpu")
    eval_views = select_views(read_cameras_json(workdir / "cams.json"), "eval")
    for cam in eval_views:
        img = np.clip(oracle.render(scene, cam).image, 0, 1)
        Image.fromarray(np.rint(img * 255).astype(np.uint8)).save(photos / f"{cam.name}.png")
    result = run_eval(workdir, "--metrics", CHEAP, "--images", str(photos), "--budgets", "50,100")
    assert result.exit_code == 0, result.output
    table = pq.read_table(workdir / "scene.results.parquet").to_pylist()
    full = [r for r in table if r["budget_frac"] == 1.0]
    assert all(r["reference"] == "images" for r in table)
    assert all(45 < r["psnr"] < math.inf for r in full)  # 8-bit rounding only

    (photos / f"{eval_views[0].name}.png").unlink()
    result = run_eval(workdir, "--metrics", "psnr", "--images", str(photos))
    assert result.exit_code == 1 and "no image for camera" in result.output


def test_load_image_reads_png_and_reports_bad_files(tmp_path: Path) -> None:
    from PIL import Image

    pixels = np.array([[[0, 128, 255], [255, 0, 0]]], np.uint8)
    Image.fromarray(pixels).save(tmp_path / "a.png")
    np.testing.assert_allclose(ev.load_image(tmp_path / "a.png"), pixels / 255.0, rtol=1e-6)
    Image.fromarray(pixels[..., 0]).save(tmp_path / "grey.png")  # greyscale becomes RGB
    assert ev.load_image(tmp_path / "grey.png").shape == (1, 2, 3)
    (tmp_path / "bad.png").write_bytes(b"not an image")
    with pytest.raises(ev.ImageError, match="cannot read image"):
        ev.load_image(tmp_path / "bad.png")


def test_photograph_of_the_wrong_size_is_refused(workdir: Path) -> None:
    from PIL import Image

    photos = workdir / "photos"
    photos.mkdir()
    for cam in select_views(read_cameras_json(workdir / "cams.json"), "eval"):
        Image.new("RGB", (12, 10)).save(photos / f"{cam.name}.jpg")
    result = run_eval(workdir, "--metrics", "psnr", "--images", str(photos))
    assert result.exit_code == 1 and "matching resolution" in result.output


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (["--metrics", "sharpness"], "unknown metric"),
        (["--budgets", "0,50"], "--budgets"),
        (["--backend", "null", "--metrics", "psnr"], "null backend cannot render"),
    ],
)
def test_eval_refusals(workdir: Path, args: list[str], message: str) -> None:
    result = run_eval(workdir, *args)
    assert result.exit_code == 1
    assert message in result.output


def test_cameras_without_an_eval_split_say_what_to_do(workdir: Path) -> None:
    from dataclasses import replace

    from psplat.io.cameras import write_cameras

    cams = read_cameras_json(workdir / "cams.json")
    write_cameras([replace(c, split=None) for c in cams], workdir / "cams.json")
    result = run_eval(workdir, "--metrics", "psnr")
    assert result.exit_code == 1 and "--views all" in result.output


def test_eval_without_a_manifest_uses_default_budgets(workdir: Path) -> None:
    (workdir / "scene.psplat.json").unlink()
    result = run_eval(workdir, "--metrics", "psnr")
    assert result.exit_code == 0, result.output
    table = pq.read_table(workdir / "scene.results.parquet").to_pylist()
    assert sorted({r["budget_frac"] for r in table}) == [0.05, 0.1, 0.25, 0.5, 1.0]
    assert table[0]["ordering"] == "file order"
    result = run_eval(workdir, "--metrics", "psnr", "--write-manifest")
    assert result.exit_code == 1 and "needs a manifest" in result.output


def test_budgets_for_matches_the_manifest_rule() -> None:
    got = ev.budgets_for(200, [50, 5, 100, 10])
    assert [(b.frac, b.splats) for b in got] == [(0.05, 10), (0.1, 20), (0.5, 100), (1.0, 200)]
    cuts = mf.cuts_for(3_512_000, [5, 10, 25, 50, 75])
    assert [b.splats for b in ev.budgets_for(3_512_000, [5, 10, 25, 50, 75, 100])] == [
        c["index"] for c in cuts
    ]
    # Two percentages that land on the same splat count are one budget.
    assert len(ev.budgets_for(3, [10, 20])) == 1


def test_summary_keeps_inf_and_averages_the_rest() -> None:
    rows = [
        {"budget_frac": 0.5, "budget_splats": 5, "psnr": 20.0, "flip": 0.1},
        {"budget_frac": 0.5, "budget_splats": 5, "psnr": 30.0, "flip": 0.3},
        {"budget_frac": 1.0, "budget_splats": 10, "psnr": math.inf, "flip": 0.0},
    ]
    summary = ev.summarise(rows, ["psnr", "flip"])
    assert summary[0]["psnr"] == 25.0 and summary[0]["flip"] == pytest.approx(0.2)
    assert summary[1]["psnr"] == math.inf
