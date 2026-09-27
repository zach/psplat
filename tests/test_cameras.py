"""P0.2: cams.json schema v1 and the COLMAP / nerfstudio converters."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from psplat.io.cameras import (
    Camera,
    CameraFormatError,
    assign_holdout_split,
    load_cameras,
    look_at,
    read_cameras_json,
    read_colmap,
    read_nerfstudio,
    select_views,
    write_cameras,
)

COLMAP = Path(__file__).parent / "fixtures" / "colmap_tiny"


def _camera(name: str = "a", eye: tuple[float, float, float] = (3.0, 0.0, 1.0)) -> Camera:
    return Camera(
        name=name,
        width=64,
        height=48,
        fx=60.0,
        fy=61.0,
        cx=32.0,
        cy=24.0,
        w2c=look_at(eye, (0.0, 0.0, 0.0)),
    )


# COLMAP -----------------------------------------------------------------------


@pytest.mark.parametrize("encoding", ["bin", "txt"])
def test_colmap_matches_pycolmap(encoding: str) -> None:
    expected = json.loads((COLMAP / "expected.json").read_text())
    cams = read_colmap(COLMAP / encoding)
    assert [c.name for c in cams] == [e["name"] for e in expected]
    for cam, exp in zip(cams, expected, strict=True):
        assert (cam.width, cam.height) == (exp["width"], exp["height"])
        assert (cam.fx, cam.fy, cam.cx, cam.cy) == pytest.approx(
            (exp["fx"], exp["fy"], exp["cx"], exp["cy"]), abs=1e-12
        )
        np.testing.assert_allclose(cam.w2c, np.asarray(exp["w2c"]), atol=1e-12)


def test_colmap_binary_and_text_agree() -> None:
    for a, b in zip(read_colmap(COLMAP / "bin"), read_colmap(COLMAP / "txt"), strict=True):
        assert a.name == b.name and a.split == b.split
        np.testing.assert_allclose(a.w2c, b.w2c, atol=1e-12)


def test_colmap_split_is_every_eighth_by_name() -> None:
    cams = read_colmap(COLMAP / "bin")
    assert [c.name for c in cams] == sorted(c.name for c in cams)
    assert [i for i, c in enumerate(cams) if c.split == "eval"] == [0, 8]
    assert all(c.split == "train" for c in read_colmap(COLMAP / "bin", holdout_every=0))


def test_load_cameras_finds_a_dataset_sparse_dir(tmp_path: Path) -> None:
    model = tmp_path / "garden" / "sparse" / "0"
    model.mkdir(parents=True)
    for f in (COLMAP / "bin").iterdir():
        (model / f.name).write_bytes(f.read_bytes())
    assert len(load_cameras(tmp_path / "garden")) == 10


def _write_colmap_txt(path: Path, model: str, params: str) -> None:
    path.mkdir()
    (path / "cameras.txt").write_text(f"1 {model} 100 80 {params}\n")
    (path / "images.txt").write_text("1 1 0 0 0 0 0 1 1 a.png\n\n")


def test_colmap_distortion_is_refused_unless_allowed(tmp_path: Path) -> None:
    _write_colmap_txt(tmp_path / "d", "OPENCV", "90 91 50 40 0.1 0 0 0")
    with pytest.raises(CameraFormatError, match="image_undistorter"):
        read_colmap(tmp_path / "d")
    with pytest.warns(UserWarning, match="distortion"):
        (cam,) = read_colmap(tmp_path / "d", allow_distortion=True)
    assert (cam.fx, cam.fy, cam.cx, cam.cy) == (90, 91, 50, 40)


def test_colmap_zero_distortion_is_a_pinhole(tmp_path: Path) -> None:
    _write_colmap_txt(tmp_path / "z", "SIMPLE_RADIAL", "90 50 40 0")
    (cam,) = read_colmap(tmp_path / "z")
    assert (cam.fx, cam.fy) == (90, 90)


def test_colmap_fisheye_is_refused_even_when_allowed(tmp_path: Path) -> None:
    _write_colmap_txt(tmp_path / "f", "OPENCV_FISHEYE", "90 91 50 40 0.1 0 0 0")
    with pytest.raises(CameraFormatError, match="fisheye"):
        read_colmap(tmp_path / "f", allow_distortion=True)


# cams.json --------------------------------------------------------------------


def test_cams_json_round_trip_is_exact_and_deterministic(tmp_path: Path) -> None:
    cams = [_camera("a"), replace(_camera("b", (0.0, 3.0, 0.5)), split="eval")]
    write_cameras(cams, tmp_path / "one.json")
    write_cameras(read_cameras_json(tmp_path / "one.json"), tmp_path / "two.json")
    assert (tmp_path / "one.json").read_bytes() == (tmp_path / "two.json").read_bytes()
    back = load_cameras(tmp_path / "one.json")
    for a, b in zip(cams, back, strict=True):
        assert a.to_json() == b.to_json()


def test_cams_json_rejects_duplicates_and_bad_versions(tmp_path: Path) -> None:
    with pytest.raises(CameraFormatError, match="unique"):
        write_cameras([_camera("a"), _camera("a")], tmp_path / "dup.json")
    write_cameras([_camera()], tmp_path / "c.json")
    doc = json.loads((tmp_path / "c.json").read_text())
    doc["version"] = 99
    (tmp_path / "c.json").write_text(json.dumps(doc))
    with pytest.raises(CameraFormatError, match="version"):
        read_cameras_json(tmp_path / "c.json")


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"w2c": np.diag([1.0, 1.0, -1.0, 1.0])}, "rotation"),
        ({"w2c": np.eye(3)}, "shape"),
        ({"width": 0}, "size"),
        ({"fx": -1.0}, "focal"),
    ],
)
def test_camera_validation(change: dict[str, object], message: str) -> None:
    with pytest.raises(CameraFormatError, match=message):
        replace(_camera(), **change)  # type: ignore[arg-type]


# Geometry -------------------------------------------------------------------------


def test_look_at_uses_opencv_axes() -> None:
    cam = _camera(eye=(3.0, 0.0, 0.0))
    target = cam.w2c @ np.array([0.0, 0.0, 0.0, 1.0])
    assert target[:2] == pytest.approx([0.0, 0.0], abs=1e-12) and target[2] == pytest.approx(3.0)
    above = cam.w2c @ np.array([0.0, 0.0, 1.0, 1.0])
    assert above[1] < 0  # world up is image up, which is -y in OpenCV
    np.testing.assert_allclose(cam.position, [3.0, 0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(cam.c2w @ cam.w2c, np.eye(4), atol=1e-12)


def test_downscale_rounds_half_up_and_scales_intrinsics() -> None:
    cam = replace(_camera(), width=4946, height=3286, fx=4000.0, fy=4000.0, cx=2473.0, cy=1643.0)
    small = cam.downscaled(4)
    assert (small.width, small.height) == (1237, 822)  # ImageMagick sizes for bicycle images_4
    assert small.fx == pytest.approx(4000.0 * 1237 / 4946)
    assert small.cy == pytest.approx(1643.0 * 822 / 3286)


# nerfstudio -----------------------------------------------------------------------


def test_nerfstudio_converts_opengl_poses(tmp_path: Path) -> None:
    cams = [_camera(f"images/{i}.png", (3 * np.cos(i), 3 * np.sin(i), 0.5)) for i in range(9)]
    gl_from_cv = np.diag([1.0, -1.0, -1.0, 1.0])
    doc = {
        "camera_model": "OPENCV",
        "w": 64,
        "h": 48,
        "fl_x": 60.0,
        "fl_y": 61.0,
        "cx": 32.0,
        "cy": 24.0,
        "frames": [
            {"file_path": c.name, "transform_matrix": (c.c2w @ gl_from_cv).tolist()} for c in cams
        ],
    }
    (tmp_path / "transforms.json").write_text(json.dumps(doc))
    back = load_cameras(tmp_path / "transforms.json")
    assert [c.name for c in back] == sorted(c.name for c in cams)
    by_name = {c.name: c for c in cams}
    for cam in back:
        np.testing.assert_allclose(cam.w2c, by_name[cam.name].w2c, atol=1e-12)
    assert [c.split for c in back].count("eval") == 2


def test_nerfstudio_refuses_distortion(tmp_path: Path) -> None:
    doc = {
        "w": 64,
        "h": 48,
        "fl_x": 60.0,
        "fl_y": 60.0,
        "cx": 32.0,
        "cy": 24.0,
        "k1": 0.2,
        "frames": [{"file_path": "a.png", "transform_matrix": np.eye(4).tolist()}],
    }
    (tmp_path / "t.json").write_text(json.dumps(doc))
    with pytest.raises(CameraFormatError, match="distortion"):
        read_nerfstudio(tmp_path / "t.json")


# View selection --------------------------------------------------------------------


def test_select_views() -> None:
    cams = assign_holdout_split([_camera(f"{i:02d}") for i in range(16)])
    assert len(select_views(cams, "all")) == 16
    assert [c.name for c in select_views(cams, "eval")] == ["00", "08"]
    assert len(select_views(cams, "train")) == 14
    assert [c.name for c in select_views(cams, 3)] == ["00", "08", "15"]
    assert len(select_views(cams, "4")) == 4
    unlabelled = [_camera(f"{i}") for i in range(3)]
    assert len(select_views(unlabelled, "train")) == 3
    with pytest.raises(ValueError, match="views"):
        select_views(cams, "some")


def test_cameras_command_converts_colmap(tmp_path: Path) -> None:
    from typer.testing import CliRunner

    from psplat.cli import app

    out = tmp_path / "cams.json"
    result = CliRunner().invoke(
        app, ["cameras", str(COLMAP / "bin"), "--out", str(out), "--downscale", "2"]
    )
    assert result.exit_code == 0, result.output
    assert "10 views (8 train, 2 eval)" in result.output
    cams = read_cameras_json(out)
    assert (cams[1].width, cams[1].height) == (320, 240)


def test_cameras_command_reports_bad_input(tmp_path: Path) -> None:
    from typer.testing import CliRunner

    from psplat.cli import app

    result = CliRunner().invoke(app, ["cameras", str(tmp_path)])
    assert result.exit_code == 1
    assert "no COLMAP model" in result.output
