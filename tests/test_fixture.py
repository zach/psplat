"""P0.4: the synthetic fixture and its oracles.

The oracles are properties known by construction (see
tests/fixtures/synthetic/generate.py). They are checked here with the torch
oracle backend, and later backends are held to the same answers.
"""

from __future__ import annotations

import importlib.util
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from psplat.io.cameras import read_cameras_json
from psplat.io.ply import read_ply
from psplat.render.project import SceneTensors, project
from psplat.render.torch_backend import TorchRenderer


@pytest.fixture(scope="module")
def analysed(synthetic_dir: Path) -> dict[str, Any]:
    scene = read_ply(synthetic_dir / "scene.ply")
    cams = read_cameras_json(synthetic_dir / "cams.json")
    start = time.perf_counter()
    contrib = TorchRenderer("cpu").contributions(scene, cams)
    seconds = time.perf_counter() - start
    splats = SceneTensors.from_scene(scene)
    visible = np.stack([project(splats, cam).visible.numpy() for cam in cams], axis=1)
    return {
        "scene": scene,
        "names": [c.name for c in cams],
        "contrib": contrib,
        "visible": visible,
        "seconds": seconds,
    }


def test_fixture_shape(analysed: dict[str, Any], synthetic_dir: Path) -> None:
    scene = analysed["scene"]
    assert scene.count == 200 and scene.sh_degree == 3
    cams = read_cameras_json(synthetic_dir / "cams.json")
    assert len(cams) == 8
    assert [c.name for c in cams if c.split == "eval"] == ["view_00", "view_04"]


def test_offscreen_splats_are_invisible_everywhere(
    analysed: dict[str, Any], oracles: dict[str, Any]
) -> None:
    idx = oracles["offscreen"]["splats"]
    assert not analysed["visible"][idx].any()
    assert np.all(analysed["contrib"][idx] == 0)


def test_occluded_target_is_hidden_in_one_view_and_seen_opposite(
    analysed: dict[str, Any], oracles: dict[str, Any]
) -> None:
    o, names = oracles["occluded"], analysed["names"]
    hidden, seen = names.index(o["hidden_in"]), names.index(o["visible_in"])
    target = o["splat"]
    assert analysed["contrib"][target, hidden] == 0.0
    assert analysed["visible"][target, hidden]  # in frame, just behind the occluders
    assert analysed["contrib"][target, seen] > 0.5


def test_overlap_front_beats_back(analysed: dict[str, Any], oracles: dict[str, Any]) -> None:
    o = oracles["overlap"]
    view = analysed["names"].index(o["view"])
    front, back = analysed["contrib"][o["front"], view], analysed["contrib"][o["back"], view]
    assert front > back > 0


def test_every_cloud_splat_contributes_somewhere(analysed: dict[str, Any]) -> None:
    assert np.all(analysed["contrib"][:188].sum(axis=1) > 0)


def test_fixture_analyse_is_fast_on_cpu(analysed: dict[str, Any]) -> None:
    assert analysed["seconds"] < 30.0  # handoff P0.6 acceptance


def test_generator_reproduces_the_checked_in_files(tmp_path: Path, synthetic_dir: Path) -> None:
    # Compared numerically, not byte for byte: sin and cos may differ in the
    # last bit across platforms. The checked-in files are the source of truth.
    spec = importlib.util.spec_from_file_location("generate", synthetic_dir / "generate.py")
    assert spec is not None and spec.loader is not None
    generate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(generate)
    generate.main(tmp_path)
    fresh, kept = read_ply(tmp_path / "scene.ply"), read_ply(synthetic_dir / "scene.ply")
    for field in ("means", "scales", "quats", "opacities", "sh0", "shN"):
        np.testing.assert_allclose(
            getattr(fresh, field), getattr(kept, field), rtol=1e-6, atol=1e-7
        )
    for a, b in zip(
        read_cameras_json(tmp_path / "cams.json"),
        read_cameras_json(synthetic_dir / "cams.json"),
        strict=True,
    ):
        np.testing.assert_allclose(a.w2c, b.w2c, atol=1e-12)
    assert json.loads((tmp_path / "oracles.json").read_text()) == json.loads(
        (synthetic_dir / "oracles.json").read_text()
    )
