"""Interop with PlayCanvas splat-transform, an independent PLY reader and writer.

Opt-in, because it fetches splat-transform through npx: set PSPLAT_INTEROP=1.
CI runs it on Linux. Locally::

    PSPLAT_INTEROP=1 pytest tests/test_interop_splat_transform.py
"""

from __future__ import annotations

import csv
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from psplat.io.ply import read_ply, write_ply
from psplat.io.scene import Scene

SPLAT_TRANSFORM = "@playcanvas/splat-transform@3.4.2"

pytestmark = pytest.mark.skipif(
    os.environ.get("PSPLAT_INTEROP") != "1" or shutil.which("npx") is None,
    reason="set PSPLAT_INTEROP=1 with Node's npx on PATH to run",
)


def _splat_transform(*args: str) -> None:
    npx = shutil.which("npx")
    assert npx is not None
    subprocess.run([npx, "--yes", SPLAT_TRANSFORM, "-w", *args], check=True, capture_output=True)


def _scene(n: int = 64) -> Scene:
    rng = np.random.default_rng(11)
    f = np.float32
    return Scene(
        means=rng.normal(size=(n, 3)).astype(f),
        scales=(rng.normal(size=(n, 3)) - 3).astype(f),
        quats=np.abs(rng.normal(size=(n, 4))).astype(f) + 0.1,
        opacities=rng.normal(size=n).astype(f),
        sh0=rng.normal(size=(n, 3)).astype(f),
        shN=(rng.normal(size=(n, 15, 3)) * 0.1).astype(f),
    )


def test_their_ply_matches_ours_bit_for_bit_and_keeps_order(tmp_path: Path) -> None:
    ours = _scene()
    write_ply(ours, tmp_path / "ours.ply")
    _splat_transform(str(tmp_path / "ours.ply"), str(tmp_path / "theirs.ply"))
    theirs = read_ply(tmp_path / "theirs.ply")
    # splat-transform writes a different property order, so this also checks
    # that the reader goes by name, not position.
    assert [n for n, _ in (theirs.ply_layout.properties if theirs.ply_layout else ())][:4] != [
        "x",
        "y",
        "z",
        "nx",
    ]
    for field in ("means", "scales", "quats", "opacities", "sh0", "shN"):
        np.testing.assert_array_equal(getattr(theirs, field), getattr(ours, field), err_msg=field)


def test_their_reading_of_our_file_matches(tmp_path: Path) -> None:
    ours = _scene()
    write_ply(ours, tmp_path / "ours.ply")
    _splat_transform(str(tmp_path / "ours.ply"), str(tmp_path / "theirs.csv"))
    with (tmp_path / "theirs.csv").open(newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == ours.count
    col = {k: np.array([float(r[k]) for r in rows], np.float32) for k in rows[0]}
    np.testing.assert_allclose(col["x"], ours.means[:, 0], rtol=1e-6)
    np.testing.assert_allclose(col["opacity"], ours.opacities, rtol=1e-6)
    # f_rest is channel-major: green (channel 1) coefficient 2 is f_rest_17.
    np.testing.assert_allclose(col["f_rest_17"], ours.shN[:, 2, 1], rtol=1e-5, atol=1e-7)
