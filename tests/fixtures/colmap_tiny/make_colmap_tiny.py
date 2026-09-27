"""Regenerate the tiny COLMAP model used by tests/test_cameras.py.

This is written with pycolmap, the reference implementation, so the psplat
reader is tested against files it did not write itself. pycolmap is not a
psplat dependency; run this in a scratch environment:

    uv venv -p 3.12 /tmp/colmapenv
    uv pip install --python /tmp/colmapenv pycolmap numpy
    /tmp/colmapenv/bin/python tests/fixtures/colmap_tiny/make_colmap_tiny.py

It writes ``bin/`` and ``txt/`` (the same model in both encodings) and
``expected.json``, which records each image's world-to-camera matrix and
pinhole intrinsics as pycolmap itself reports them.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pycolmap

HERE = Path(__file__).resolve().parent
rng = np.random.default_rng(1234)

rec = pycolmap.Reconstruction()
cameras = [
    pycolmap.Camera(
        model="PINHOLE", width=640, height=480, params=[500.0, 510.0, 320.5, 239.5], camera_id=1
    ),
    pycolmap.Camera(
        model="SIMPLE_PINHOLE", width=320, height=200, params=[250.0, 160.0, 100.0], camera_id=2
    ),
]
for cam in cameras:
    rec.add_camera_with_trivial_rig(cam)

# Names deliberately out of order, one with a space, so sorting and the
# every-eighth split are exercised. Ten images: indices 0 and 8 are eval.
names = [f"img_{i:02d}.png" for i in (5, 3, 9, 0, 1, 7, 2, 8, 6)] + ["img 10 with space.png"]
for image_id, name in enumerate(names, start=1):
    axis_angle = rng.normal(size=3) * 0.7
    translation = rng.normal(size=3) * 2.0
    pose = pycolmap.Rigid3d(pycolmap.Rotation3d(axis_angle), translation)
    image = pycolmap.Image(name=name, camera_id=1 + image_id % 2, image_id=image_id)
    rec.add_image_with_trivial_frame(image, pose)

for sub in ("bin", "txt"):
    (HERE / sub).mkdir(exist_ok=True)
rec.write_binary(str(HERE / "bin"))
rec.write_text(str(HERE / "txt"))

expected = []
for image in sorted(rec.images.values(), key=lambda im: im.name):
    cam = rec.cameras[image.camera_id]
    w2c = np.eye(4)
    w2c[:3, :] = image.cam_from_world().matrix()
    K = cam.calibration_matrix()
    expected.append(
        {
            "name": image.name,
            "width": cam.width,
            "height": cam.height,
            "fx": K[0, 0],
            "fy": K[1, 1],
            "cx": K[0, 2],
            "cy": K[1, 2],
            "w2c": w2c.tolist(),
        }
    )
(HERE / "expected.json").write_text(json.dumps(expected, indent=2) + "\n")
print(f"wrote {len(expected)} images, pycolmap {pycolmap.__version__}")
