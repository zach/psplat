"""Cameras: the ``cams.json`` schema v1 and converters into it.

Everything in psplat reads cameras through this module. COLMAP and
nerfstudio formats are converted here and nowhere else (decision D1).

Schema v1::

    {
      "format": "psplat-cameras",
      "version": 1,
      "convention": "opencv",
      "views": [
        {"name": "DSC07956.JPG", "width": 1237, "height": 822,
         "fx": 1150.1, "fy": 1150.1, "cx": 618.5, "cy": 411.0,
         "w2c": [[r00, r01, r02, tx], [r10, r11, r12, ty],
                 [r20, r21, r22, tz], [0, 0, 0, 1]],
         "split": "train"}
      ]
    }

``w2c`` maps world points into the camera frame. The camera convention is
OpenCV's, the same as COLMAP's: +x right, +y down, +z forward. ``split`` is
optional and is either ``"train"`` or ``"eval"``. Pixel ``(u, v)`` of a camera
point ``(x, y, z)`` is ``(fx * x / z + cx, fy * y / z + cy)``, with pixel
centres at half-integer coordinates.
"""

from __future__ import annotations

import json
import math
import struct
import warnings
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, BinaryIO, Literal

import numpy as np
from numpy.typing import NDArray

F64 = NDArray[np.float64]
Split = Literal["train", "eval"]

FORMAT = "psplat-cameras"
VERSION = 1
# Standard 3DGS / gsplat evaluation split: after sorting views by name,
# every eighth view (index % 8 == 0) is held out for evaluation.
DEFAULT_HOLDOUT = 8


class CameraFormatError(ValueError):
    """A camera file could not be read or failed validation."""


@dataclass(frozen=True)
class Camera:
    """One pinhole view. See the module docstring for conventions."""

    name: str
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    w2c: F64
    split: Split | None = None

    def __post_init__(self) -> None:
        w2c = np.asarray(self.w2c, dtype=np.float64)
        object.__setattr__(self, "w2c", w2c)
        if w2c.shape != (4, 4):
            raise CameraFormatError(f"{self.name}: w2c has shape {w2c.shape}, expected (4, 4)")
        if not np.allclose(w2c[3], [0.0, 0.0, 0.0, 1.0], atol=1e-9):
            raise CameraFormatError(f"{self.name}: w2c last row must be [0, 0, 0, 1]")
        rot = w2c[:3, :3]
        if not np.allclose(rot @ rot.T, np.eye(3), atol=1e-5) or np.linalg.det(rot) < 0:
            raise CameraFormatError(f"{self.name}: w2c rotation is not a proper rotation")
        if self.width <= 0 or self.height <= 0:
            raise CameraFormatError(f"{self.name}: image size must be positive")
        if self.fx <= 0 or self.fy <= 0:
            raise CameraFormatError(f"{self.name}: focal lengths must be positive")
        if self.split not in (None, "train", "eval"):
            raise CameraFormatError(f"{self.name}: split must be 'train', 'eval' or absent")

    @property
    def c2w(self) -> F64:
        return np.linalg.inv(self.w2c)

    @property
    def position(self) -> F64:
        """Camera centre in world coordinates."""
        rot, t = self.w2c[:3, :3], self.w2c[:3, 3]
        return np.asarray(-rot.T @ t, dtype=np.float64)

    @property
    def K(self) -> F64:
        return np.array(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )

    def resized(self, width: int, height: int) -> Camera:
        """The same view rendered at a different resolution."""
        sx, sy = width / self.width, height / self.height
        return replace(
            self,
            width=width,
            height=height,
            fx=self.fx * sx,
            fy=self.fy * sy,
            cx=self.cx * sx,
            cy=self.cy * sy,
        )

    def downscaled(self, factor: float) -> Camera:
        """Downscale by ``factor``, rounding sizes half up like ImageMagick does."""
        width = max(1, math.floor(self.width / factor + 0.5))
        height = max(1, math.floor(self.height / factor + 0.5))
        return self.resized(width, height)

    def to_json(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "name": self.name,
            "width": self.width,
            "height": self.height,
            "fx": self.fx,
            "fy": self.fy,
            "cx": self.cx,
            "cy": self.cy,
            "w2c": [[float(v) for v in row] for row in self.w2c],
        }
        if self.split is not None:
            out["split"] = self.split
        return out


# cams.json ------------------------------------------------------------------


def write_cameras(cameras: Sequence[Camera], path: str | Path) -> None:
    """Write ``cams.json``. Output is deterministic for a given input."""
    _check_unique_names(cameras)
    doc = {
        "format": FORMAT,
        "version": VERSION,
        "convention": "opencv",
        "views": [cam.to_json() for cam in cameras],
    }
    Path(path).write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")


def read_cameras_json(path: str | Path) -> list[Camera]:
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    if doc.get("format") != FORMAT:
        raise CameraFormatError(f"{path}: not a psplat cameras file")
    if doc.get("version") != VERSION:
        raise CameraFormatError(f"{path}: cameras version {doc.get('version')} is not supported")
    if doc.get("convention", "opencv") != "opencv":
        raise CameraFormatError(f"{path}: only the opencv convention is supported")
    try:
        cameras = [
            Camera(
                name=str(v["name"]),
                width=int(v["width"]),
                height=int(v["height"]),
                fx=float(v["fx"]),
                fy=float(v["fy"]),
                cx=float(v["cx"]),
                cy=float(v["cy"]),
                w2c=np.asarray(v["w2c"], dtype=np.float64),
                split=v.get("split"),
            )
            for v in doc["views"]
        ]
    except KeyError as exc:
        raise CameraFormatError(f"{path}: a view is missing field {exc}") from None
    _check_unique_names(cameras)
    return cameras


def load_cameras(
    path: str | Path,
    *,
    allow_distortion: bool = False,
    holdout_every: int = DEFAULT_HOLDOUT,
) -> list[Camera]:
    """Load cameras from ``cams.json``, a COLMAP model, or a nerfstudio file.

    ``path`` may be a ``cams.json``, a nerfstudio ``transforms.json``, a COLMAP
    model directory holding ``cameras.bin`` / ``images.bin`` (or the ``.txt``
    pair), or a dataset directory containing ``sparse/0``. Views read from
    COLMAP or nerfstudio are given the standard every-eighth evaluation split
    (``holdout_every=0`` marks them all train); a ``cams.json`` keeps its own.
    """
    p = Path(path)
    if p.is_dir():
        return read_colmap(p, allow_distortion=allow_distortion, holdout_every=holdout_every)
    if p.suffix.lower() == ".json":
        doc = json.loads(p.read_text(encoding="utf-8"))
        if doc.get("format") == FORMAT:
            return read_cameras_json(p)
        if "frames" in doc:
            return read_nerfstudio(
                p, allow_distortion=allow_distortion, holdout_every=holdout_every
            )
        raise CameraFormatError(f"{p}: neither a psplat cameras file nor a nerfstudio file")
    raise CameraFormatError(f"{p}: expected cams.json, transforms.json or a COLMAP directory")


def select_views(cameras: Sequence[Camera], views: str | int) -> list[Camera]:
    """Pick the views an analysis runs over.

    ``"train"`` keeps views whose split is ``train`` (or all views if none are
    labelled), ``"eval"`` keeps evaluation views, ``"all"`` keeps everything,
    and an integer ``n`` keeps ``n`` views spread evenly across the set.
    """
    cams = list(cameras)
    if isinstance(views, int) or (isinstance(views, str) and views.isdigit()):
        n = int(views)
        if n <= 0:
            raise ValueError("number of views must be positive")
        if n >= len(cams):
            return cams
        picks = np.linspace(0, len(cams) - 1, n).round().astype(int)
        return [cams[i] for i in picks]
    if views == "all":
        return cams
    if views in ("train", "eval"):
        if all(c.split is None for c in cams):
            return cams if views == "train" else []
        return [c for c in cams if c.split == views]
    raise ValueError(f"views must be 'train', 'eval', 'all' or a count, not {views!r}")


def assign_holdout_split(cameras: Iterable[Camera], every: int = DEFAULT_HOLDOUT) -> list[Camera]:
    """Sort by name and mark every ``every``-th view (index % every == 0) as eval."""
    ordered = sorted(cameras, key=lambda c: c.name)
    return [
        replace(cam, split="eval" if every > 0 and i % every == 0 else "train")
        for i, cam in enumerate(ordered)
    ]


def _check_unique_names(cameras: Sequence[Camera]) -> None:
    names = [c.name for c in cameras]
    if len(set(names)) != len(names):
        raise CameraFormatError("camera names must be unique")


# COLMAP -----------------------------------------------------------------------

# model id -> (name, number of parameters)
COLMAP_MODELS: dict[int, tuple[str, int]] = {
    0: ("SIMPLE_PINHOLE", 3),
    1: ("PINHOLE", 4),
    2: ("SIMPLE_RADIAL", 4),
    3: ("RADIAL", 5),
    4: ("OPENCV", 8),
    5: ("OPENCV_FISHEYE", 8),
    6: ("FULL_OPENCV", 12),
    7: ("FOV", 5),
    8: ("SIMPLE_RADIAL_FISHEYE", 4),
    9: ("RADIAL_FISHEYE", 5),
    10: ("THIN_PRISM_FISHEYE", 12),
}
_MODEL_BY_NAME = {name: (mid, n) for mid, (name, n) in COLMAP_MODELS.items()}
_FISHEYE = {
    "OPENCV_FISHEYE",
    "SIMPLE_RADIAL_FISHEYE",
    "RADIAL_FISHEYE",
    "THIN_PRISM_FISHEYE",
    "FOV",
}


_ColmapImage = tuple[str, tuple[float, ...], tuple[float, ...], int]


@dataclass(frozen=True)
class _ColmapCamera:
    model: str
    width: int
    height: int
    params: tuple[float, ...]


def _intrinsics(cam: _ColmapCamera, allow_distortion: bool) -> tuple[float, float, float, float]:
    p = cam.params
    if cam.model == "SIMPLE_PINHOLE":
        return p[0], p[0], p[1], p[2]
    if cam.model == "PINHOLE":
        return p[0], p[1], p[2], p[3]
    if cam.model in _FISHEYE:
        raise CameraFormatError(
            f"COLMAP camera model {cam.model} is a fisheye model, which psplat does not "
            "support. Undistort first: colmap image_undistorter"
        )
    pinhole_params = 3 if cam.model in ("SIMPLE_RADIAL", "RADIAL") else 4
    distortion = p[pinhole_params:]
    if all(k == 0.0 for k in distortion):
        # A distortion model with zero coefficients is a pinhole camera.
        allow_distortion = True
    elif not allow_distortion:
        raise CameraFormatError(
            f"COLMAP camera model {cam.model} has lens distortion. Undistort the images "
            "first (colmap image_undistorter) or pass allow_distortion to ignore it."
        )
    else:
        warnings.warn(f"ignoring lens distortion of COLMAP {cam.model} camera", stacklevel=3)
    if cam.model in ("SIMPLE_RADIAL", "RADIAL"):
        return p[0], p[0], p[1], p[2]
    return p[0], p[1], p[2], p[3]  # OPENCV, FULL_OPENCV


def _qvec_to_rotation(qw: float, qx: float, qy: float, qz: float) -> F64:
    q = np.array([qw, qx, qy, qz], dtype=np.float64)
    w, x, y, z = q / np.linalg.norm(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _find_colmap_model(path: Path) -> Path:
    for candidate in (path, path / "sparse" / "0", path / "sparse"):
        if (candidate / "cameras.bin").exists() or (candidate / "cameras.txt").exists():
            return candidate
    raise CameraFormatError(f"{path}: no COLMAP model (cameras.bin or cameras.txt) found")


def read_colmap(
    path: str | Path,
    *,
    allow_distortion: bool = False,
    holdout_every: int = DEFAULT_HOLDOUT,
) -> list[Camera]:
    """Read a COLMAP sparse model (binary or text) into cameras.

    Views are sorted by image name and given the every-``holdout_every``
    evaluation split. Pass ``holdout_every=0`` to mark every view as train.
    """
    model = _find_colmap_model(Path(path))
    if (model / "cameras.bin").exists():
        cams = _read_colmap_cameras_bin(model / "cameras.bin")
        images = _read_colmap_images_bin(model / "images.bin")
    else:
        cams = _read_colmap_cameras_txt(model / "cameras.txt")
        images = _read_colmap_images_txt(model / "images.txt")

    out = []
    for name, qvec, tvec, camera_id in images:
        if camera_id not in cams:
            raise CameraFormatError(f"image {name} refers to missing camera {camera_id}")
        ccam = cams[camera_id]
        fx, fy, cx, cy = _intrinsics(ccam, allow_distortion)
        w2c = np.eye(4, dtype=np.float64)
        w2c[:3, :3] = _qvec_to_rotation(*qvec)
        w2c[:3, 3] = tvec
        out.append(
            Camera(
                name=name,
                width=ccam.width,
                height=ccam.height,
                fx=fx,
                fy=fy,
                cx=cx,
                cy=cy,
                w2c=w2c,
            )
        )
    return assign_holdout_split(out, holdout_every)


def _read(f: BinaryIO, fmt: str) -> tuple[Any, ...]:
    size = struct.calcsize(fmt)
    data = f.read(size)
    if len(data) != size:
        raise CameraFormatError("unexpected end of COLMAP binary file")
    return struct.unpack(fmt, data)


def _read_colmap_cameras_bin(path: Path) -> dict[int, _ColmapCamera]:
    cams = {}
    with path.open("rb") as f:
        (count,) = _read(f, "<Q")
        for _ in range(count):
            camera_id, model_id, width, height = _read(f, "<iiQQ")
            if model_id not in COLMAP_MODELS:
                raise CameraFormatError(f"{path}: unknown COLMAP camera model id {model_id}")
            model, nparams = COLMAP_MODELS[model_id]
            params = _read(f, f"<{nparams}d")
            cams[camera_id] = _ColmapCamera(model, int(width), int(height), params)
    return cams


def _read_colmap_images_bin(
    path: Path,
) -> list[_ColmapImage]:
    images: list[_ColmapImage] = []
    with path.open("rb") as f:
        (count,) = _read(f, "<Q")
        for _ in range(count):
            _image_id, qw, qx, qy, qz, tx, ty, tz, camera_id = _read(f, "<idddddddi")
            name = bytearray()
            while (ch := f.read(1)) not in (b"\x00", b""):
                name += ch
            (num_points,) = _read(f, "<Q")
            f.seek(num_points * 24, 1)  # x, y (double) and point3D id (int64) each
            images.append((name.decode("utf-8"), (qw, qx, qy, qz), (tx, ty, tz), camera_id))
    return images


def _data_lines(path: Path) -> list[str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [ln for ln in lines if not ln.lstrip().startswith("#")]


def _read_colmap_cameras_txt(path: Path) -> dict[int, _ColmapCamera]:
    cams = {}
    for line in _data_lines(path):
        parts = line.split()
        if not parts:
            continue
        camera_id, model, width, height = int(parts[0]), parts[1], int(parts[2]), int(parts[3])
        if model not in _MODEL_BY_NAME:
            raise CameraFormatError(f"{path}: unknown COLMAP camera model {model}")
        params = tuple(float(v) for v in parts[4:])
        if len(params) != _MODEL_BY_NAME[model][1]:
            raise CameraFormatError(f"{path}: camera {camera_id} has the wrong parameter count")
        cams[camera_id] = _ColmapCamera(model, width, height, params)
    return cams


def _read_colmap_images_txt(
    path: Path,
) -> list[_ColmapImage]:
    # Images come in pairs of lines: the pose line, then the 2D points line,
    # which may be empty. Comment lines are not part of the pairing.
    lines = _data_lines(path)
    images: list[_ColmapImage] = []
    for number, pose_line in enumerate(lines[0::2]):
        parts = pose_line.split(maxsplit=9)
        if not parts and number == len(lines[0::2]) - 1:
            continue  # trailing blank line
        if len(parts) < 10:
            raise CameraFormatError(f"{path}: malformed image line {pose_line!r}")
        qvec = tuple(float(v) for v in parts[1:5])
        tvec = tuple(float(v) for v in parts[5:8])
        images.append((parts[9].strip(), qvec, tvec, int(parts[8])))
    return images


# nerfstudio -------------------------------------------------------------------

_NS_DISTORTION = ("k1", "k2", "k3", "k4", "p1", "p2")
# nerfstudio stores camera-to-world in the OpenGL convention (+y up, +z back).
_GL_TO_CV = np.diag([1.0, -1.0, -1.0, 1.0])


def read_nerfstudio(
    path: str | Path,
    *,
    allow_distortion: bool = False,
    holdout_every: int = DEFAULT_HOLDOUT,
) -> list[Camera]:
    """Read a nerfstudio ``transforms.json``.

    Poses are taken in the file's own world frame, so the splat PLY must have
    been trained in that frame too. ``applied_transform`` is not undone.
    """
    p = Path(path)
    doc = json.loads(p.read_text(encoding="utf-8"))
    if doc.get("camera_model", "OPENCV") not in ("OPENCV", "PINHOLE"):
        raise CameraFormatError(f"{p}: camera model {doc['camera_model']} is not supported")
    out = []
    any_distorted = False
    for frame in doc["frames"]:

        def get(key: str, frame: dict[str, Any] = frame) -> Any:
            if key in frame:
                return frame[key]
            if key in doc:
                return doc[key]
            raise CameraFormatError(f"{p}: frame {frame.get('file_path')} has no {key}")

        distorted = any(abs(float(frame.get(k, doc.get(k, 0.0)))) > 0 for k in _NS_DISTORTION)
        if distorted and not allow_distortion:
            raise CameraFormatError(
                f"{p}: frames carry lens distortion. Undistort first or pass allow_distortion."
            )
        any_distorted = any_distorted or distorted
        c2w_gl = np.asarray(frame["transform_matrix"], dtype=np.float64)
        if c2w_gl.shape == (3, 4):
            c2w_gl = np.vstack([c2w_gl, [0.0, 0.0, 0.0, 1.0]])
        w2c = np.linalg.inv(c2w_gl @ _GL_TO_CV)
        w2c[3] = [0.0, 0.0, 0.0, 1.0]
        out.append(
            Camera(
                name=str(frame["file_path"]),
                width=int(get("w")),
                height=int(get("h")),
                fx=float(get("fl_x")),
                fy=float(get("fl_y")),
                cx=float(get("cx")),
                cy=float(get("cy")),
                w2c=w2c,
            )
        )
    if any_distorted:
        warnings.warn("ignoring lens distortion in transforms.json", stacklevel=2)
    return assign_holdout_split(out, holdout_every)


# Construction helpers -----------------------------------------------------------


def look_at(
    eye: Sequence[float], target: Sequence[float], up: Sequence[float] = (0.0, 0.0, 1.0)
) -> F64:
    """World-to-camera matrix (OpenCV convention) for a camera at ``eye``."""
    eye_v = np.asarray(eye, dtype=np.float64)
    forward = np.asarray(target, dtype=np.float64) - eye_v
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, np.asarray(up, dtype=np.float64))
    if np.linalg.norm(right) < 1e-9:
        raise ValueError("up vector is parallel to the viewing direction")
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    rot = np.stack([right, down, forward])  # rows: camera axes in world coordinates
    w2c = np.eye(4, dtype=np.float64)
    w2c[:3, :3] = rot
    w2c[:3, 3] = -rot @ eye_v
    return w2c
