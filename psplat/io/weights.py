"""The ``scene.weights.npz`` file written by ``psplat analyze``.

Arrays, all indexed ``[splat, view]`` in the scene's splat order and the
analysis's view order:

- ``contribution`` — float32, ``c_iv``: summed alpha × transmittance the
  splat deposits in the view (the rendering-contribution term, handoff §2.3);
- ``sigma_px``, ``sigma_px_minor``, ``sigma_px_major`` — float16, the
  splat's screen footprint in pixels: the isotropic equivalent
  (``sqrt(minor · major)``) and the two principal axes. Zero where the splat
  is not visible. The CSF term reads these as spatial frequency;
- ``visible`` — bool, the splat's footprint reaches a pixel of the view;
- ``view_names`` — the camera names, in column order;
- ``meta`` — a JSON string: format, version, terms, backend and provenance.

Optional per-splat arrays, summed over the views, present only when the
analysis term that produces them was requested (``docs/baselines.md``):

- ``hit_transmittance`` — float32 ``(N,)``, transmittance in front of the
  splat summed over the pixels it is composited into (term ``hits``);
- ``top_count`` — int64 ``(N,)``, pixels where the splat deposits the largest
  ``alpha · T`` of any splat (term ``top``);
- ``fisher`` — float32 ``(N, 6, 6)``, the Fisher block over the splat's mean
  and log-scale (term ``fisher``).

Footprints are float16 because at garden scale (3M splats × 160 views) each
float32 ``(N, V)`` array is about 2 GB; half precision keeps three
significant digits, far more than a CSF weight needs. The file is
compressed, which also shrinks the many zeros of invisible entries.
Perceptual terms (P2) add arrays under their own keys.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

FORMAT = "psplat-weights"
VERSION = 1
# Optional per-splat arrays: trailing shape after N, and stored dtype.
OPTIONAL_SHAPES: dict[str, tuple[int, ...]] = {
    "hit_transmittance": (),
    "top_count": (),
    "fisher": (6, 6),
}
OPTIONAL_DTYPES = {"hit_transmittance": np.float32, "top_count": np.int64, "fisher": np.float32}


class WeightsFormatError(ValueError):
    """A weights file is missing arrays or has inconsistent shapes."""


@dataclass
class Weights:
    contribution: NDArray[np.float32]
    sigma_px: NDArray[np.float16]
    sigma_px_minor: NDArray[np.float16]
    sigma_px_major: NDArray[np.float16]
    visible: NDArray[np.bool_]
    view_names: list[str]
    meta: dict[str, Any] = field(default_factory=dict)
    hit_transmittance: NDArray[np.float32] | None = None
    top_count: NDArray[np.int64] | None = None
    fisher: NDArray[np.float32] | None = None

    def __post_init__(self) -> None:
        shape = self.contribution.shape
        if len(shape) != 2:
            raise WeightsFormatError(f"contribution must be (N, V), not {shape}")
        for name in ("sigma_px", "sigma_px_minor", "sigma_px_major", "visible"):
            if getattr(self, name).shape != shape:
                raise WeightsFormatError(f"{name} has shape {getattr(self, name).shape} != {shape}")
        if len(self.view_names) != shape[1]:
            raise WeightsFormatError("view_names does not match the number of views")
        for name, want in OPTIONAL_SHAPES.items():
            array = getattr(self, name)
            if array is not None and array.shape != (shape[0], *want):
                raise WeightsFormatError(f"{name} has shape {array.shape} != {(shape[0], *want)}")

    @property
    def count(self) -> int:
        return int(self.contribution.shape[0])

    def save(self, path: str | Path) -> None:
        meta = {"format": FORMAT, "version": VERSION, **self.meta}
        path = Path(path)
        tmp = path.with_name(path.name + ".tmp.npz")
        optional = {
            name: getattr(self, name).astype(OPTIONAL_DTYPES[name])
            for name in OPTIONAL_SHAPES
            if getattr(self, name) is not None
        }
        np.savez_compressed(
            tmp,
            **optional,
            contribution=self.contribution.astype(np.float32),
            sigma_px=self.sigma_px.astype(np.float16),
            sigma_px_minor=self.sigma_px_minor.astype(np.float16),
            sigma_px_major=self.sigma_px_major.astype(np.float16),
            visible=self.visible.astype(bool),
            view_names=np.array(self.view_names, dtype=np.str_),
            meta=np.array(json.dumps(meta, sort_keys=True)),
        )
        tmp.replace(path)

    @staticmethod
    def load(path: str | Path) -> Weights:
        with np.load(path, allow_pickle=False) as data:
            try:
                meta = json.loads(str(data["meta"]))
                if meta.get("format") != FORMAT:
                    raise WeightsFormatError(f"{path}: not a psplat weights file")
                if meta.get("version") != VERSION:
                    raise WeightsFormatError(f"{path}: weights version {meta.get('version')}")
                return Weights(
                    contribution=data["contribution"],
                    sigma_px=data["sigma_px"],
                    sigma_px_minor=data["sigma_px_minor"],
                    sigma_px_major=data["sigma_px_major"],
                    visible=data["visible"],
                    view_names=[str(v) for v in data["view_names"]],
                    meta=meta,
                    **{name: data[name] for name in OPTIONAL_SHAPES if name in data.files},
                )
            except KeyError as exc:
                raise WeightsFormatError(f"{path}: missing array {exc}") from None
