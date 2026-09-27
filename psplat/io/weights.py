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

    def __post_init__(self) -> None:
        shape = self.contribution.shape
        if len(shape) != 2:
            raise WeightsFormatError(f"contribution must be (N, V), not {shape}")
        for name in ("sigma_px", "sigma_px_minor", "sigma_px_major", "visible"):
            if getattr(self, name).shape != shape:
                raise WeightsFormatError(f"{name} has shape {getattr(self, name).shape} != {shape}")
        if len(self.view_names) != shape[1]:
            raise WeightsFormatError("view_names does not match the number of views")

    @property
    def count(self) -> int:
        return int(self.contribution.shape[0])

    def save(self, path: str | Path) -> None:
        meta = {"format": FORMAT, "version": VERSION, **self.meta}
        path = Path(path)
        tmp = path.with_name(path.name + ".tmp.npz")
        np.savez_compressed(
            tmp,
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
                )
            except KeyError as exc:
                raise WeightsFormatError(f"{path}: missing array {exc}") from None
