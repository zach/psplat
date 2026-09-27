"""The Renderer protocol every backend implements (handoff §0.2).

Arrays cross this boundary as NumPy, so callers never depend on which
framework a backend uses internally. Each backend provides two things:

- ``render`` — an image of the scene from one camera;
- ``contributions`` — the ``(N, V)`` matrix of each splat's rendering
  contribution ``c_iv`` to each view: the sum over pixels of the
  ``alpha · T`` it deposits, where ``T`` is transmittance in front of it.

Screen-space footprints and visibility come from :mod:`psplat.render.project`,
not from backends, so they mean the same thing whichever backend ran.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from psplat.io.cameras import Camera
from psplat.io.scene import Scene


class BackendUnavailable(RuntimeError):
    """The requested backend cannot run here; the message says what to do instead."""


@dataclass(frozen=True)
class RenderResult:
    image: NDArray[np.float32]  # (H, W, 3), linear, background black
    alpha: NDArray[np.float32]  # (H, W), 1 - final transmittance


@runtime_checkable
class Renderer(Protocol):
    name: str

    def describe(self) -> str:
        """One line for notices, e.g. ``"torch on cpu"``."""
        ...

    def render(self, scene: Scene, camera: Camera) -> RenderResult: ...

    def contributions(self, scene: Scene, cameras: Sequence[Camera]) -> NDArray[np.float32]:
        """``(N, V)`` float32 rendering contribution of each splat to each view."""
        ...

    def estimate_seconds(self, scene: Scene, cameras: Sequence[Camera]) -> float | None:
        """Rough wall time for ``contributions``, or ``None`` if unknown."""
        ...
