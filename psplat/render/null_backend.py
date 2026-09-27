"""A backend that renders nothing.

It exists so that ``order``, ``pack``, ``eval`` on pre-analysed inputs, and
unit tests can run with no renderer at all. Anything that needs pixels
raises :class:`BackendUnavailable` with a message saying what to use instead.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from psplat.io.cameras import Camera
from psplat.io.scene import Scene
from psplat.render.base import BackendUnavailable, RenderResult

MESSAGE = (
    "the null backend cannot render. It is for working with files that were "
    "already analysed. Use --backend torch, which runs on any machine, or "
    "--backend auto to pick the fastest available."
)


class NullRenderer:
    name = "null"

    def describe(self) -> str:
        return "null (no rendering)"

    def render(self, scene: Scene, camera: Camera) -> RenderResult:
        raise BackendUnavailable(MESSAGE)

    def contributions(self, scene: Scene, cameras: Sequence[Camera]) -> NDArray[np.float32]:
        raise BackendUnavailable(MESSAGE)

    def estimate_seconds(self, scene: Scene, cameras: Sequence[Camera]) -> float | None:
        return None
