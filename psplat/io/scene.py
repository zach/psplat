"""The in-memory Gaussian splat scene.

Arrays are stored exactly as a 3DGS PLY stores them, so reading and writing
never changes a value:

- ``scales`` are log-space (``exp`` gives world-space standard deviations);
- ``opacities`` are logits (``sigmoid`` gives opacity in [0, 1]);
- ``quats`` are ``(w, x, y, z)`` and not necessarily unit length;
- ``sh0`` holds the degree-0 spherical-harmonic coefficients per channel;
- ``shN`` holds the higher-order coefficients as ``(N, K, 3)``, with
  ``K = (degree + 1) ** 2 - 1``.

Use the ``activated_*`` helpers to get renderable values. Every other
per-splat attribute read from a file lives in ``extra``, in file order, so
nothing is dropped on a round trip.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

F32 = NDArray[np.float32]

# Number of higher-order SH coefficients per channel for each degree.
SH_REST_BY_DEGREE = {0: 0, 1: 3, 2: 8, 3: 15}


@dataclass(frozen=True)
class PlyLayout:
    """How a scene was laid out in the PLY it came from.

    Kept so that writing an unmodified scene reproduces the source file
    byte for byte. ``properties`` are ``(name, ply_type)`` pairs in file
    order; ``comments`` are the header comment lines without the keyword.
    """

    properties: tuple[tuple[str, str], ...]
    comments: tuple[str, ...] = ()
    obj_info: tuple[str, ...] = ()


@dataclass
class Scene:
    """A Gaussian splat scene; see the module docstring for storage conventions."""

    means: F32
    scales: F32
    quats: F32
    opacities: F32
    sh0: F32
    shN: F32
    extra: dict[str, NDArray[np.generic]] = field(default_factory=dict)
    ply_layout: PlyLayout | None = field(default=None, compare=False)

    def __post_init__(self) -> None:
        self.validate()

    @property
    def count(self) -> int:
        return int(self.means.shape[0])

    @property
    def sh_degree(self) -> int:
        k = int(self.shN.shape[1])
        for degree, rest in SH_REST_BY_DEGREE.items():
            if rest == k:
                return degree
        raise ValueError(f"shN has {k} coefficients per channel, which is no SH degree")

    def validate(self) -> None:
        """Raise ``ValueError`` if array shapes are inconsistent."""
        n = self.means.shape[0] if self.means.ndim == 2 else -1
        expected: Mapping[str, tuple[int, ...]] = {
            "means": (n, 3),
            "scales": (n, 3),
            "quats": (n, 4),
            "opacities": (n,),
            "sh0": (n, 3),
        }
        for name, shape in expected.items():
            actual = getattr(self, name).shape
            if actual != shape:
                raise ValueError(f"{name} has shape {actual}, expected {shape}")
        if self.shN.ndim != 3 or self.shN.shape[0] != n or self.shN.shape[2] != 3:
            raise ValueError(f"shN has shape {self.shN.shape}, expected ({n}, K, 3)")
        if self.shN.shape[1] not in SH_REST_BY_DEGREE.values():
            raise ValueError(f"shN has {self.shN.shape[1]} coefficients per channel")
        for name, values in self.extra.items():
            if values.shape != (n,):
                raise ValueError(f"extra attribute {name!r} has shape {values.shape}")

    # Activated values -------------------------------------------------------

    def activated_scales(self) -> F32:
        """World-space standard deviations along each local axis."""
        return np.exp(self.scales).astype(np.float32)

    def activated_opacities(self) -> F32:
        """Opacity in [0, 1]."""
        return (1.0 / (1.0 + np.exp(-self.opacities.astype(np.float64)))).astype(np.float32)

    def normalized_quats(self) -> F32:
        """Unit quaternions ``(w, x, y, z)``."""
        norm = np.linalg.norm(self.quats, axis=1, keepdims=True)
        return (self.quats / np.maximum(norm, 1e-12)).astype(np.float32)

    # Reordering ---------------------------------------------------------------

    def take(self, indices: NDArray[np.integer]) -> Scene:
        """Return a new scene holding the splats at ``indices``, in that order.

        This is how an ordering is applied and how a budget is cut: the first
        ``k`` entries of an ordered scene are ``scene.take(order[:k])``.
        """
        idx = np.asarray(indices)
        return Scene(
            means=self.means[idx],
            scales=self.scales[idx],
            quats=self.quats[idx],
            opacities=self.opacities[idx],
            sh0=self.sh0[idx],
            shN=self.shN[idx],
            extra={name: values[idx] for name, values in self.extra.items()},
            ply_layout=self.ply_layout,
        )

    def equals(self, other: Scene) -> bool:
        """Bitwise equality of every array, including extras and their order.

        Bitwise rather than numeric, so NaN payloads and signed zeros count.
        """
        if list(self.extra) != list(other.extra):
            return False
        pairs = [
            (self.means, other.means),
            (self.scales, other.scales),
            (self.quats, other.quats),
            (self.opacities, other.opacities),
            (self.sh0, other.sh0),
            (self.shN, other.shN),
            *((self.extra[k], other.extra[k]) for k in self.extra),
        ]
        return all(
            a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes()
            for a, b in pairs
        )
