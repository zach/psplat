"""Sort splats by score and cut the order into budgets (P1.2, #21).

The order is a pure function of the scores: highest first, and splats with
equal scores keep their input order, so the same inputs always give the same
file. ``-inf`` is allowed and sorts last (the sampled baselines give it to
splats no view sees); NaN is refused, because it has no place in an order.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from psplat.io.scene import Scene


def sort_order(scores: NDArray[np.floating]) -> NDArray[np.int64]:
    """Indices of ``scores`` from highest to lowest; ties keep input order."""
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError(f"scores must be one per splat, got shape {values.shape}")
    if np.isnan(values).any():
        raise ValueError(f"{int(np.isnan(values).sum())} scores are NaN")
    if np.isposinf(values).any():
        raise ValueError("scores must not be +inf")
    # Stable sort of the negated scores: descending, ties by index.
    return np.argsort(-values, kind="stable").astype(np.int64)


def apply_order(scene: Scene, order: NDArray[np.integer]) -> Scene:
    """The scene with its splats permuted; ``order`` must be a permutation."""
    order = np.asarray(order)
    if order.shape != (scene.count,) or not np.array_equal(np.sort(order), np.arange(scene.count)):
        raise ValueError("order must be a permutation of the scene's splats")
    return scene.take(order)
