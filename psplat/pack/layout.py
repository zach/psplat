"""Texel and record layouts that keep psplat's ordering truncatable.

SOG stores splats as image pixels, and its writer Morton-sorts them for
compression unless given an explicit order (``docs/formats.md``). psplat
passes a *banded* order instead: splats keep the cut-point band they belong
to, and are Morton-sorted only within a band. Every cut point is then an exact
prefix of the perceptual order, while most of the compression returns.
"""

from __future__ import annotations

import itertools
from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray


def morton_codes(points: NDArray[np.floating], bits: int = 10) -> NDArray[np.uint64]:
    """3D Morton (Z-order) codes, each axis quantised to ``bits`` bits over the points' bounds."""
    if not 1 <= bits <= 21:
        raise ValueError("bits must be between 1 and 21")
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) == 0:
        return np.zeros(0, np.uint64)
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    q = ((pts - lo) / np.maximum(hi - lo, 1e-12) * (2**bits - 1)).astype(np.uint64)
    code = np.zeros(len(pts), np.uint64)
    for b in range(bits):
        for axis in range(3):
            bit = (q[:, axis] >> np.uint64(b)) & np.uint64(1)
            code |= bit << np.uint64(3 * b + axis)
    return code


def band_bounds(count: int, cuts_percent: Sequence[float]) -> list[int]:
    """Band boundaries ``[0, i1, i2, ..., count]`` for cut points given in percent."""
    if count < 0:
        raise ValueError("count must be non-negative")
    if any(not 0 < c <= 100 for c in cuts_percent):
        raise ValueError("cut points must be in (0, 100]")
    inner = {min(count, max(1, round(count * c / 100))) for c in cuts_percent}
    return sorted({0, count, *inner}) if count else [0]


def banded_order(means: NDArray[np.floating], cuts_percent: Sequence[float]) -> NDArray[np.uint32]:
    """Texel order for splats already in perceptual order.

    Entry ``k`` is the splat placed at texel ``k``. Splats never leave their
    band, so the first ``i`` texels at any cut point ``i`` hold exactly
    splats ``0 .. i-1``; within a band they are in Morton order.
    """
    return banded_order_at(means, band_bounds(len(means), cuts_percent))


def banded_order_at(means: NDArray[np.floating], bounds: Sequence[int]) -> NDArray[np.uint32]:
    """:func:`banded_order` with the band edges given as splat counts.

    ``bounds`` must start at 0, end at ``len(means)`` and never decrease;
    the manifest's ``cuts[].index`` values, with 0 in front, are such a list.
    """
    n = len(means)
    bounds = list(bounds)
    if not bounds or bounds[0] != 0 or bounds[-1] != n or bounds != sorted(bounds):
        raise ValueError(f"band edges must rise from 0 to {n}, got {bounds}")
    order = []
    for start, stop in itertools.pairwise(bounds):
        idx = np.arange(start, stop)
        order.append(idx[np.argsort(morton_codes(np.asarray(means)[idx]), kind="stable")])
    return (np.concatenate(order) if order else np.zeros(0)).astype(np.uint32)
