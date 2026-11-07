"""Banded texel order for SOG (decision D7, docs/formats.md)."""

from __future__ import annotations

import numpy as np
import pytest

from psplat.pack.layout import band_bounds, banded_order, morton_codes


def test_morton_codes_interleave_axes() -> None:
    # The corners of a unit cube at 1 bit per axis are the codes 0..7, x lowest.
    corners = np.array([[x, y, z] for z in (0, 1) for y in (0, 1) for x in (0, 1)], float)
    assert morton_codes(corners, bits=1).tolist() == list(range(8))
    assert morton_codes(np.zeros((0, 3))).shape == (0,)
    with pytest.raises(ValueError, match="bits"):
        morton_codes(corners, bits=22)


def test_band_bounds() -> None:
    assert band_bounds(1000, [5, 10, 25, 50, 75, 100]) == [0, 50, 100, 250, 500, 750, 1000]
    assert band_bounds(3, [1, 50]) == [0, 1, 2, 3]  # tiny scenes still get a first band
    assert band_bounds(0, [50]) == [0]
    with pytest.raises(ValueError, match="cut points"):
        band_bounds(10, [0])


def test_banded_order_keeps_every_cut_point_an_exact_prefix() -> None:
    rng = np.random.default_rng(0)
    means = rng.normal(size=(1000, 3))
    cuts = [5, 10, 25, 50, 75, 100]
    order = banded_order(means, cuts)
    assert order.dtype == np.uint32 and sorted(order.tolist()) == list(range(1000))
    for k in band_bounds(1000, cuts):
        assert set(order[:k].tolist()) == set(range(k))
    assert not np.array_equal(order, np.arange(1000))  # it did reorder within bands
    # Within a band, texels follow the Morton order of that band's own splats.
    expected = 100 + np.argsort(morton_codes(means[100:250]), kind="stable")
    np.testing.assert_array_equal(order[100:250], expected)
