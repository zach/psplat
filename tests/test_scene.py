"""P0.2: the Scene dataclass."""

from __future__ import annotations

import numpy as np
import pytest

from psplat.io.scene import Scene


def make_scene(n: int = 5, degree: int = 1, seed: int = 0) -> Scene:
    rng = np.random.default_rng(seed)
    k = (degree + 1) ** 2 - 1
    f = np.float32
    return Scene(
        means=rng.normal(size=(n, 3)).astype(f),
        scales=rng.normal(size=(n, 3)).astype(f),
        quats=rng.normal(size=(n, 4)).astype(f),
        opacities=rng.normal(size=n).astype(f),
        sh0=rng.normal(size=(n, 3)).astype(f),
        shN=rng.normal(size=(n, k, 3)).astype(f),
        extra={"nx": np.zeros(n, f), "label": np.arange(n, dtype=np.int32)},
    )


def test_counts_and_degree() -> None:
    s = make_scene(7, degree=3)
    assert s.count == 7 and s.sh_degree == 3
    assert make_scene(2, degree=0).sh_degree == 0


def test_activation_helpers() -> None:
    s = make_scene()
    np.testing.assert_allclose(s.activated_scales(), np.exp(s.scales), rtol=1e-6)
    np.testing.assert_allclose(s.activated_opacities(), 1 / (1 + np.exp(-s.opacities)), rtol=1e-6)
    np.testing.assert_allclose(np.linalg.norm(s.normalized_quats(), axis=1), 1.0, rtol=1e-6)


def test_take_permutes_every_array_including_extras() -> None:
    s = make_scene(6)
    order = np.array([5, 0, 3])
    t = s.take(order)
    assert t.count == 3
    np.testing.assert_array_equal(t.means, s.means[order])
    np.testing.assert_array_equal(t.shN, s.shN[order])
    np.testing.assert_array_equal(t.extra["label"], [5, 0, 3])
    assert s.take(np.arange(6)).equals(s)
    assert not t.equals(s)


def test_equals_is_bitwise() -> None:
    a = make_scene()
    a.means[0, 0] = np.nan
    b = a.take(np.arange(a.count))
    assert a.equals(b)
    b.means[1, 1] = -0.0 if a.means[1, 1] == 0 else a.means[1, 1] * 2
    assert not a.equals(b)


@pytest.mark.parametrize("field", ["scales", "quats", "opacities", "sh0"])
def test_shape_validation(field: str) -> None:
    s = make_scene()
    kwargs = {
        name: getattr(s, name) for name in ("means", "scales", "quats", "opacities", "sh0", "shN")
    }
    kwargs[field] = kwargs[field][:-1]
    with pytest.raises(ValueError, match=field):
        Scene(**kwargs)


def test_rejects_non_sh_coefficient_count() -> None:
    s = make_scene()
    with pytest.raises(ValueError, match="coefficients"):
        Scene(s.means, s.scales, s.quats, s.opacities, s.sh0, np.zeros((5, 4, 3), np.float32))
