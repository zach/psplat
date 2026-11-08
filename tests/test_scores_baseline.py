"""P1.1: baseline importance scores.

Toy cases build the weights arrays by hand, so each property is checked
against the score's definition alone. The integration cases run the torch
backend on the synthetic fixture and on a rendered duplicate.
"""

from __future__ import annotations

import math
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from typer.testing import CliRunner

from psplat.analyze import scores_baseline as sb
from psplat.analyze.contribution import analyze, check_terms
from psplat.cli import app
from psplat.io.cameras import Camera, look_at, read_cameras_json
from psplat.io.ply import read_ply
from psplat.io.scene import Scene
from psplat.io.weights import Weights
from psplat.render.base import BackendUnavailable
from psplat.render.null_backend import NullRenderer
from psplat.render.torch_backend import TorchRenderer

DETERMINISTIC = [b.name for b in sb.BASELINES.values() if not b.sampled]
SAMPLED = [b.name for b in sb.BASELINES.values() if b.sampled]


def toy_scene(n: int, opacity: float = 0.5, log_scale: float = -2.0) -> Scene:
    f = np.float32
    return Scene(
        means=np.zeros((n, 3), f),
        scales=np.full((n, 3), log_scale, f),
        quats=np.tile(np.array([1, 0, 0, 0], f), (n, 1)),
        opacities=np.full(n, math.log(opacity / (1 - opacity)), f),
        sh0=np.zeros((n, 3), f),
        shN=np.zeros((n, 0, 3), f),
    )


def toy_weights(contribution: np.ndarray, **optional: np.ndarray) -> Weights:
    n, v = contribution.shape
    zeros = np.zeros((n, v), np.float16)
    return Weights(
        contribution=contribution.astype(np.float32),
        sigma_px=zeros,
        sigma_px_minor=zeros,
        sigma_px_major=zeros,
        visible=contribution > 0,
        view_names=[f"v{k}" for k in range(v)],
        **optional,
    )


def random_weights(n: int = 12, v: int = 4, seed: int = 0) -> Weights:
    """Every array present; splat 0 invisible everywhere."""
    rng = np.random.default_rng(seed)
    c = rng.uniform(0.1, 5.0, (n, v))
    c[0] = 0
    g = rng.normal(size=(n, 6, 9))
    fisher = (g @ g.transpose(0, 2, 1)).astype(np.float32)
    fisher[0] = 0
    hits = rng.uniform(1, 50, n).astype(np.float32)
    hits[0] = 0
    top = rng.integers(0, 4, n).astype(np.int64)
    top[0] = 0
    return toy_weights(c, hit_transmittance=hits, top_count=top, fisher=fisher)


# Toy cases ----------------------------------------------------------------------------------


@pytest.mark.parametrize("method", DETERMINISTIC)
def test_invisible_splat_scores_zero(method: str) -> None:
    w = random_weights()
    scores = sb.score(method, toy_scene(w.count), w)
    assert scores.dtype == np.float64 and scores.shape == (w.count,)
    if method == "opacity":  # by definition blind to visibility
        assert scores[0] == pytest.approx(0.5)
    else:
        assert scores[0] == 0 and np.all(scores[1:] > 0)


@pytest.mark.parametrize("method", SAMPLED)
def test_invisible_splat_is_drawn_last(method: str) -> None:
    w = random_weights()
    scores = sb.score(method, toy_scene(w.count), w)
    assert scores[0] == -np.inf and np.isfinite(scores[1:]).all()


def split_row(w: Weights, i: int) -> Weights:
    """Splat ``i`` replaced by two copies that each carry half its arrays."""

    def halve(a: np.ndarray | None) -> np.ndarray | None:
        if a is None:
            return None
        out = np.concatenate([a, a[i : i + 1]])
        if out.dtype.kind == "f":
            out[i] = out[-1] = a[i] / 2
        return out

    return Weights(
        contribution=halve(w.contribution),  # type: ignore[arg-type]
        sigma_px=np.concatenate([w.sigma_px, w.sigma_px[i : i + 1]]),
        sigma_px_minor=np.concatenate([w.sigma_px_minor, w.sigma_px_minor[i : i + 1]]),
        sigma_px_major=np.concatenate([w.sigma_px_major, w.sigma_px_major[i : i + 1]]),
        visible=np.concatenate([w.visible, w.visible[i : i + 1]]),
        view_names=w.view_names,
        hit_transmittance=halve(w.hit_transmittance),
        top_count=halve(w.top_count),
        fisher=halve(w.fisher),
    )


@pytest.mark.parametrize("method", ["contribution", "lightgaussian"])
def test_splitting_a_splats_mass_in_two_halves_its_score(method: str) -> None:
    w = random_weights()
    before = sb.score(method, toy_scene(w.count), w)
    after = sb.score(method, toy_scene(w.count + 1), split_row(w, 3))
    np.testing.assert_allclose(after[[3, -1]], before[3] / 2, rtol=1e-6)
    np.testing.assert_allclose(np.delete(after[:-1], 3), np.delete(before, 3), rtol=1e-6)


def test_splitting_halves_coreset_sensitivity_and_lowers_pup() -> None:
    w = random_weights()
    s = sb.coreset_sensitivity(w)
    s2 = sb.coreset_sensitivity(split_row(w, 3))
    np.testing.assert_allclose(s2[[3, -1]], s[3] / 2, rtol=1e-6)
    pup = sb.pup_fisher(w)
    pup2 = sb.pup_fisher(split_row(w, 3))
    assert 0 < pup2[3] < pup[3]


def test_lightgaussian_volume_factor() -> None:
    # Same opacity and hits; volumes span the 90th percentile. Splats at or
    # above it get γ = 1; a splat 1000x smaller in volume gets 1000^-0.1.
    n = 10
    scene = toy_scene(n)
    scene.scales[:] = -1.0
    scene.scales[0] = -1.0 - math.log(10.0)  # each axis /10: volume /1000
    w = toy_weights(np.ones((n, 1)), hit_transmittance=np.full(n, 7.0, np.float32))
    got = sb.lightgaussian(scene, w)
    np.testing.assert_allclose(got[1:], 0.5 * 7.0)
    assert got[0] == pytest.approx(0.5 * 7.0 * 1000.0**-0.1, rel=1e-5)


def test_pup_is_zero_without_gradient_and_finite_when_rank_deficient() -> None:
    w = random_weights()
    assert w.fisher is not None
    v = np.zeros(6, np.float32)
    v[0] = 3.0
    w.fisher[5] = np.outer(v, v)  # one pixel's worth of gradient: rank 1
    pup = sb.pup_fisher(w)
    assert pup[0] == 0 and np.isfinite(pup).all() and 0 < pup[5] < pup[1:5].min()
    # Equal to log det(F + εI) - 6 log ε, with ε from the median trace.
    trace = np.trace(w.fisher, axis1=1, axis2=2)
    eps = sb.PUP_DAMPING * np.median(trace[trace > 0]) / 6
    want = np.linalg.slogdet(w.fisher[1].astype(np.float64) + eps * np.eye(6))[1] - 6 * np.log(eps)
    assert pup[1] == pytest.approx(want, rel=1e-9)


def test_coreset_sensitivity_is_the_largest_view_share() -> None:
    c = np.array([[1.0, 0.0], [3.0, 1.0], [0.0, 1.0]])
    np.testing.assert_allclose(sb.coreset_sensitivity(toy_weights(c)), [0.25, 0.75, 0.5])


def test_sampled_order_draws_in_proportion_to_weight() -> None:
    # The first key is the first draw, so its frequency is w / Σw.
    w = np.array([1.0, 2.0, 3.0, 4.0])
    firsts = np.bincount([int(np.argmax(sb.gumbel_keys(w, s))) for s in range(4000)], minlength=4)
    np.testing.assert_allclose(firsts / 4000, w / w.sum(), atol=0.03)


@pytest.mark.parametrize("method", SAMPLED)
def test_sampled_scores_are_reproducible_from_the_seed(method: str) -> None:
    w = random_weights(n=200)
    scene = toy_scene(w.count)
    a = sb.score(method, scene, w, seed=4)
    np.testing.assert_array_equal(a, sb.score(method, scene, w, seed=4))
    assert not np.array_equal(np.argsort(a), np.argsort(sb.score(method, scene, w, seed=5)))


def test_minisplatting_puts_every_top_contributor_first() -> None:
    w = random_weights(n=300)
    assert w.top_count is not None
    scores = sb.minisplatting(w, seed=1)
    tier = w.top_count > 0
    assert 0 < tier.sum() < w.count - 1
    assert scores[tier].min() > scores[~tier].max()


def test_missing_arrays_name_the_analysis_term() -> None:
    w = toy_weights(np.ones((3, 2)))
    scene = toy_scene(3)
    for method, term in [("lightgaussian", "hits"), ("pup-simple", "fisher"),
                         ("minisplatting-simple", "top")]:  # fmt: skip
        with pytest.raises(sb.MissingStatistic, match=f"contribution,{term}"):
            sb.score(method, scene, w)
    with pytest.raises(sb.MissingStatistic, match="needs a weights file"):
        sb.score("contribution", scene)
    with pytest.raises(sb.MissingStatistic, match="same PLY"):
        sb.score("contribution", toy_scene(4), w)
    with pytest.raises(ValueError, match="unknown baseline"):
        sb.score("random", scene, w)
    assert sb.score("opacity", scene).shape == (3,)


# Integration with the torch backend --------------------------------------------------------


def test_terms_expand_and_order() -> None:
    assert check_terms(["baselines"]) == ["contribution", "hits", "top", "fisher"]
    assert check_terms(["fisher", "hits"]) == ["contribution", "hits", "fisher"]
    assert check_terms(["contribution"]) == ["contribution"]


def test_a_backend_without_statistics_says_what_to_use() -> None:
    with pytest.raises(BackendUnavailable, match="--backend torch"):
        analyze(toy_scene(2), [], NullRenderer(), ["hits"])


@pytest.fixture(scope="module")
def fixture_weights(synthetic_dir: Path) -> tuple[Scene, Weights]:
    scene = read_ply(synthetic_dir / "scene.ply")
    cams = read_cameras_json(synthetic_dir / "cams.json")
    return scene, analyze(scene, cams, TorchRenderer("cpu"), ["baselines"])


@pytest.mark.parametrize("method", list(sb.BASELINES))
def test_fixture_offscreen_splats_rank_last(
    method: str, fixture_weights: tuple[Scene, Weights], oracles: dict[str, Any]
) -> None:
    scene, w = fixture_weights
    scores = sb.score(method, scene, w)
    off = oracles["offscreen"]["splats"]
    if method == "opacity":
        return  # opacity cannot see that they are off screen; documented
    floor = -np.inf if sb.BASELINES[method].sampled else 0.0
    assert np.all(scores[off] == floor)
    on = np.setdiff1d(np.arange(scene.count), off)
    # Every on-screen splat contributes to some view in this fixture; the
    # Fisher only needs one pixel of gradient.
    assert np.all(scores[on] > floor) or method == "minisplatting-simple"


def test_rendered_split_halves_scores_that_measure_mass() -> None:
    # One splat, then two coincident copies whose opacities compose to the
    # original's at the centre: 1 - (1 - o')² = o. They render nearly the same
    # image, so a score that measures what a splat adds should give each copy
    # about half. The split is exact only to first order in alpha, and each
    # copy loses a little more tail to the 1/255 skip threshold, so "about"
    # is 12% here. PUP does not halve: each copy's gradient is about half the
    # original's, so its Fisher block is about a quarter.
    o = 0.3
    one = toy_scene(1, opacity=o, log_scale=math.log(0.2))
    half = toy_scene(2, opacity=1 - math.sqrt(1 - o), log_scale=math.log(0.2))
    cams = [
        Camera("a", 40, 30, 36.0, 36.0, 20.0, 15.0, look_at((0, -4, 0), (0, 0, 0))),
        Camera("b", 40, 30, 36.0, 36.0, 20.0, 15.0, look_at((4, 0, 1), (0, 0, 0))),
    ]
    oracle = TorchRenderer("cpu")
    w1 = analyze(one, cams, oracle, ["baselines"])
    w2 = analyze(half, cams, oracle, ["baselines"])
    for method in ("opacity", "contribution", "lightgaussian"):
        s1 = sb.score(method, one, w1)[0]
        np.testing.assert_allclose(sb.score(method, half, w2), s1 / 2, rtol=0.12, err_msg=method)
    # Alone in every view, a splat's share is 1 however it is split; give the
    # views a second, fixed splat so the share means something.
    other = Weights(
        contribution=np.vstack([w1.contribution, [[3.0, 3.0]]]).astype(np.float32),
        sigma_px=np.zeros((2, 2), np.float16), sigma_px_minor=np.zeros((2, 2), np.float16),
        sigma_px_major=np.zeros((2, 2), np.float16), visible=np.ones((2, 2), bool),
        view_names=["a", "b"],
    )  # fmt: skip
    split = Weights(
        contribution=np.vstack([w2.contribution, [[3.0, 3.0]]]).astype(np.float32),
        sigma_px=np.zeros((3, 2), np.float16), sigma_px_minor=np.zeros((3, 2), np.float16),
        sigma_px_major=np.zeros((3, 2), np.float16), visible=np.ones((3, 2), bool),
        view_names=["a", "b"],
    )  # fmt: skip
    s1 = sb.coreset_sensitivity(other)[0]
    np.testing.assert_allclose(sb.coreset_sensitivity(split)[:2], s1 / 2, rtol=0.12)
    assert w1.fisher is not None and w2.fisher is not None
    trace = np.trace(w2.fisher, axis1=1, axis2=2)
    np.testing.assert_allclose(trace, np.trace(w1.fisher[0]) / 4, rtol=0.25)


def test_an_exact_copy_keeps_almost_its_whole_score() -> None:
    # The contrast case: a copy with the same opacity doubles the faint
    # splat's alpha, so each copy is scored nearly as the original was.
    one = toy_scene(1, opacity=0.02, log_scale=math.log(0.2))
    cam = Camera("a", 40, 30, 36.0, 36.0, 20.0, 15.0, look_at((0, -4, 0), (0, 0, 0)))
    c1 = TorchRenderer("cpu").contributions(one, [cam])[0, 0]
    c2 = TorchRenderer("cpu").contributions(one.take(np.array([0, 0])), [cam])[:, 0]
    np.testing.assert_allclose(c2, c1, rtol=0.03)


def test_cli_baselines_terms_round_trip(tmp_path: Path, synthetic_dir: Path) -> None:
    for name in ("scene.ply", "cams.json"):
        shutil.copy(synthetic_dir / name, tmp_path / name)
    result = CliRunner().invoke(app, [
        "analyze", str(tmp_path / "scene.ply"), "--cameras", str(tmp_path / "cams.json"),
        "--backend", "torch", "--device", "cpu", "--terms", "baselines",
        "--out", str(tmp_path / "w.npz"),
    ])  # fmt: skip
    assert result.exit_code == 0, result.output
    w = Weights.load(tmp_path / "w.npz")
    assert w.meta["terms"] == ["contribution", "hits", "top", "fisher"]
    assert w.hit_transmittance is not None and w.hit_transmittance.dtype == np.float32
    assert w.top_count is not None and w.top_count.dtype == np.int64
    assert w.fisher is not None and w.fisher.shape == (200, 6, 6)
    plain = tmp_path / "plain.npz"
    result = CliRunner().invoke(app, [
        "analyze", str(tmp_path / "scene.ply"), "--cameras", str(tmp_path / "cams.json"),
        "--backend", "torch", "--device", "cpu", "--out", str(plain),
    ])  # fmt: skip
    assert result.exit_code == 0, result.output
    assert Weights.load(plain).fisher is None


def test_optional_arrays_are_shape_checked() -> None:
    from psplat.io.weights import WeightsFormatError

    with pytest.raises(WeightsFormatError, match="fisher"):
        toy_weights(np.ones((3, 2)), fisher=np.zeros((3, 6), np.float32))
    with pytest.raises(WeightsFormatError, match="top_count"):
        toy_weights(np.ones((3, 2)), top_count=np.zeros(4, np.int64))
