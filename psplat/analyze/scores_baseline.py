"""Baseline importance scores (P1.1, https://github.com/zach/psplat/issues/20).

Each baseline is a pure function of the scene and the arrays in
``scene.weights.npz``, returning one float64 score per splat. Higher means
more important, so ``psplat order`` keeps the highest-scoring splats first.
The published methods leave choices open; ``docs/baselines.md`` records each
one and the reading taken here. In brief:

``opacity``
    Activated opacity. Blind to visibility and size.
``contribution``
    ``Σ_v c_iv``: the alpha × transmittance a splat deposits over all views.
``lightgaussian``
    LightGaussian's global significance, ``σ_i · Σ_pixels T · γ(V_i)``, summed
    over the pixels the splat is composited into, with the volume factor
    ``γ = min(V / V_90, 1)^β``, ``β = 0.1``.
``pup-simple``
    PUP 3D-GS's sensitivity, ``log det`` of the 6×6 Fisher block over mean and
    log-scale, taking only the direct term of each pixel's gradient, and
    damped so splats seen in too few pixels stay finite.
``coreset-simple``
    Sensitivity sampling: ``s_i = max_v c_iv / Σ_j c_jv``, the largest share
    of any view's total contribution, then splats in the order sampling with
    probability proportional to ``s`` would draw them.
``minisplatting-simple``
    Mini-Splatting: splats that are the largest contributor of some pixel come
    first, then every tier in the order importance sampling (``∝ Σ_v c_iv``)
    would draw them.

The sampled methods are reproducible from ``seed``. Their scores are
Gumbel keys, ``log w + G``, so sorting them descending gives exactly the
order of successive weighted sampling without replacement, and a splat with
zero weight scores ``-inf`` and comes last.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from psplat.io.scene import Scene
from psplat.io.weights import Weights

F64 = NDArray[np.float64]

LIGHTGAUSSIAN_BETA = 0.1
LIGHTGAUSSIAN_VOLUME_QUANTILE = 0.9
# PUP damping, relative to the median per-parameter Fisher magnitude.
PUP_DAMPING = 1e-6
# Splats per batch when taking 6×6 determinants, to bound float64 memory.
_DET_BATCH = 1 << 18


class MissingStatistic(ValueError):
    """A score needs an array the weights file does not have."""


@dataclass(frozen=True)
class Baseline:
    name: str
    needs: tuple[str, ...]  # analysis terms whose arrays the score reads
    sampled: bool  # uses the random seed
    summary: str


def _require(weights: Weights | None, method: str, scene: Scene, *arrays: str) -> Weights:
    if weights is None:
        raise MissingStatistic(f"{method} needs a weights file from psplat analyze")
    if weights.count != scene.count:
        raise MissingStatistic(
            f"the weights file has {weights.count:,} splats but the scene has {scene.count:,}; "
            "were they analysed from the same PLY?"
        )
    for name in arrays:
        if getattr(weights, name) is None:
            term = TERM_OF_ARRAY[name]
            raise MissingStatistic(
                f"{method} needs the {term} term, which this weights file lacks; "
                f"rerun psplat analyze with --terms contribution,{term} (or --terms baselines)"
            )
    return weights


TERM_OF_ARRAY = {"hit_transmittance": "hits", "top_count": "top", "fisher": "fisher"}


def gumbel_keys(weights: F64, seed: int) -> F64:
    """``log w + Gumbel noise``: sorting descending samples ``∝ w`` without replacement."""
    rng = np.random.default_rng(seed)
    noise = rng.gumbel(size=weights.shape)
    with np.errstate(divide="ignore"):
        return np.where(weights > 0, np.log(weights) + noise, -np.inf)


# The scores --------------------------------------------------------------------------------


def opacity(scene: Scene) -> F64:
    return scene.activated_opacities().astype(np.float64)


def contribution(weights: Weights) -> F64:
    return weights.contribution.astype(np.float64).sum(axis=1)


def lightgaussian(
    scene: Scene,
    weights: Weights,
    beta: float = LIGHTGAUSSIAN_BETA,
    volume_quantile: float = LIGHTGAUSSIAN_VOLUME_QUANTILE,
) -> F64:
    """Global significance (LightGaussian, Eq. 1) with its volume factor.

    ``Σ_pixels 1(G_i, r) · σ_i · T`` is the opacity times the transmittance
    summed over the pixels the splat is composited into, which is exactly
    ``hit_transmittance``.
    """
    assert weights.hit_transmittance is not None
    volume = (4.0 / 3.0) * np.pi * np.exp(scene.scales.astype(np.float64).sum(axis=1))
    reference = np.quantile(volume, volume_quantile) if volume.size else 1.0
    gamma = np.minimum(volume / max(reference, np.finfo(np.float64).tiny), 1.0) ** beta
    return opacity(scene) * weights.hit_transmittance.astype(np.float64) * gamma


def pup_fisher(weights: Weights, damping: float = PUP_DAMPING) -> F64:
    """``log det(I + F / ε)``: PUP's ``log det F``, damped and shifted to start at 0.

    ``ε`` is ``damping`` times the median of ``trace(F) / 6`` over splats with
    any Fisher mass, so it scales with the scene. Adding ``ε I`` keeps splats
    seen in fewer than six independent pixel gradients finite, and dividing
    by ``ε`` shifts every score by the same ``6 log ε``, so the order is that
    of ``log det(F + ε I)`` while a splat with no gradient at all scores 0.
    """
    assert weights.fisher is not None
    fisher = weights.fisher
    n = fisher.shape[0]
    out = np.zeros(n, np.float64)
    trace = np.trace(fisher, axis1=1, axis2=2).astype(np.float64)
    seen = trace > 0
    if not seen.any():
        return out
    eps = damping * float(np.median(trace[seen])) / 6.0
    eye = np.eye(6)
    for lo in range(0, n, _DET_BATCH):
        block = fisher[lo : lo + _DET_BATCH].astype(np.float64)
        sign, logdet = np.linalg.slogdet(eye + block / eps)
        # I + F/ε is symmetric positive definite; a non-positive sign could only
        # come from a corrupted file.
        if not np.all(sign > 0):
            raise ValueError("Fisher blocks are not positive semi-definite")
        out[lo : lo + _DET_BATCH] = logdet
    return out


def coreset_sensitivity(weights: Weights) -> F64:
    """``s_i = max_v c_iv / Σ_j c_jv``, the largest share of any one view."""
    c = weights.contribution.astype(np.float64)
    totals = c.sum(axis=0)
    share = np.divide(c, totals, out=np.zeros_like(c), where=totals > 0)
    return share.max(axis=1) if c.shape[1] else np.zeros(c.shape[0])


def coreset(weights: Weights, seed: int = 0) -> F64:
    return gumbel_keys(coreset_sensitivity(weights), seed)


def minisplatting(weights: Weights, seed: int = 0) -> F64:
    """Top contributors of some pixel first, each tier in importance-sampling order."""
    assert weights.top_count is not None
    keys = gumbel_keys(contribution(weights), seed)
    tier = weights.top_count > 0  # every top contributor has positive contribution
    rest = keys[~tier & np.isfinite(keys)]
    if tier.any() and rest.size:
        shift = max(0.0, float(rest.max() - keys[tier].min()) + 1.0)
        keys = np.where(tier, keys + shift, keys)
    return keys


# Registry -----------------------------------------------------------------------------------

BASELINES: dict[str, Baseline] = {
    b.name: b
    for b in (
        Baseline("opacity", (), False, "activated opacity"),
        Baseline("contribution", ("contribution",), False, "alpha x T summed over views"),
        Baseline("lightgaussian", ("contribution", "hits"), False,
                 "LightGaussian global significance with volume factor"),
        Baseline("pup-simple", ("contribution", "fisher"), False,
                 "PUP 3D-GS Fisher log-determinant, direct term, damped"),
        Baseline("coreset-simple", ("contribution",), True,
                 "sensitivity sampling on per-view contribution share"),
        Baseline("minisplatting-simple", ("contribution", "top"), True,
                 "top contributors first, then importance sampling"),
    )
}  # fmt: skip


def score(method: str, scene: Scene, weights: Weights | None = None, *, seed: int = 0) -> F64:
    """Score every splat with a baseline from :data:`BASELINES`."""
    if method not in BASELINES:
        raise ValueError(f"unknown baseline {method!r}; known: {', '.join(BASELINES)}")
    run: dict[str, Callable[[], F64]] = {
        "opacity": lambda: opacity(scene),
        "contribution": lambda: contribution(_require(weights, method, scene)),
        "lightgaussian": lambda: lightgaussian(
            scene, _require(weights, method, scene, "hit_transmittance")
        ),
        "pup-simple": lambda: pup_fisher(_require(weights, method, scene, "fisher")),
        "coreset-simple": lambda: coreset(_require(weights, method, scene), seed),
        "minisplatting-simple": lambda: minisplatting(
            _require(weights, method, scene, "top_count"), seed
        ),
    }
    return run[method]()
