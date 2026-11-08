"""The rendering-contribution term and per-view footprints.

:func:`analyze` is the Python form of ``psplat analyze``: it runs a backend
over a camera set and returns :class:`~psplat.io.weights.Weights`. Footprints
and visibility come from the shared projection, so they are identical
whichever backend computed the contributions.

Besides ``contribution``, which is always computed, three terms gather the
per-splat statistics that the baseline importance scores need
(:mod:`psplat.analyze.scores_baseline`): ``hits``, ``top`` and ``fisher``.
``baselines`` is shorthand for all of them.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray

from psplat.io.cameras import Camera
from psplat.io.scene import Scene
from psplat.io.weights import Weights
from psplat.render.base import BackendUnavailable, ProgressFn, Renderer
from psplat.render.project import SceneTensors, project

SUPPORTED_TERMS = ("contribution", "hits", "top", "fisher")
# Each extra term and the backend statistic that provides it.
TERM_STATISTICS = {"hits": "hit_transmittance", "top": "top_count", "fisher": "fisher"}
ALIASES = {"baselines": SUPPORTED_TERMS}
ISSUES = "https://github.com/zach/psplat/issues/"
# Terms that are specified but not written yet, and the issue tracking each.
PLANNED_TERMS = {"csf": 32, "masking": 33, "saliency": 34}


class TermUnavailable(ValueError):
    """A requested analysis term is unknown or not implemented yet."""


def check_terms(terms: Sequence[str]) -> list[str]:
    """Validate requested terms; returns them deduplicated, in canonical order.

    ``contribution`` is always included, because every other term and score
    builds on it.
    """
    out = ["contribution"]
    for term in terms:
        term = term.strip()
        if term in ALIASES:
            out.extend(ALIASES[term])
        elif term in SUPPORTED_TERMS:
            out.append(term)
        elif term in PLANNED_TERMS:
            raise TermUnavailable(
                f"the {term} term is not implemented yet; it is tracked in "
                f"{ISSUES}{PLANNED_TERMS[term]}"
            )
        else:
            known = ", ".join((*SUPPORTED_TERMS, *ALIASES, *PLANNED_TERMS))
            raise TermUnavailable(f"unknown term {term!r}; known terms: {known}")
    if not any(t.strip() for t in terms):
        raise TermUnavailable("at least one term is required")
    return [t for t in SUPPORTED_TERMS if t in out]


def footprints(
    scene: Scene, cameras: Sequence[Camera]
) -> tuple[NDArray[np.float16], NDArray[np.float16], NDArray[np.float16], NDArray[np.bool_]]:
    """``(sigma_px, minor, major, visible)``, each ``(N, V)``; zero where invisible."""
    splats = SceneTensors.from_scene(scene, "cpu", torch.float32)
    n, v = scene.count, len(cameras)
    sigma = np.zeros((n, v), np.float16)
    minor = np.zeros((n, v), np.float16)
    major = np.zeros((n, v), np.float16)
    visible = np.zeros((n, v), bool)
    for j, cam in enumerate(cameras):
        proj = project(splats, cam)
        vis = proj.visible.numpy()
        lo, hi = proj.sigma_px_axes()
        visible[:, j] = vis
        sigma[:, j] = np.where(vis, proj.sigma_px.numpy(), 0)
        minor[:, j] = np.where(vis, lo.numpy(), 0)
        major[:, j] = np.where(vis, hi.numpy(), 0)
    return sigma, minor, major, visible


def analyze(
    scene: Scene,
    cameras: Sequence[Camera],
    renderer: Renderer,
    terms: Sequence[str] = ("contribution",),
    progress: ProgressFn | None = None,
    provenance: dict[str, Any] | None = None,
) -> Weights:
    """Compute the requested per-splat, per-view terms."""
    active = check_terms(terms)
    wanted = [TERM_STATISTICS[t] for t in active if t in TERM_STATISTICS]
    start = time.perf_counter()
    extra: dict[str, NDArray[Any]] = {}
    if wanted:
        statistics = getattr(renderer, "statistics", None)
        if statistics is None:
            names = ", ".join(t for t in active if t in TERM_STATISTICS)
            raise BackendUnavailable(
                f"the {renderer.name} backend cannot compute the {names} term(s); "
                "use --backend torch"
            )
        extra = statistics(scene, cameras, wanted, progress=progress)
        contrib = extra.pop("contribution")
    else:
        contrib = renderer.contributions(scene, cameras, progress=progress)
    sigma, minor, major, visible = footprints(scene, cameras)
    meta: dict[str, Any] = {
        "terms": active,
        "backend": renderer.name,
        "backend_detail": renderer.describe(),
        "count": scene.count,
        "views": len(cameras),
        "wall_seconds": round(time.perf_counter() - start, 3),
    }
    if provenance:
        meta["provenance"] = provenance
    return Weights(
        contribution=contrib.astype(np.float32),
        sigma_px=sigma,
        sigma_px_minor=minor,
        sigma_px_major=major,
        visible=visible,
        view_names=[c.name for c in cameras],
        meta=meta,
        **extra,
    )
