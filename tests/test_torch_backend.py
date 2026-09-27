"""P0.6: the torch backend, the correctness oracle.

The strongest check here is agreement with ``naive_composite``: a literal,
per-pixel transcription of gsplat's CUDA inner loop
(``RasterizeToPixels3DGSFwd.cu``) with no tiles, no chunks and no bounding
boxes. It considers every splat in front of the camera for every pixel, so
it also proves the oracle's culling drops nothing that would have rendered.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

import psplat.render.torch_backend as tb
from psplat.io.cameras import Camera, look_at
from psplat.io.scene import Scene
from psplat.render.base import Renderer
from psplat.render.project import ALPHA_MIN, NEAR, SceneTensors, project
from psplat.render.sh import eval_sh
from psplat.render.torch_backend import TorchRenderer

F64 = torch.float64


def camera(w: int = 40, h: int = 30, eye: tuple[float, float, float] = (0.0, -4.0, 0.0)) -> Camera:
    return Camera("c", w, h, 36.0, 37.0, w / 2 - 0.3, h / 2 + 0.2, look_at(eye, (0.0, 0.0, 0.0)))


def logit(p: float) -> float:
    return math.log(p / (1 - p))


def scene_of(means: list[list[float]], sigma: list[float], opacity: list[float]) -> Scene:
    n = len(means)
    f = np.float32
    return Scene(
        means=np.asarray(means, f),
        scales=np.log(np.repeat(np.asarray(sigma)[:, None], 3, 1)).astype(f),
        quats=np.tile(np.array([1, 0, 0, 0], f), (n, 1)),
        opacities=np.array([logit(o) for o in opacity], f),
        sh0=np.zeros((n, 3), f),
        shN=np.zeros((n, 0, 3), f),
    )


def random_scene(n: int, seed: int, degree: int = 1, opaque: bool = False) -> Scene:
    """Random splats. ``opaque`` makes them dense enough that most pixels stop early."""
    rng = np.random.default_rng(seed)
    f = np.float32
    return Scene(
        means=rng.uniform(-1.2, 1.2, (n, 3)).astype(f),
        scales=rng.normal(-1.3 if opaque else -2.2, 0.4, (n, 3)).astype(f),
        quats=rng.normal(size=(n, 4)).astype(f),
        opacities=rng.normal(4.0 if opaque else 1.0, 1.5 if opaque else 2.0, n).astype(f),
        sh0=rng.normal(0, 1, (n, 3)).astype(f),
        shN=rng.normal(0, 0.3, (n, (degree + 1) ** 2 - 1, 3)).astype(f),
    )


def naive_composite(
    scene: Scene, cam: Camera, alpha_max: float = 0.999
) -> tuple[np.ndarray, np.ndarray]:
    """Per-pixel front-to-back loop, exactly as gsplat's forward kernel runs it."""
    s = SceneTensors.from_scene(scene, "cpu", F64)
    proj = project(s, cam)
    usable = (proj.depths >= NEAR) & torch.isfinite(proj.conics).all(-1)
    order = [int(i) for i in torch.argsort(proj.depths, stable=True) if usable[i]]
    colours = (eval_sh(s.sh, s.means - torch.as_tensor(cam.position)) + 0.5).clamp_min(0)
    m2d, con, opa = proj.means2d.numpy(), proj.conics.numpy(), proj.opacities.numpy()
    col = colours.numpy()
    contrib = np.zeros(scene.count)
    image = np.zeros((cam.height, cam.width, 3))
    for i in range(cam.height):
        for j in range(cam.width):
            px, py = j + 0.5, i + 0.5
            t = 1.0
            for g in order:
                dx, dy = m2d[g, 0] - px, m2d[g, 1] - py
                sigma = 0.5 * (con[g, 0] * dx * dx + con[g, 2] * dy * dy) + con[g, 1] * dx * dy
                alpha = min(alpha_max, opa[g] * math.exp(-sigma))
                if sigma < 0 or alpha < ALPHA_MIN:
                    continue
                next_t = t * (1 - alpha)
                if next_t <= 1e-4:
                    break
                contrib[g] += alpha * t
                image[i, j] += col[g] * alpha * t
                t = next_t
    return contrib, image


# Agreement with the literal per-pixel loop -------------------------------------------


@pytest.mark.parametrize(("seed", "opaque"), [(0, False), (1, False), (2, True), (3, True)])
def test_matches_naive_per_pixel_loop(seed: int, opaque: bool) -> None:
    scene = random_scene(60 if opaque else 40, seed, opaque=opaque)
    cam = camera()
    want_c, want_img = naive_composite(scene, cam)
    oracle = TorchRenderer("cpu", dtype=F64)
    splats = SceneTensors.from_scene(scene, "cpu", F64)
    image, alpha, contrib = oracle.rasterize(splats, cam, want_image=True)
    if opaque:
        # Many pixels must hit early termination, or this case tests nothing new.
        assert float((alpha > 1 - 1e-3).float().mean()) > 0.15
    np.testing.assert_allclose(contrib.numpy(), want_c, rtol=1e-9, atol=1e-12)
    assert image is not None
    np.testing.assert_allclose(image.numpy(), want_img, rtol=1e-9, atol=1e-12)


def test_small_tiles_and_chunks_change_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    # Exercise tile boundaries and the transmittance carried across chunks.
    scene = random_scene(60, 7, opaque=True)
    cam = camera(37, 29)  # sizes that do not divide by the tile
    splats = SceneTensors.from_scene(scene, "cpu", F64)
    _, alpha_ref, ref = TorchRenderer("cpu", dtype=F64).rasterize(splats, cam, want_image=False)
    monkeypatch.setattr(tb, "CHUNK", 3)
    _, alpha_small, small = TorchRenderer("cpu", dtype=F64, tile=5).rasterize(
        splats, cam, want_image=False
    )
    torch.testing.assert_close(small, ref, rtol=1e-12, atol=1e-14)
    torch.testing.assert_close(alpha_small, alpha_ref, rtol=1e-12, atol=1e-14)
    want, _ = naive_composite(scene, cam)
    np.testing.assert_allclose(small.numpy(), want, rtol=1e-9, atol=1e-12)


def test_inria_alpha_cap_is_one_argument_away() -> None:
    scene = scene_of([[0, 0, 0]], [0.3], [0.99999])
    cam = camera()
    want, _ = naive_composite(scene, cam, alpha_max=0.99)
    got = TorchRenderer("cpu", dtype=F64, alpha_max=0.99).contributions(scene, [cam])
    np.testing.assert_allclose(got[:, 0], want, rtol=1e-6)


# Analytic cases -------------------------------------------------------------------------


def test_single_splat_contribution_is_its_alpha_mass() -> None:
    # Alone, T is 1 everywhere, so the contribution is the sum of alpha over
    # pixels: close to opacity * 2π σx σy for a footprint well inside the image.
    scene = scene_of([[0, 0, 0]], [0.12], [0.5])
    cam = camera(64, 64)
    proj = project(SceneTensors.from_scene(scene, "cpu", F64), cam)
    sx, sy = math.sqrt(proj.cov2d[0, 0, 0]), math.sqrt(proj.cov2d[0, 1, 1])
    got = TorchRenderer("cpu", dtype=F64).contributions(scene, [cam])[0, 0]
    assert got == pytest.approx(0.5 * 2 * math.pi * sx * sy, rel=0.02)


def test_two_overlapping_splats_composite_front_to_back() -> None:
    # Identical footprints; only depth differs. The front splat takes its
    # alpha mass, the back one only what the front lets through.
    near, far = [0.0, -0.5, 0.0], [0.0, 0.5, 0.0]
    oracle = TorchRenderer("cpu", dtype=F64)
    cam = camera()
    c = oracle.contributions(scene_of([far, near], [0.2, 0.2], [0.6, 0.6]), [cam])[:, 0]
    assert c[1] > c[0] > 0  # index 1 is nearer the camera
    swapped = oracle.contributions(scene_of([near, far], [0.2, 0.2], [0.6, 0.6]), [cam])[:, 0]
    assert swapped[0] > swapped[1] > 0
    np.testing.assert_allclose(sorted(c), sorted(swapped), rtol=0.05)


def test_full_occluder_gives_zero_contribution_but_stays_visible() -> None:
    # Three near-opaque, wide splats in front: after two, T is about 1e-6
    # (0.001²), so the third would breach 1e-4 and stops the pixel; the
    # small splat behind deposits exactly nothing.
    cam = camera()
    target = [0.0, 0.8, 0.0]
    occluders = [[0.0, -0.6, 0.0], [0.0, -0.4, 0.0], [0.0, -0.2, 0.0]]
    scene = scene_of([target, *occluders], [0.02, 1.0, 1.0, 1.0], [0.9, 0.99999, 0.99999, 0.99999])
    splats = SceneTensors.from_scene(scene, "cpu", F64)
    c = TorchRenderer("cpu", dtype=F64).contributions(scene, [cam])[:, 0]
    assert c[0] == 0.0
    assert bool(project(splats, cam).visible[0])


def test_off_screen_and_behind_splats_contribute_nothing() -> None:
    cam = camera()
    scene = scene_of([[6.0, 0.0, 0.0], [0.0, -8.0, 0.0], [0.0, 0.0, 0.0]], [0.05] * 3, [0.9] * 3)
    c = TorchRenderer("cpu", dtype=F64).contributions(scene, [cam])[:, 0]
    assert c[0] == 0.0 and c[1] == 0.0 and c[2] > 0


def test_render_colour_and_alpha() -> None:
    scene = scene_of([[0, 0, 0]], [0.3], [0.8])
    scene.sh0[:] = [1.0, 0.0, -1.0]
    cam = camera(41, 31)
    result = TorchRenderer("cpu", dtype=F64).render(scene, cam)
    colour = np.clip(np.array([1.0, 0.0, -1.0]) * 0.28209479177387814 + 0.5, 0, None)
    centre = result.image[15, 20]
    a = result.alpha[15, 20]
    np.testing.assert_allclose(centre, colour * a, rtol=1e-5)
    assert result.image.dtype == np.float32 and result.alpha.max() <= 0.999 + 1e-6


def test_background_fills_uncovered_pixels() -> None:
    scene = scene_of([[0, 0, 0]], [0.05], [0.9])
    result = TorchRenderer("cpu", background=(0.2, 0.4, 0.6)).render(scene, camera())
    np.testing.assert_allclose(result.image[0, 0], [0.2, 0.4, 0.6], rtol=1e-6)


# Protocol behaviour -----------------------------------------------------------------------


def test_protocol_shapes_determinism_and_estimate() -> None:
    oracle = TorchRenderer("cpu")
    assert isinstance(oracle, Renderer)
    scene = random_scene(80, 3)
    cams = [camera(eye=(4 * math.cos(a), 4 * math.sin(a), 0.3)) for a in np.linspace(0, 6, 5)]
    seen = []
    first = oracle.contributions(scene, cams, progress=lambda done, total: seen.append(done))
    assert first.shape == (80, 5) and first.dtype == np.float32
    assert seen == [1, 2, 3, 4, 5]
    np.testing.assert_array_equal(first, oracle.contributions(scene, cams))
    estimate = oracle.estimate_seconds(scene, cams)
    assert estimate is not None and 0 < estimate < 60
    assert oracle.describe() == "torch on cpu"


def test_empty_view_and_empty_scene() -> None:
    oracle = TorchRenderer("cpu")
    nothing = scene_of([[0.0, -9.0, 0.0]], [0.05], [0.9])  # behind the camera
    assert oracle.contributions(nothing, [camera()]).sum() == 0
    empty = random_scene(0, 0)
    assert oracle.contributions(empty, [camera()]).shape == (0, 1)
    assert oracle.render(empty, camera()).image.sum() == 0
