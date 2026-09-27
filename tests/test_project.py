"""P0.5: shared projection, SH evaluation and the null backend.

Where gsplat's pure-PyTorch reference functions are importable (they run on
CPU without CUDA), projection and SH are checked against them directly.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from psplat.io.cameras import Camera, look_at
from psplat.io.scene import Scene
from psplat.render.base import BackendUnavailable, Renderer
from psplat.render.null_backend import NullRenderer
from psplat.render.project import (
    ALPHA_MAX,
    EPS2D,
    SceneTensors,
    covariance_3d,
    gaussian_alpha,
    project,
    quat_to_rotation,
)
from psplat.render.sh import eval_sh

try:
    from gsplat.cuda._torch_impl import (  # type: ignore[import-not-found,unused-ignore]
        _fully_fused_projection,
        _spherical_harmonics,
    )

    HAVE_GSPLAT = True
except ImportError:  # pragma: no cover - depends on the environment
    HAVE_GSPLAT = False

needs_gsplat = pytest.mark.skipif(not HAVE_GSPLAT, reason="gsplat reference not installed")


def camera(width: int = 80, height: int = 60, eye: tuple[float, ...] = (0.0, -4.0, 0.0)) -> Camera:
    return Camera(
        name="c",
        width=width,
        height=height,
        fx=70.0,
        fy=72.0,
        cx=40.0,
        cy=30.0,
        w2c=look_at(eye, (0.0, 0.0, 0.0)),
    )


def scene_of(
    means: list[list[float]],
    sigma: float | list[float] = 0.05,
    opacity: float = 0.9,
    quats: np.ndarray | None = None,
    sh_degree: int = 0,
) -> Scene:
    n = len(means)
    f = np.float32
    s = np.broadcast_to(np.asarray(sigma, np.float64), (n, 3))
    q = np.tile([1.0, 0.0, 0.0, 0.0], (n, 1)) if quats is None else quats
    return Scene(
        means=np.asarray(means, f),
        scales=np.log(s).astype(f),
        quats=np.asarray(q, f),
        opacities=np.full(n, math.log(opacity / (1 - opacity)), f),
        sh0=np.zeros((n, 3), f),
        shN=np.zeros((n, (sh_degree + 1) ** 2 - 1, 3), f),
    )


def splats(scene: Scene) -> SceneTensors:
    return SceneTensors.from_scene(scene, "cpu", torch.float64)


# Rotations and covariance ------------------------------------------------------------


def test_quaternion_rotation() -> None:
    q = torch.tensor([[1.0, 0, 0, 0], [math.cos(math.pi / 4), 0, 0, math.sin(math.pi / 4)]])
    r = quat_to_rotation(q)
    torch.testing.assert_close(r[0], torch.eye(3))
    torch.testing.assert_close(r[1] @ torch.tensor([1.0, 0, 0]), torch.tensor([0.0, 1.0, 0.0]))
    rand = torch.nn.functional.normalize(torch.randn(50, 4, dtype=torch.float64), dim=-1)
    rr = quat_to_rotation(rand)
    torch.testing.assert_close(
        rr @ rr.transpose(-1, -2), torch.eye(3, dtype=torch.float64).expand(50, 3, 3)
    )


def test_covariance_from_scale_and_rotation() -> None:
    scales = torch.tensor([[1.0, 2.0, 3.0]])
    cov = covariance_3d(scales, torch.tensor([[1.0, 0, 0, 0]]))
    torch.testing.assert_close(cov[0], torch.diag(torch.tensor([1.0, 4.0, 9.0])))
    about_z = torch.tensor([[math.cos(math.pi / 4), 0, 0, math.sin(math.pi / 4)]])
    swapped = covariance_3d(scales, about_z)[0]
    assert swapped[0, 0].item() == pytest.approx(4.0) and swapped[1, 1].item() == pytest.approx(1.0)


# Projection ---------------------------------------------------------------------------


def test_on_axis_splat_projects_to_principal_point_with_expected_footprint() -> None:
    cam = camera()
    proj = project(splats(scene_of([[0.0, 0.0, 0.0]], sigma=0.1)), cam)
    np.testing.assert_allclose(proj.means2d[0].numpy(), [cam.cx, cam.cy], atol=1e-12)
    assert proj.depths[0].item() == pytest.approx(4.0)
    expected_x = (cam.fx * 0.1 / 4.0) ** 2 + EPS2D
    expected_y = (cam.fy * 0.1 / 4.0) ** 2 + EPS2D
    assert proj.cov2d[0, 0, 0].item() == pytest.approx(expected_x)
    assert proj.cov2d[0, 1, 1].item() == pytest.approx(expected_y)
    assert proj.sigma_px[0].item() == pytest.approx((expected_x * expected_y) ** 0.25)
    minor, major = proj.sigma_px_axes()
    assert minor[0].item() == pytest.approx(math.sqrt(expected_x))
    assert major[0].item() == pytest.approx(math.sqrt(expected_y))
    conic = proj.conics[0]
    inv = torch.tensor([[conic[0], conic[1]], [conic[1], conic[2]]], dtype=torch.float64)
    torch.testing.assert_close(inv @ proj.cov2d[0], torch.eye(2, dtype=torch.float64))
    assert bool(proj.visible[0])


@needs_gsplat
def test_projection_matches_gsplat_reference() -> None:
    rng = np.random.default_rng(5)
    n = 300
    quats = rng.normal(size=(n, 4))
    sigmas = np.exp(rng.normal(-2.5, 0.5, (n, 3))).tolist()
    scene = scene_of(rng.normal(size=(n, 3)).tolist(), sigma=sigmas, quats=quats)
    cam = camera(160, 120, eye=(1.0, -5.0, 0.7))
    s = splats(scene)
    ours = project(s, cam)
    covars = covariance_3d(s.scales, s.quats)
    viewmat = torch.as_tensor(cam.w2c)[None]
    K = torch.as_tensor(cam.K)[None]
    radii, means2d, depths, conics, _ = _fully_fused_projection(
        s.means, covars, viewmat, K, cam.width, cam.height
    )
    torch.testing.assert_close(ours.means2d, means2d[0])
    torch.testing.assert_close(ours.depths, depths[0])
    torch.testing.assert_close(ours.conics, conics[0])
    # gsplat's reference culls with a fixed 3.33 sigma radius; ours is
    # opacity-aware and is never larger, so anything we keep, it keeps.
    theirs_visible = (radii[0] > 0).all(-1)
    assert bool((ours.visible & ~theirs_visible).sum() == 0)


def test_off_axis_jacobian_is_clamped_like_gsplat() -> None:
    # A splat far outside the field of view must not produce an exploding
    # footprint; the Jacobian is evaluated at 1.3x the half field of view.
    cam = camera()
    far_right = project(splats(scene_of([[30.0, 0.0, 0.0]], sigma=0.1)), cam)
    assert torch.isfinite(far_right.cov2d).all()
    assert not bool(far_right.visible[0])


def test_visibility_rules() -> None:
    cam = camera()  # at y = -4 looking along +y, image x is world +x
    behind = [0.0, -6.0, 0.0]
    small_offscreen = [3.0, 0.0, 0.0]  # just beyond the right edge
    big_offscreen = [3.0, 0.0, 0.0]
    proj = project(splats(scene_of([behind, small_offscreen], sigma=0.02)), cam)
    assert not proj.visible.any()
    big = project(splats(scene_of([big_offscreen], sigma=1.0)), cam)
    assert bool(big.visible[0])
    faint = project(splats(scene_of([[0.0, 0.0, 0.0]], opacity=0.5 / 255)), cam)
    assert not bool(faint.visible[0])


# Alpha --------------------------------------------------------------------------------


def test_gaussian_alpha_follows_rasteriser_rules() -> None:
    means2d = torch.tensor([[10.0, 10.0]], dtype=torch.float64)
    conics = torch.tensor([[1.0, 0.0, 1.0]], dtype=torch.float64)
    pix = torch.tensor([[10.0, 10.0], [11.0, 10.0], [30.0, 30.0]], dtype=torch.float64)
    opaque = gaussian_alpha(conics, means2d, torch.tensor([1.0], dtype=torch.float64), pix)
    assert opaque[0, 0].item() == pytest.approx(ALPHA_MAX)  # capped
    assert opaque[0, 1].item() == pytest.approx(math.exp(-0.5), rel=1e-12)
    assert opaque[0, 2].item() == 0.0  # below 1/255: skipped
    indefinite = torch.tensor([[-1.0, 0.0, -1.0]], dtype=torch.float64)
    skipped = gaussian_alpha(indefinite, means2d, torch.tensor([1.0], dtype=torch.float64), pix)
    assert skipped[0, 1].item() == 0.0  # negative quadratic form: skipped


# Spherical harmonics --------------------------------------------------------------------


def test_sh_degree_zero_is_the_dc_term() -> None:
    coeffs = torch.tensor([[[1.0, 2.0, 3.0]]])
    out = eval_sh(coeffs, torch.tensor([[0.0, 0.0, 1.0]]))
    torch.testing.assert_close(out, coeffs[:, 0] * 0.28209479177387814)


@needs_gsplat
@pytest.mark.parametrize("degree", [0, 1, 2, 3])
def test_sh_matches_gsplat_reference(degree: int) -> None:
    gen = torch.Generator().manual_seed(degree)
    k = (degree + 1) ** 2
    coeffs = torch.randn(64, k, 3, generator=gen, dtype=torch.float64)
    dirs = torch.randn(64, 3, generator=gen, dtype=torch.float64)
    theirs = _spherical_harmonics(degree, dirs, coeffs)
    torch.testing.assert_close(eval_sh(coeffs, dirs), theirs)


# Null backend ----------------------------------------------------------------------------


def test_null_backend_satisfies_the_protocol_and_refuses_to_render() -> None:
    null = NullRenderer()
    assert isinstance(null, Renderer)
    scene = scene_of([[0.0, 0.0, 0.0]])
    with pytest.raises(BackendUnavailable, match="--backend torch"):
        null.contributions(scene, [camera()])
    with pytest.raises(BackendUnavailable):
        null.render(scene, camera())
    assert null.estimate_seconds(scene, [camera()]) is None
