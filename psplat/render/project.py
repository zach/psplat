"""Shared projection of 3D Gaussians into a camera, in pure PyTorch.

Every backend and every analysis term that needs a splat's screen-space
footprint uses this module, so "footprint" means one thing everywhere. The
footprint is also what the CSF term later reads as spatial frequency
(handoff §1.3), which is why the perceptual weighting costs almost nothing
beyond the projection pass.

Conventions match gsplat's classic 3DGS rasteriser. They were read from
the gsplat 1.5.3 source release (``ProjectionEWA3DGSFused.cu``,
``RasterizeToPixels3DGSFwd.cu``, ``rendering.py``), not recalled:

- camera frame is OpenCV (+x right, +y down, +z forward);
- a camera point projects to ``(fx * x / z + cx, fy * y / z + cy)`` and
  pixel ``(i, j)`` is sampled at its centre ``(j + 0.5, i + 0.5)``;
- the 2D covariance is ``J W Σ Wᵀ Jᵀ`` with the perspective Jacobian
  evaluated at a position clamped to 1.3× the field of view, then dilated
  by ``EPS2D = 0.3`` pixels² (the anti-aliasing low-pass);
- splats with depth below ``NEAR = 0.01`` are culled;
- per pixel, alpha is ``min(0.999, opacity · exp(-q/2))``; alphas below
  1/255, or with ``q < 0``, are skipped; a pixel stops, exclusively, at the
  first splat that would take transmittance to ``1e-4`` or below.

The Inria reference differs in two constants: it caps alpha at 0.99 and
stops when transmittance would fall strictly below 1e-4. ``ALPHA_MAX`` is a
parameter wherever it matters, so that variant is one argument away.

The screen-space extent is opacity-aware: the ellipse where
``opacity · g ≥ 1/255``. gsplat does the same but caps it at 3.33σ, which
only binds for opacity above 1, and rounds radii up to whole pixels. So
culling here is lossless, and gsplat's cull agrees with it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

from psplat.io.cameras import Camera
from psplat.io.scene import Scene

NEAR = 0.01
EPS2D = 0.3
ALPHA_MIN = 1.0 / 255.0
ALPHA_MAX = 0.999  # gsplat; the Inria reference uses 0.99
ALPHA_MAX_INRIA = 0.99
# A pixel stops at the first splat that would bring transmittance to T_MIN
# or below, and that splat is not accumulated.
T_MIN = 1e-4
# Fraction of the half field of view the Jacobian may be evaluated beyond.
FOV_CLAMP = 0.3


@dataclass(frozen=True)
class SceneTensors:
    """Activated splat parameters as tensors on one device."""

    means: Tensor  # (N, 3)
    scales: Tensor  # (N, 3) world-space standard deviations
    quats: Tensor  # (N, 4) unit (w, x, y, z)
    opacities: Tensor  # (N,) in [0, 1]
    sh: Tensor  # (N, (degree + 1) ** 2, 3), DC first

    @property
    def count(self) -> int:
        return int(self.means.shape[0])

    @staticmethod
    def from_scene(
        scene: Scene, device: torch.device | str = "cpu", dtype: torch.dtype = torch.float32
    ) -> SceneTensors:
        def t(a: np.ndarray) -> Tensor:
            return torch.as_tensor(np.ascontiguousarray(a), device=device, dtype=dtype)

        sh = np.concatenate([scene.sh0[:, None, :], scene.shN], axis=1)
        return SceneTensors(
            means=t(scene.means),
            scales=t(scene.activated_scales()),
            quats=t(scene.normalized_quats()),
            opacities=t(scene.activated_opacities()),
            sh=t(sh),
        )


def quat_to_rotation(quats: Tensor) -> Tensor:
    """``(N, 4)`` unit quaternions ``(w, x, y, z)`` to ``(N, 3, 3)`` rotations."""
    w, x, y, z = quats.unbind(-1)
    return torch.stack(
        [
            1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
            2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
            2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
        ],
        dim=-1,
    ).reshape(*quats.shape[:-1], 3, 3)  # fmt: skip


def covariance_3d(scales: Tensor, quats: Tensor) -> Tensor:
    """World-space covariance ``R S Sᵀ Rᵀ`` for each splat, ``(N, 3, 3)``."""
    m = quat_to_rotation(quats) * scales[..., None, :]
    return m @ m.transpose(-1, -2)


@dataclass(frozen=True)
class Projection:
    """One camera's view of every splat. All tensors have leading size N."""

    means2d: Tensor  # (N, 2) pixel coordinates
    depths: Tensor  # (N,) camera-space z
    cov2d: Tensor  # (N, 2, 2) dilated screen covariance, pixels²
    conics: Tensor  # (N, 3) inverse covariance (a, b, c) of [[a, b], [b, c]]
    opacities: Tensor  # (N,)
    bbox: Tensor  # (N, 4) x_min, y_min, x_max, y_max of the alpha ≥ 1/255 ellipse
    visible: Tensor  # (N,) bool: in front, finite, and the extent overlaps a pixel centre

    @property
    def sigma_px(self) -> Tensor:
        """Isotropic-equivalent footprint: the geometric mean of the axis σs."""
        det = (self.cov2d[:, 0, 0] * self.cov2d[:, 1, 1] - self.cov2d[:, 0, 1] ** 2).clamp_min(0)
        return det.sqrt().sqrt()

    def sigma_px_axes(self) -> tuple[Tensor, Tensor]:
        """Footprint σ along the minor and major screen axes."""
        a, b, c = self.cov2d[:, 0, 0], self.cov2d[:, 0, 1], self.cov2d[:, 1, 1]
        mid = 0.5 * (a + c)
        spread = torch.sqrt((0.5 * (a - c)) ** 2 + b * b)
        return (mid - spread).clamp_min(0).sqrt(), (mid + spread).sqrt()


def camera_tensors(camera: Camera, device: torch.device | str, dtype: torch.dtype) -> Tensor:
    return torch.as_tensor(camera.w2c, device=device, dtype=dtype)


def project(splats: SceneTensors, camera: Camera) -> Projection:
    """Project every splat into ``camera``. Invisible splats are flagged, not dropped."""
    device, dtype = splats.means.device, splats.means.dtype
    w2c = camera_tensors(camera, device, dtype)
    rot, trans = w2c[:3, :3], w2c[:3, 3]
    p = splats.means @ rot.T + trans
    x, y, z = p.unbind(-1)
    in_front = z >= NEAR
    zs = torch.where(in_front, z, torch.ones_like(z))  # keep culled splats finite

    fx, fy, cx, cy = camera.fx, camera.fy, camera.cx, camera.cy
    width, height = camera.width, camera.height
    tan_x, tan_y = 0.5 * width / fx, 0.5 * height / fy
    tx = zs * torch.clamp(
        x / zs, -(cx / fx + FOV_CLAMP * tan_x), (width - cx) / fx + FOV_CLAMP * tan_x
    )
    ty = zs * torch.clamp(
        y / zs, -(cy / fy + FOV_CLAMP * tan_y), (height - cy) / fy + FOV_CLAMP * tan_y
    )
    zero = torch.zeros_like(zs)
    jac = torch.stack(
        [fx / zs, zero, -fx * tx / zs**2, zero, fy / zs, -fy * ty / zs**2], dim=-1
    ).reshape(-1, 2, 3)
    cov_cam = rot @ covariance_3d(splats.scales, splats.quats) @ rot.T
    cov2d = jac @ cov_cam @ jac.transpose(-1, -2)
    cov2d = cov2d + EPS2D * torch.eye(2, device=device, dtype=dtype)

    a, b, c = cov2d[:, 0, 0], cov2d[:, 0, 1], cov2d[:, 1, 1]
    det = a * c - b * b
    ok_det = det > 0
    det_safe = torch.where(ok_det, det, torch.ones_like(det))
    conics = torch.stack([c / det_safe, -b / det_safe, a / det_safe], dim=-1)

    means2d = torch.stack([fx * x / zs + cx, fy * y / zs + cy], dim=-1)

    # Opacity-aware extent: alpha = o·exp(-q/2) ≥ 1/255  ⇔  q ≤ 2·ln(255·o).
    o = splats.opacities
    q_max = 2.0 * torch.log((o / ALPHA_MIN).clamp_min(1.0))
    half_x = torch.sqrt(q_max * a.clamp_min(0))
    half_y = torch.sqrt(q_max * c.clamp_min(0))
    bbox = torch.stack(
        [means2d[:, 0] - half_x, means2d[:, 1] - half_y, means2d[:, 0] + half_x,
         means2d[:, 1] + half_y],
        dim=-1,
    )  # fmt: skip
    overlaps = (
        (bbox[:, 2] >= 0.5)
        & (bbox[:, 0] <= width - 0.5)
        & (bbox[:, 3] >= 0.5)
        & (bbox[:, 1] <= height - 0.5)
    )
    finite = torch.isfinite(means2d).all(-1) & torch.isfinite(conics).all(-1)
    visible = in_front & ok_det & (q_max > 0) & overlaps & finite
    return Projection(
        means2d=means2d,
        depths=z,
        cov2d=cov2d,
        conics=conics,
        opacities=o,
        bbox=bbox,
        visible=visible,
    )


def pixel_centres(camera: Camera, device: torch.device | str, dtype: torch.dtype) -> Tensor:
    """``(H, W, 2)`` pixel-centre coordinates ``(x, y)``."""
    ys = torch.arange(camera.height, device=device, dtype=dtype) + 0.5
    xs = torch.arange(camera.width, device=device, dtype=dtype) + 0.5
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([gx, gy], dim=-1)


def gaussian_alpha(
    conics: Tensor,
    means2d: Tensor,
    opacities: Tensor,
    pix: Tensor,
    alpha_max: float = ALPHA_MAX,
) -> Tensor:
    """Alpha of each splat at each pixel, ``(K, P)``, with the rasteriser's rules.

    ``pix`` is ``(P, 2)``. Alpha is capped at ``alpha_max``. Entries the
    rasteriser skips (alpha below 1/255, or a negative quadratic form from a
    numerically indefinite conic) are exactly zero, which leaves transmittance
    unchanged, just as skipping does.
    """
    d = means2d[:, None, :] - pix[None, :, :]
    dx, dy = d[..., 0], d[..., 1]
    a, b, c = conics[:, 0:1], conics[:, 1:2], conics[:, 2:3]
    sigma = 0.5 * (a * dx * dx + c * dy * dy) + b * dx * dy
    alpha = (opacities[:, None] * torch.exp(-sigma)).clamp_max(alpha_max)
    keep = (sigma >= 0) & (alpha >= ALPHA_MIN)
    return torch.where(keep, alpha, torch.zeros_like(alpha))


def fov_degrees(camera: Camera) -> tuple[float, float]:
    """Horizontal and vertical field of view, for notices and sanity checks."""
    return (
        math.degrees(2 * math.atan(0.5 * camera.width / camera.fx)),
        math.degrees(2 * math.atan(0.5 * camera.height / camera.fy)),
    )
