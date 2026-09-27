"""Real spherical-harmonic colour evaluation, degrees 0 to 3.

Constants and sign conventions follow the Inria reference implementation and
gsplat, so a scene trained by either renders with the same colours here.
"""

from __future__ import annotations

import torch
from torch import Tensor

C0 = 0.28209479177387814
C1 = 0.4886025119029199
C2 = (
    1.0925484305920792,
    -1.0925484305920792,
    0.31539156525252005,
    -1.0925484305920792,
    0.5462742152960396,
)
C3 = (
    -0.5900435899266435,
    2.890611442640554,
    -0.4570457994644658,
    0.3731763325901154,
    -0.4570457994644658,
    1.445305721320277,
    -0.5900435899266435,
)


def eval_sh(coeffs: Tensor, dirs: Tensor) -> Tensor:
    """Evaluate SH colour.

    ``coeffs`` is ``(N, (degree + 1) ** 2, 3)`` with the DC term first;
    ``dirs`` is ``(N, 3)`` view directions (camera to splat), not necessarily
    unit length. Returns ``(N, 3)`` before the +0.5 offset and clamp that
    renderers apply.
    """
    k = coeffs.shape[1]
    result: Tensor = coeffs[:, 0] * C0
    if k == 1:
        return result
    d = dirs / torch.linalg.norm(dirs, dim=-1, keepdim=True).clamp_min(1e-12)
    x, y, z = d[:, 0:1], d[:, 1:2], d[:, 2:3]
    result = result - C1 * y * coeffs[:, 1] + C1 * z * coeffs[:, 2] - C1 * x * coeffs[:, 3]
    if k == 4:
        return result
    xx, yy, zz = x * x, y * y, z * z
    xy, yz, xz = x * y, y * z, x * z
    result = (
        result
        + C2[0] * xy * coeffs[:, 4]
        + C2[1] * yz * coeffs[:, 5]
        + C2[2] * (2.0 * zz - xx - yy) * coeffs[:, 6]
        + C2[3] * xz * coeffs[:, 7]
        + C2[4] * (xx - yy) * coeffs[:, 8]
    )
    if k == 9:
        return result
    full: Tensor = (
        result
        + C3[0] * y * (3.0 * xx - yy) * coeffs[:, 9]
        + C3[1] * xy * z * coeffs[:, 10]
        + C3[2] * y * (4.0 * zz - xx - yy) * coeffs[:, 11]
        + C3[3] * z * (2.0 * zz - 3.0 * xx - 3.0 * yy) * coeffs[:, 12]
        + C3[4] * x * (4.0 * zz - xx - yy) * coeffs[:, 13]
        + C3[5] * z * (xx - yy) * coeffs[:, 14]
        + C3[6] * x * (xx - 3.0 * yy) * coeffs[:, 15]
    )
    return full
