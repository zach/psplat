"""Portable PyTorch backend: the correctness oracle (handoff §0.2, §8.1a).

A direct transcription of gsplat's classic tile rasteriser in PyTorch, so it
runs on CUDA, Apple MPS or plain CPU and every other backend can be tested
against it. Per view it:

1. projects every splat (:mod:`psplat.render.project`);
2. sorts visible splats front to back by depth;
3. bins them into 16×16 pixel tiles by expanding (tile, splat) pairs and
   stable-sorting by tile, which keeps depth order within a tile, exactly
   as gsplat's ``isect_tiles`` does;
4. for each tile, composites front to back: ``alpha`` per splat and pixel,
   transmittance ``T`` as a running product, and a pixel stops, exclusively,
   at the first splat that would bring ``T`` to ``1e-4`` or below.

Each splat's contribution to the view is the sum over pixels of the
``alpha · T`` it deposited. Splats are processed in bounded chunks with
``T`` carried across chunks, so memory stays flat however many splats pile
into one tile.

The same pass can gather the extra per-splat statistics the baseline
importance scores need (:data:`STATISTICS`, P1.1), summed over views:
transmittance over the pixels a splat is composited into (LightGaussian),
the number of pixels where it is the largest contributor (Mini-Splatting),
and a 6×6 Fisher block over its mean and log-scale (PUP 3D-GS). They cost
nothing unless asked for. ``docs/baselines.md`` gives the definitions.

Speed, measured in this repository's container (4 CPU threads): about 3–6 s
per 1237×822 view for 300k splats. The cost is the alpha evaluation itself
(tile pixels × splat pairs), not Python overhead, so a 3M-splat scene over
~160 views is hours on CPU rather than the tens of minutes the handoff
hoped for. It is the guaranteed path and the oracle; gsplat is the fast one.
"""

from __future__ import annotations

import functools
import math
import os
import time
import warnings
from collections.abc import Collection, Sequence
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor

from psplat.io.cameras import Camera
from psplat.io.scene import Scene
from psplat.render.base import ProgressFn, RenderResult
from psplat.render.project import (
    ALPHA_MAX,
    T_MIN,
    Projection,
    SceneTensors,
    gaussian_alpha,
    project,
)
from psplat.render.sh import eval_sh

TILE = 16
# Splats composited per step within a tile. It bounds memory at CHUNK × TILE²,
# and it is the granularity at which a tile whose pixels have all stopped
# skips the rest of its splats. Measured on 300k-splat scenes at 1237×822 on
# CPU: 256 was fastest for both an opaque surface (3.2 s vs 13.5 s at 4096)
# and translucent clutter (5.8 s vs 7.1 s).
CHUNK = 256

# Per-splat statistics, summed over views, that rasterize() can accumulate
# alongside the contribution. Shapes: (N,), (N,) and (N, 6, 6).
STATISTICS = ("hit_transmittance", "top_count", "fisher")
# Parameters the Fisher block is taken over: world-space mean, then log-scale.
FISHER_PARAMS = ("x", "y", "z", "log_sx", "log_sy", "log_sz")


def _usable(device: str) -> bool:
    """Whether a small computation actually runs on ``device``.

    ``is_available()`` is not enough: GitHub's macOS runners report MPS as
    available, then fail every allocation with "MPS backend out of memory".
    """
    try:
        x = torch.ones(4096, device=device)
        return float((x * 2).sum().item()) == 8192.0
    except RuntimeError as exc:
        first = str(exc).splitlines()[0] if str(exc) else type(exc).__name__
        warnings.warn(
            f"{device} reports available but is unusable ({first}); skipping it", stacklevel=3
        )
        return False


@functools.cache
def default_device() -> torch.device:
    """``PSPLAT_DEVICE`` if set, else CUDA, else Apple MPS, else CPU, whichever works."""
    forced = os.environ.get("PSPLAT_DEVICE")
    if forced:
        return torch.device(forced)
    if torch.cuda.is_available() and _usable("cuda"):
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available() and _usable("mps"):
        return torch.device("mps")
    return torch.device("cpu")


@dataclass
class _Bins:
    """Visible splats of one view, depth-sorted and binned into tiles."""

    ids: Tensor  # (M,) splat indices, front to back
    proj: Projection  # projection of all splats
    tile_splats: Tensor  # (pairs,) positions into ``ids``, grouped by tile, depth order kept
    tile_start: Tensor  # (tiles + 1,) offsets into ``tile_splats``
    tiles_x: int
    tiles_y: int

    @property
    def pairs(self) -> int:
        return int(self.tile_splats.shape[0])


class TorchRenderer:
    """Pure-PyTorch rasteriser; see the module docstring."""

    name = "torch"

    def __init__(
        self,
        device: str | torch.device | None = None,
        *,
        dtype: torch.dtype = torch.float32,
        alpha_max: float = ALPHA_MAX,
        background: tuple[float, float, float] = (0.0, 0.0, 0.0),
        tile: int = TILE,
    ) -> None:
        self.device = torch.device(device) if device is not None else default_device()
        self.dtype = dtype
        self.alpha_max = alpha_max
        self.background = background
        self.tile = tile
        self._seconds_per_pixel_pair: float | None = None

    def describe(self) -> str:
        return f"torch on {self.device.type}"

    # Public protocol ---------------------------------------------------------

    def render(self, scene: Scene, camera: Camera) -> RenderResult:
        splats = SceneTensors.from_scene(scene, self.device, self.dtype)
        image, alpha, _ = self.rasterize(splats, camera, want_image=True)
        assert image is not None
        return RenderResult(
            image=image.cpu().numpy().astype(np.float32),
            alpha=alpha.cpu().numpy().astype(np.float32),
        )

    def contributions(
        self,
        scene: Scene,
        cameras: Sequence[Camera],
        progress: ProgressFn | None = None,
    ) -> NDArray[np.float32]:
        return self.statistics(scene, cameras, (), progress)["contribution"]

    def statistics(
        self,
        scene: Scene,
        cameras: Sequence[Camera],
        stats: Collection[str],
        progress: ProgressFn | None = None,
    ) -> dict[str, NDArray[Any]]:
        """The ``(N, V)`` contribution plus the requested :data:`STATISTICS`.

        Each statistic is summed over ``cameras``: ``hit_transmittance`` and
        ``fisher`` as float32, ``top_count`` as int64.
        """
        unknown = set(stats) - set(STATISTICS)
        if unknown:
            raise ValueError(f"unknown statistics {sorted(unknown)}; known: {STATISTICS}")
        splats = SceneTensors.from_scene(scene, self.device, self.dtype)
        n = scene.count
        acc = self.accumulators(n, stats)
        out = np.zeros((n, len(cameras)), np.float32)
        for v, camera in enumerate(cameras):
            _, _, contrib = self.rasterize(splats, camera, want_image=False, extras=acc)
            out[:, v] = contrib.cpu().numpy()
            if progress is not None:
                progress(v + 1, len(cameras))
        result: dict[str, NDArray[Any]] = {"contribution": out}
        for name, tensor in acc.items():
            array = tensor.cpu().numpy()
            result[name] = array if name == "top_count" else array.astype(np.float32)
        return result

    def accumulators(self, count: int, stats: Collection[str]) -> dict[str, Tensor]:
        """Zeroed accumulators for :meth:`rasterize`'s ``extras``."""
        shapes = {"hit_transmittance": (count,), "top_count": (count,), "fisher": (count, 6, 6)}
        return {
            name: torch.zeros(
                shapes[name],
                device=self.device,
                dtype=torch.int64 if name == "top_count" else self.dtype,
            )
            for name in STATISTICS
            if name in stats
        }

    def estimate_seconds(self, scene: Scene, cameras: Sequence[Camera]) -> float | None:
        """Estimate wall time for :meth:`contributions` before running it.

        Bins a few evenly spaced views to count (tile, splat) pairs, and times
        a small synthetic workload once per process to learn this device's
        cost per pixel-pair. Good to a factor of two or so, which is what a
        "this will take about N minutes" notice needs.
        """
        if not cameras or scene.count == 0:
            return 0.0
        splats = SceneTensors.from_scene(scene, self.device, self.dtype)
        picks = np.linspace(0, len(cameras) - 1, min(4, len(cameras))).round().astype(int)
        sample = [cameras[i] for i in picks]
        start = time.perf_counter()
        pair_pixels = [self._bin(splats, cam).pairs * self.tile * self.tile for cam in sample]
        binning = (time.perf_counter() - start) / len(sample)
        per_view = binning + float(np.mean(pair_pixels)) * self._pixel_pair_cost()
        return per_view * len(cameras)

    # Rasterisation -------------------------------------------------------------------

    def _bin(self, splats: SceneTensors, camera: Camera) -> _Bins:
        proj = project(splats, camera)
        vis = torch.nonzero(proj.visible).squeeze(1)
        order = torch.sort(proj.depths[vis], stable=True).indices
        ids = vis[order]
        t = self.tile
        tiles_x = math.ceil(camera.width / t)
        tiles_y = math.ceil(camera.height / t)

        # Pixel j's centre j + 0.5 lies in [lo, hi] iff ceil(lo - 0.5) <= j <= floor(hi - 0.5).
        bb = proj.bbox[ids]
        jx0 = torch.ceil(bb[:, 0] - 0.5).clamp(0, camera.width - 1)
        jx1 = torch.floor(bb[:, 2] - 0.5).clamp(-1, camera.width - 1)
        jy0 = torch.ceil(bb[:, 1] - 0.5).clamp(0, camera.height - 1)
        jy1 = torch.floor(bb[:, 3] - 0.5).clamp(-1, camera.height - 1)
        tx0, tx1 = (jx0 // t).long(), (jx1 // t).long()
        ty0, ty1 = (jy0 // t).long(), (jy1 // t).long()
        nx = (tx1 - tx0 + 1).clamp_min(0)
        ny = (ty1 - ty0 + 1).clamp_min(0)
        empty = (jx1 < jx0) | (jy1 < jy0)
        counts = torch.where(empty, torch.zeros_like(nx), nx * ny)

        total = int(counts.sum())
        dev = splats.means.device
        pos = torch.repeat_interleave(torch.arange(ids.shape[0], device=dev), counts)
        first = torch.cumsum(counts, 0) - counts
        local = torch.arange(total, device=dev) - first[pos]
        tile_x = tx0[pos] + local % nx[pos]
        tile_y = ty0[pos] + local // nx[pos]
        tile_id = tile_y * tiles_x + tile_x
        # Stable sort by tile keeps the depth order the splats already have.
        by_tile = torch.sort(tile_id, stable=True).indices
        per_tile = torch.bincount(tile_id, minlength=tiles_x * tiles_y)
        tile_start = torch.zeros(tiles_x * tiles_y + 1, dtype=torch.long, device=dev)
        tile_start[1:] = torch.cumsum(per_tile, 0)
        return _Bins(ids, proj, pos[by_tile], tile_start, tiles_x, tiles_y)

    def rasterize(
        self,
        splats: SceneTensors,
        camera: Camera,
        *,
        want_image: bool,
        extras: dict[str, Tensor] | None = None,
    ) -> tuple[Tensor | None, Tensor, Tensor]:
        """Composite one view.

        Returns ``(image or None, alpha image, contribution per splat)``; the
        contribution is ``(N,)`` in the dtype of ``splats``. ``extras`` maps
        names from :data:`STATISTICS` to accumulators from
        :meth:`accumulators`, which this view's values are added into.
        """
        dev, dtype = splats.means.device, splats.means.dtype
        extras = extras or {}
        bins = self._bin(splats, camera)
        proj, ids = bins.proj, bins.ids
        means2d = proj.means2d[ids]
        conics = proj.conics[ids]
        opac = proj.opacities[ids]
        hit_t = extras.get("hit_transmittance")
        top = extras.get("top_count")
        fisher = extras.get("fisher")
        if fisher is not None:
            jac_mean2d, jac_conic = projection_jacobians(splats, camera, ids)
        colours: Tensor | None = None
        if want_image:
            campos = torch.as_tensor(camera.position, device=dev, dtype=dtype)
            colours = (eval_sh(splats.sh[ids], splats.means[ids] - campos) + 0.5).clamp_min(0.0)

        h, w, t = camera.height, camera.width, self.tile
        contrib_sorted = torch.zeros(ids.shape[0], device=dev, dtype=dtype)
        bg = torch.tensor(self.background, device=dev, dtype=dtype)
        image = bg.expand(h, w, 3).clone() if want_image else None
        alpha_img = torch.zeros(h, w, device=dev, dtype=dtype)
        starts = bins.tile_start.tolist()

        for tile in range(bins.tiles_x * bins.tiles_y):
            lo, hi = starts[tile], starts[tile + 1]
            if lo == hi:
                continue
            ty, tx = divmod(tile, bins.tiles_x)
            y0, x0 = ty * t, tx * t
            y1, x1 = min(y0 + t, h), min(x0 + t, w)
            ys = torch.arange(y0, y1, device=dev, dtype=dtype) + 0.5
            xs = torch.arange(x0, x1, device=dev, dtype=dtype) + 0.5
            gy, gx = torch.meshgrid(ys, xs, indexing="ij")
            pix = torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=-1)

            members = bins.tile_splats[lo:hi]
            trans = torch.ones(pix.shape[0], device=dev, dtype=dtype)
            live = torch.ones(pix.shape[0], device=dev, dtype=torch.bool)
            tile_rgb = torch.zeros(pix.shape[0], 3, device=dev, dtype=dtype)
            if top is not None:
                best_w = torch.zeros(pix.shape[0], device=dev, dtype=dtype)
                best_id = torch.zeros(pix.shape[0], device=dev, dtype=torch.long)
            for c0 in range(0, members.shape[0], CHUNK):
                chunk = members[c0 : c0 + CHUNK]
                alpha = gaussian_alpha(
                    conics[chunk], means2d[chunk], opac[chunk], pix, self.alpha_max
                )
                alpha = alpha * live[None, :]
                t_after = trans[None, :] * torch.cumprod(1.0 - alpha, dim=0)
                t_before = torch.cat([trans[None, :], t_after[:-1]], dim=0)
                # Exclusive stop: the splat that would take T to T_MIN or below,
                # and everything behind it, deposits nothing.
                keep = t_after > T_MIN
                weight = alpha * t_before * keep
                contrib_sorted.index_add_(0, chunk, weight.sum(dim=1))
                if colours is not None:
                    tile_rgb += weight.T @ colours[chunk]
                if hit_t is not None:
                    hit = (alpha > 0) & keep
                    hit_t.index_add_(0, ids[chunk], (t_before * hit).sum(dim=1))
                if top is not None:
                    # Strictly greater, so on a tie the nearer splat keeps the pixel.
                    vals, arg = weight.max(dim=0)
                    better = vals > best_w
                    best_w = torch.where(better, vals, best_w)
                    best_id = torch.where(better, ids[chunk][arg], best_id)
                if fisher is not None:
                    grad = _alpha_param_grads(
                        alpha, t_before * keep, conics[chunk], means2d[chunk], pix,
                        jac_mean2d[chunk], jac_conic[chunk], self.alpha_max,
                    )  # fmt: skip
                    fisher.index_add_(0, ids[chunk], torch.einsum("kpi,kpj->kij", grad, grad))
                stopped = ~keep.all(dim=0)
                # Transmittance after the last splat actually composited.
                last = torch.where(keep, t_after, torch.full_like(t_after, math.inf)).amin(dim=0)
                trans = torch.where(keep.any(dim=0), last, trans)
                live = live & ~stopped
                if not bool(live.any()):
                    break
            if top is not None:
                won = best_w > 0
                top.index_add_(0, best_id[won], torch.ones_like(best_id[won]))
            alpha_img[y0:y1, x0:x1] = (1.0 - trans).reshape(y1 - y0, x1 - x0)
            if image is not None:
                image[y0:y1, x0:x1] = (tile_rgb + trans[:, None] * bg).reshape(y1 - y0, x1 - x0, 3)

        contrib = torch.zeros(splats.count, device=dev, dtype=dtype)
        contrib[ids] = contrib_sorted
        return image, alpha_img, contrib

    # Calibration ------------------------------------------------------------------------

    def _pixel_pair_cost(self) -> float:
        """Seconds per (tile pixel × splat pair) on this device, measured once."""
        if self._seconds_per_pixel_pair is None:
            rng = np.random.default_rng(0)
            n = 3000
            f = np.float32
            scene = Scene(
                means=rng.uniform(-1, 1, (n, 3)).astype(f),
                scales=np.full((n, 3), math.log(0.08), f),
                quats=np.tile(np.array([1, 0, 0, 0], f), (n, 1)),
                opacities=np.full(n, -2.0, f),  # translucent, so tiles do not stop early
                sh0=np.zeros((n, 3), f),
                shN=np.zeros((n, 0, 3), f),
            )
            from psplat.io.cameras import look_at

            cam = Camera(
                "calibrate", 128, 128, 110.0, 110.0, 64.0, 64.0, look_at((0, -4, 0), (0, 0, 0))
            )
            splats = SceneTensors.from_scene(scene, self.device, self.dtype)
            self.rasterize(splats, cam, want_image=False)  # warm up
            start = time.perf_counter()
            self.rasterize(splats, cam, want_image=False)
            elapsed = time.perf_counter() - start
            pairs = self._bin(splats, cam).pairs
            self._seconds_per_pixel_pair = elapsed / max(1, pairs * self.tile * self.tile)
        return self._seconds_per_pixel_pair


# Fisher helpers -------------------------------------------------------------------------


def projection_jacobians(
    splats: SceneTensors, camera: Camera, ids: Tensor
) -> tuple[Tensor, Tensor]:
    """Derivatives of each splat's screen mean and conic, for the splats ``ids``.

    Returns ``(M, 2, 6)`` and ``(M, 3, 6)``: d(means2d)/dθ and d(conic)/dθ, with
    θ the world-space mean then the log-scale (:data:`FISHER_PARAMS`). Each
    splat's projection depends only on its own parameters, so one forward-mode
    pass per parameter, with the same unit tangent for every splat, gives every
    splat's partial derivatives at once.
    """
    sub = SceneTensors(
        means=splats.means[ids],
        scales=splats.scales[ids],
        quats=splats.quats[ids],
        opacities=splats.opacities[ids],
        sh=splats.sh[ids, :1],
    )
    log_scales = torch.log(sub.scales)

    def screen(means: Tensor, log_s: Tensor) -> tuple[Tensor, Tensor]:
        proj = project(replace(sub, means=means, scales=torch.exp(log_s)), camera)
        return proj.means2d, proj.conics

    d_mean2d, d_conic = [], []
    for k in range(6):
        t_mean = torch.zeros_like(sub.means)
        t_scale = torch.zeros_like(log_scales)
        (t_mean if k < 3 else t_scale)[:, k % 3] = 1.0
        with warnings.catch_warnings():
            # torch.func's first use scripts a helper internally, and PyTorch
            # warns that torch.jit.script is deprecated; nothing for users to act on.
            warnings.filterwarnings("ignore", "`torch.jit.script` is deprecated", FutureWarning)
            tangents = torch.func.jvp(screen, (sub.means, log_scales), (t_mean, t_scale))[1]
        dm, dc = tangents
        d_mean2d.append(dm)
        d_conic.append(dc)
    return torch.stack(d_mean2d, dim=-1), torch.stack(d_conic, dim=-1)


def _alpha_param_grads(
    alpha: Tensor,
    trans: Tensor,
    conics: Tensor,
    means2d: Tensor,
    pix: Tensor,
    jac_mean2d: Tensor,
    jac_conic: Tensor,
    alpha_max: float,
) -> Tensor:
    """``T · dα/dθ`` for each splat and pixel, ``(K, P, 6)``.

    The direct term only: transmittance in front of the splat is held fixed,
    and how a change to this splat would dim the splats behind it is left out
    (docs/baselines.md, ``pup-simple``). Where alpha was skipped, clamped at
    ``alpha_max`` or not composited, the derivative is zero, as it is in the
    rasteriser's backward pass.
    """
    d = means2d[:, None, :] - pix[None, :, :]
    dx, dy = d[..., 0], d[..., 1]
    a, b, c = conics[:, 0:1], conics[:, 1:2], conics[:, 2:3]
    # alpha = o · exp(-q), q = (a dx² + c dy²) / 2 + b dx dy, dx = mean_x - pixel_x.
    s = alpha * trans * (alpha < alpha_max)
    g_mean2d = -s[..., None] * torch.stack([a * dx + b * dy, b * dx + c * dy], dim=-1)
    g_conic = -s[..., None] * torch.stack([0.5 * dx * dx, dx * dy, 0.5 * dy * dy], dim=-1)
    return torch.einsum("kpi,kij->kpj", g_mean2d, jac_mean2d) + torch.einsum(
        "kpi,kij->kpj", g_conic, jac_conic
    )
