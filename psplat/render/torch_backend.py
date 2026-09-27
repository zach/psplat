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

Speed, measured in this repository's container (4 CPU threads): about 3–6 s
per 1237×822 view for 300k splats. The cost is the alpha evaluation itself
(tile pixels × splat pairs), not Python overhead, so a 3M-splat scene over
~160 views is hours on CPU rather than the tens of minutes the handoff
hoped for. It is the guaranteed path and the oracle; gsplat is the fast one.
"""

from __future__ import annotations

import math
import time
from collections.abc import Sequence
from dataclasses import dataclass

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


def default_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
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
        splats = SceneTensors.from_scene(scene, self.device, self.dtype)
        out = np.zeros((scene.count, len(cameras)), np.float32)
        for v, camera in enumerate(cameras):
            _, _, contrib = self.rasterize(splats, camera, want_image=False)
            out[:, v] = contrib.cpu().numpy()
            if progress is not None:
                progress(v + 1, len(cameras))
        return out

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
        self, splats: SceneTensors, camera: Camera, *, want_image: bool
    ) -> tuple[Tensor | None, Tensor, Tensor]:
        """Composite one view.

        Returns ``(image or None, alpha image, contribution per splat)``; the
        contribution is ``(N,)`` in the dtype of ``splats``.
        """
        dev, dtype = splats.means.device, splats.means.dtype
        bins = self._bin(splats, camera)
        proj, ids = bins.proj, bins.ids
        means2d = proj.means2d[ids]
        conics = proj.conics[ids]
        opac = proj.opacities[ids]
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
                stopped = ~keep.all(dim=0)
                # Transmittance after the last splat actually composited.
                last = torch.where(keep, t_after, torch.full_like(t_after, math.inf)).amin(dim=0)
                trans = torch.where(keep.any(dim=0), last, trans)
                live = live & ~stopped
                if not bool(live.any()):
                    break
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
