"""Image quality metrics: PSNR, SSIM, LPIPS, ꟻLIP and ColorVideoVDP (P1.6, #25).

Every metric takes a test and a reference image as ``(H, W, 3)`` float
arrays and returns one number. Images are display-encoded RGB in [0, 1], as
3DGS renders are: trainers fit rendered colour straight to sRGB photographs,
so a render's values are already what a screen shows. Values outside
[0, 1] are clipped first, as a display would.

| Metric | Better | Implementation |
|---|---|---|
| ``psnr`` | higher | here; ``inf`` for identical images |
| ``ssim`` | higher | here, Wang et al. 2004: 11×11 Gaussian window, σ = 1.5, mean over channels |
| ``lpips`` | lower | the ``lpips`` package (BSD-2), AlexNet by default |
| ``flip`` | lower | NVIDIA ``flip-evaluator`` (BSD-3), LDR-ꟻLIP, mean of the error map |
| ``cvvdp`` | higher | ColorVideoVDP (``pycvvdp``, MIT), in JOD; optional: ``psplat[cvvdp]`` |

``docs/metrics.md`` records the parameters and why they were chosen.
"""

from __future__ import annotations

import contextlib
import io
import math
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
import torch
import torch.nn.functional as F
from numpy.typing import NDArray

Image = NDArray[np.floating]

# ꟻLIP's default viewing condition: 0.7 m from a 0.7 m wide, 3840-pixel
# display. Its pixels per degree are computed exactly as flip-evaluator does
# (67.0206...); rounding to 67 moves the mean error in the fifth decimal.
FLIP_PPD = 0.7 * (3840 / 0.7) * math.pi / 180
SSIM_WINDOW = 11
SSIM_SIGMA = 1.5
SSIM_K1, SSIM_K2 = 0.01, 0.03
LPIPS_NETS = ("alex", "vgg")
CVVDP_DISPLAY = "standard_4k"


class MetricUnavailable(RuntimeError):
    """A metric cannot run here; the message says what to install or fetch."""


def prepare(image: Image) -> NDArray[np.float32]:
    a = np.asarray(image, dtype=np.float32)
    if a.ndim != 3 or a.shape[2] != 3:
        raise ValueError(f"expected an (H, W, 3) image, got shape {a.shape}")
    return np.clip(np.nan_to_num(a, nan=0.0), 0.0, 1.0)


def _pair(test: Image, ref: Image) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    t, r = prepare(test), prepare(ref)
    if t.shape != r.shape:
        raise ValueError(f"image sizes differ: {t.shape} vs {r.shape}")
    return t, r


def psnr(test: Image, ref: Image) -> float:
    """Peak signal-to-noise ratio in dB, peak 1."""
    t, r = _pair(test, ref)
    mse = float(np.mean((t.astype(np.float64) - r) ** 2))
    return math.inf if mse == 0 else 10.0 * math.log10(1.0 / mse)


def _gaussian_window(size: int, sigma: float) -> torch.Tensor:
    x = torch.arange(size, dtype=torch.float64) - (size - 1) / 2
    g = torch.exp(-(x**2) / (2 * sigma**2))
    g = g / g.sum()
    return torch.outer(g, g)


def ssim(test: Image, ref: Image) -> float:
    """Structural similarity (Wang et al. 2004) over the valid region, data range 1.

    Matches scikit-image's ``structural_similarity`` with
    ``gaussian_weights=True, sigma=1.5, use_sample_covariance=False,
    data_range=1``, averaged over channels.
    """
    t, r = _pair(test, ref)
    if min(t.shape[:2]) < SSIM_WINDOW:
        raise ValueError(f"SSIM needs images at least {SSIM_WINDOW} pixels on each side")
    x = torch.from_numpy(t.astype(np.float64)).permute(2, 0, 1)[:, None]
    y = torch.from_numpy(r.astype(np.float64)).permute(2, 0, 1)[:, None]
    w = _gaussian_window(SSIM_WINDOW, SSIM_SIGMA)[None, None]

    def blur(z: torch.Tensor) -> torch.Tensor:
        return F.conv2d(z, w)

    mx, my = blur(x), blur(y)
    vx = blur(x * x) - mx * mx
    vy = blur(y * y) - my * my
    cxy = blur(x * y) - mx * my
    c1, c2 = SSIM_K1**2, SSIM_K2**2
    s = ((2 * mx * my + c1) * (2 * cxy + c2)) / ((mx * mx + my * my + c1) * (vx + vy + c2))
    return float(s.mean())


def _flip(test: Image, ref: Image, ppd: float) -> tuple[NDArray[np.float32], float]:
    try:
        import flip_evaluator
    except ImportError as exc:  # pragma: no cover - a core dependency
        raise MetricUnavailable("ꟻLIP needs flip-evaluator: pip install flip-evaluator") from exc
    t, r = _pair(test, ref)
    error, mean, _ = flip_evaluator.evaluate(
        r, t, "LDR", inputsRGB=True, applyMagma=False, computeMeanError=True,
        parameters={"ppd": float(ppd)},
    )  # fmt: skip
    out = np.asarray(error, dtype=np.float32)
    return (out[..., 0] if out.ndim == 3 else out), float(mean)


def flip(test: Image, ref: Image, ppd: float = FLIP_PPD) -> float:
    """Mean LDR-ꟻLIP error, 0 (identical) to 1, as flip-evaluator computes it."""
    return _flip(test, ref, ppd)[1]


def flip_map(test: Image, ref: Image, ppd: float = FLIP_PPD) -> NDArray[np.float32]:
    """Per-pixel LDR-ꟻLIP error, ``(H, W)``, for heatmaps."""
    return _flip(test, ref, ppd)[0]


class _Model(Protocol):
    def __call__(self, test: Image, ref: Image) -> float: ...


class Lpips:
    """LPIPS (Zhang et al. 2018) with the ``lpips`` package's v0.1 weights.

    The backbone's ImageNet weights come from download.pytorch.org through
    torchvision on first use (AlexNet 233 MB, VGG-16 528 MB) and are cached
    by PyTorch.
    """

    def __init__(self, net: str = "alex", device: torch.device | str = "cpu") -> None:
        if net not in LPIPS_NETS:
            raise ValueError(f"LPIPS network must be one of {LPIPS_NETS}")
        self.net = net
        self.device = torch.device(device)
        try:
            import lpips
        except ImportError as exc:  # pragma: no cover - a core dependency
            raise MetricUnavailable("LPIPS needs the lpips package: pip install lpips") from exc
        with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
            # torchvision deprecates lpips's `pretrained=` argument; lpips prints a banner.
            warnings.simplefilter("ignore", UserWarning)
            try:
                self.model = lpips.LPIPS(net=net, verbose=False).to(self.device).eval()
            except (OSError, RuntimeError) as exc:
                raise MetricUnavailable(
                    f"LPIPS could not load the {net} backbone weights, which PyTorch downloads "
                    f"from download.pytorch.org once ({exc}). Connect to the internet once, "
                    "or leave lpips out of --metrics."
                ) from exc

    def __call__(self, test: Image, ref: Image) -> float:
        t, r = _pair(test, ref)

        def tensor(a: NDArray[np.float32]) -> torch.Tensor:
            return torch.from_numpy(a).permute(2, 0, 1)[None].to(self.device) * 2 - 1

        with torch.no_grad():
            return float(self.model(tensor(t), tensor(r)).item())


class Cvvdp:
    """ColorVideoVDP (Mantiuk et al. 2024) on a still image, in JOD (10 = identical)."""

    def __init__(self, display: str = CVVDP_DISPLAY, device: torch.device | str = "cpu") -> None:
        try:
            import pycvvdp
        except ImportError as exc:
            raise MetricUnavailable(
                "ColorVideoVDP is optional: pip install 'psplat[cvvdp]' (it brings ffmpeg-python, "
                "matplotlib and other dependencies)"
            ) from exc
        self.display = display
        self.model = pycvvdp.cvvdp(display_name=display, quiet=True, device=torch.device(device))

    def __call__(self, test: Image, ref: Image) -> float:
        t, r = _pair(test, ref)
        with torch.no_grad():
            q, _ = self.model.predict(t, r, dim_order="HWC")
        return float(q)


# Registry ------------------------------------------------------------------------------------

DEFAULT_METRICS = ("psnr", "ssim", "lpips", "flip")
KNOWN_METRICS = ("psnr", "ssim", "lpips", "flip", "cvvdp")
HIGHER_IS_BETTER = {"psnr": True, "ssim": True, "lpips": False, "flip": False, "cvvdp": True}


@dataclass
class MetricSet:
    """The requested metrics, with any models loaded once."""

    names: tuple[str, ...]
    settings: dict[str, Any]
    _models: dict[str, _Model]

    def __call__(self, test: Image, ref: Image) -> dict[str, float]:
        return {name: self._models[name](test, ref) for name in self.names}


def check_metrics(names: Sequence[str]) -> tuple[str, ...]:
    out: list[str] = []
    for raw in names:
        name = raw.strip().lower()
        if not name:
            continue
        if name not in KNOWN_METRICS:
            raise ValueError(f"unknown metric {name!r}; known: {', '.join(KNOWN_METRICS)}")
        if name not in out:
            out.append(name)
    if not out:
        raise ValueError("at least one metric is required")
    return tuple(out)


def load_metrics(
    names: Sequence[str],
    *,
    device: torch.device | str = "cpu",
    lpips_net: str = "alex",
    flip_ppd: float = FLIP_PPD,
    cvvdp_display: str = CVVDP_DISPLAY,
) -> MetricSet:
    """Validate names and load models; raises :class:`MetricUnavailable` early."""
    chosen = check_metrics(names)
    models: dict[str, _Model] = {}
    settings: dict[str, Any] = {}
    for name in chosen:
        if name == "psnr":
            models[name] = psnr
        elif name == "ssim":
            models[name] = ssim
            settings["ssim"] = {"window": SSIM_WINDOW, "sigma": SSIM_SIGMA}
        elif name == "flip":
            models[name] = lambda t, r: flip(t, r, flip_ppd)
            settings["flip_ppd"] = flip_ppd
        elif name == "lpips":
            models[name] = Lpips(lpips_net, device)
            settings["lpips_net"] = lpips_net
        elif name == "cvvdp":
            models[name] = Cvvdp(cvvdp_display, device)
            settings["cvvdp_display"] = cvvdp_display
    return MetricSet(chosen, settings, models)
