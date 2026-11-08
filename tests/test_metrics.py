"""P1.6: image metrics against known pairs and reference implementations.

- PSNR and SSIM: closed forms, and scikit-image's SSIM.
- ꟻLIP: identities and properties here. Against NVIDIA's published pair
  (reference.png/test.png, expected mean 0.159693 in their tests), psplat's
  wrapper gave 0.159691 on Linux, which is flip-evaluator's own file-based
  result; those 4–5 MB images are not vendored (lab notebook 2026-11-08).
- LPIPS needs 233 MB of backbone weights; it runs when they are cached or
  ``PSPLAT_LPIPS=1`` (the Linux CI job).
- ColorVideoVDP runs when ``psplat[cvvdp]`` is installed.
"""

from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from psplat.eval import metrics as m


def noisy(seed: int = 0, shape: tuple[int, int] = (40, 52), sigma: float = 0.1) -> tuple[
    np.ndarray, np.ndarray
]:  # fmt: skip
    rng = np.random.default_rng(seed)
    ref = rng.random((*shape, 3))
    return np.clip(ref + rng.normal(0, sigma, ref.shape), 0, 1), ref


# PSNR ----------------------------------------------------------------------------------------


def test_psnr_closed_form_and_identity() -> None:
    ref = np.full((8, 8, 3), 0.5)
    assert m.psnr(ref + 0.1, ref) == pytest.approx(20.0, abs=1e-5)  # 10·log10(1 / 0.01)
    assert m.psnr(ref, ref) == math.inf
    # Out-of-range values are clipped as a display would: 1.5 shows as 1.0.
    assert m.psnr(np.full((8, 8, 3), 1.5), np.ones((8, 8, 3))) == math.inf


def test_shapes_are_checked() -> None:
    with pytest.raises(ValueError, match="sizes differ"):
        m.psnr(np.zeros((4, 4, 3)), np.zeros((4, 5, 3)))
    with pytest.raises(ValueError, match=r"\(H, W, 3\)"):
        m.psnr(np.zeros((4, 4)), np.zeros((4, 4)))


# SSIM ----------------------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_ssim_matches_scikit_image(seed: int) -> None:
    skimage_metrics = pytest.importorskip("skimage.metrics")
    test, ref = noisy(seed, sigma=0.05 * (seed + 1))
    want = skimage_metrics.structural_similarity(
        test.astype(np.float32), ref.astype(np.float32), channel_axis=2, gaussian_weights=True,
        sigma=1.5, use_sample_covariance=False, data_range=1.0,
    )  # fmt: skip
    assert m.ssim(test, ref) == pytest.approx(want, abs=1e-6)


def test_ssim_closed_form_for_flat_images() -> None:
    # No variance, so SSIM is the luminance term (2ab + C1) / (a² + b² + C1).
    a, b = 0.3, 0.6
    c1 = 0.01**2
    want = (2 * a * b + c1) / (a * a + b * b + c1)
    got = m.ssim(np.full((16, 16, 3), a), np.full((16, 16, 3), b))
    assert got == pytest.approx(want, rel=1e-9)
    assert m.ssim(np.full((16, 16, 3), a), np.full((16, 16, 3), a)) == pytest.approx(1.0)
    with pytest.raises(ValueError, match="at least 11"):
        m.ssim(np.zeros((10, 30, 3)), np.zeros((10, 30, 3)))


# ꟻLIP ----------------------------------------------------------------------------------------


def test_flip_identity_extremes_and_map() -> None:
    test, ref = noisy()
    assert m.flip(ref, ref) == 0.0
    black, white = np.zeros((32, 32, 3)), np.ones((32, 32, 3))
    assert 0.9 < m.flip(white, black) <= 1.0
    error = m.flip_map(test, ref)
    assert error.shape == ref.shape[:2] and error.dtype == np.float32
    assert float(error.mean(dtype=np.float64)) == pytest.approx(m.flip(test, ref), abs=1e-5)


def test_flip_grows_with_error_and_depends_on_viewing_distance() -> None:
    values = [m.flip(*noisy(sigma=s)) for s in (0.02, 0.05, 0.1, 0.2)]
    assert values == sorted(values) and values[0] > 0
    test, ref = noisy(sigma=0.1)
    # Seen from further away (more pixels per degree), fine noise is less visible.
    assert m.flip(test, ref, ppd=200) < m.flip(test, ref, ppd=30)


def test_flip_default_viewing_condition_is_flips_own() -> None:
    assert pytest.approx(67.0206, abs=1e-4) == m.FLIP_PPD
    import flip_evaluator

    test, ref = noisy()
    _, mean, params = flip_evaluator.evaluate(ref.astype(np.float32), test.astype(np.float32),
                                              "LDR", applyMagma=False)  # fmt: skip
    assert params["ppd"] == pytest.approx(m.FLIP_PPD, rel=1e-6)
    assert m.flip(test, ref) == pytest.approx(mean, abs=1e-7)


# LPIPS ---------------------------------------------------------------------------------------


def _lpips_weights_cached() -> bool:
    hub = Path(torch.hub.get_dir()) / "checkpoints"
    return any(hub.glob("alexnet-*.pth"))


needs_lpips = pytest.mark.skipif(
    os.environ.get("PSPLAT_LPIPS") != "1" and not _lpips_weights_cached(),
    reason="LPIPS weights not cached; set PSPLAT_LPIPS=1 to download them (233 MB)",
)


@needs_lpips
def test_lpips_matches_the_packages_own_path_and_orders_errors() -> None:
    import lpips

    metric = m.Lpips("alex")
    test, ref = noisy(shape=(64, 64))
    # The package's own route from [0, 1] HWC images to its [-1, 1] NCHW input.
    direct = metric.model(lpips.im2tensor(test * 255.0), lpips.im2tensor(ref * 255.0))
    assert metric(test, ref) == pytest.approx(float(direct.item()), rel=1e-5)
    assert metric(ref, ref) == pytest.approx(0.0, abs=1e-7)
    values = [metric(*noisy(shape=(64, 64), sigma=s)) for s in (0.02, 0.1, 0.3)]
    assert values == sorted(values)


def test_lpips_net_is_checked() -> None:
    with pytest.raises(ValueError, match="LPIPS network"):
        m.Lpips("squeeze")


# ColorVideoVDP ---------------------------------------------------------------------------------


def test_cvvdp_reports_how_to_install_when_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    import builtins

    real_import = builtins.__import__

    def no_pycvvdp(name: str, *args: object, **kwargs: object) -> object:
        if name == "pycvvdp":
            raise ImportError(name)
        return real_import(name, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(builtins, "__import__", no_pycvvdp)
    with pytest.raises(m.MetricUnavailable, match=r"psplat\[cvvdp\]"):
        m.Cvvdp()


def test_cvvdp_identity_and_order() -> None:
    pytest.importorskip("pycvvdp")
    metric = m.Cvvdp()
    test, ref = noisy(shape=(64, 64), sigma=0.05)
    assert metric(ref, ref) == pytest.approx(10.0, abs=1e-4)
    worse = metric(*noisy(shape=(64, 64), sigma=0.2))
    assert 0 < worse < metric(test, ref) < 10.0


# Registry ------------------------------------------------------------------------------------


def test_metric_names_are_checked_and_deduplicated() -> None:
    assert m.check_metrics(["PSNR", " ssim", "psnr", ""]) == ("psnr", "ssim")
    with pytest.raises(ValueError, match="unknown metric"):
        m.check_metrics(["sharpness"])
    with pytest.raises(ValueError, match="at least one"):
        m.check_metrics([""])
    metrics = m.load_metrics(["psnr", "flip"], flip_ppd=40.0)
    test, ref = noisy()
    values = metrics(test, ref)
    assert list(values) == ["psnr", "flip"]
    assert values["flip"] == pytest.approx(m.flip(test, ref, 40.0))
    assert metrics.settings == {"flip_ppd": 40.0}
