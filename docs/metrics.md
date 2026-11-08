<!-- psplat-docs: audience=dev -->
# Quality metrics

`psplat eval` (P1.6, [#25](https://github.com/zach/psplat/issues/25))
renders the first k splats of an ordered scene for every budget and scores
the result. This page says what it compares against, how each metric is
computed, and how each implementation was checked. Which metric to headline
is decided later, by E5
([#36](https://github.com/zach/psplat/issues/36)), against human ratings.

## What is compared

Each budget's render is compared with a reference image of the same view.

- **The whole scene (default).** The reference is the render of every splat.
  This measures only what truncation loses, which is what an ordering is
  judged on, and needs no dataset images. The 100% budget is perfect by
  construction: PSNR is infinite, SSIM 1, LPIPS and ꟻLIP 0.
- **Photographs (`--images DIR`).** The reference is the training
  photograph, found by camera name in `DIR`, at the cameras' resolution.
  This adds the reconstruction's own error, so the 100% budget shows how
  good the trained scene is. Benchmarks that report against ground truth
  use this.

Images are display-encoded RGB in [0, 1]. 3DGS trainers fit rendered colour
directly to sRGB photographs, so a render's values are already what a
display shows; values outside [0, 1] are clipped, as a display would.

Views default to the camera set's `eval` split. The fixture's cameras have
one, and `psplat cameras` writes one, every 8th view. Budgets default to the
manifest's cut points.

## The metrics

| Metric | Better | Implementation | Settings |
|---|---|---|---|
| PSNR | higher | psplat | peak 1; `inf` for identical images |
| SSIM | higher | psplat | Wang et al. 2004: 11×11 Gaussian window, σ = 1.5, K1 = 0.01, K2 = 0.03, valid region, mean of R, G and B |
| LPIPS | lower | `lpips` 0.1.4 (BSD-2) | AlexNet backbone, v0.1 linear weights; `--lpips-net vgg` for VGG-16 |
| ꟻLIP | lower | NVIDIA `flip-evaluator` 1.7 (BSD-3) | LDR-ꟻLIP, sRGB inputs, 67.02 pixels per degree (0.7 m from a 0.7 m wide 4K display); the evaluator's own mean |
| ColorVideoVDP | higher | `cvvdp` 0.5.7 (MIT), optional | `standard_4k` display, still image, JOD scale (10 = identical) |

### Why these settings

- **LPIPS AlexNet.** It is the `lpips` package's default, and the one its
  authors recommend as a forward metric. The original 3DGS paper used VGG, so tables meant to sit next to the 3DGS
  paper's should use `--lpips-net vgg`. Mixing the two in one comparison is
  wrong: their scales differ.
- **ꟻLIP at its default viewing condition.** That is the condition NVIDIA's
  own results use. Rounding 67.02 pixels per degree to 67 changes the
  mean in the fifth decimal, which is enough to break agreement with their
  reference values.
- **ColorVideoVDP is an extra.** Its licence is MIT, but it brings
  ffmpeg-python, matplotlib, PyEXR and huggingface_hub. psplat uses only the
  classic metric, whose data ships inside the package, so nothing is
  downloaded at run time. Install it with `pip install "psplat[cvvdp]"`.
- **D4 resolved:** `flip-evaluator` has wheels for Windows, macOS on Apple
  Silicon and Linux, for Python 3.12 and 3.13, so psplat depends on it rather
  than porting ꟻLIP to PyTorch.

## How each was checked

| Metric | Check |
|---|---|
| PSNR | closed form: a uniform 0.1 offset is exactly 20 dB |
| SSIM | equals scikit-image's `structural_similarity` with the same settings to 1e-6 on noisy pairs; equals the closed form `(2ab + C1) / (a² + b² + C1)` for flat images |
| ꟻLIP | NVIDIA's published pair (`images/reference.png`, `test.png`): psplat 0.159691, flip-evaluator from the files 0.159691, NVIDIA's test expects 0.159693 (generated on Windows; their README notes small differences between operating systems) |
| LPIPS | equals the `lpips` package's own image-to-tensor path; 0 for identical images; grows with noise |
| ColorVideoVDP | 10 JOD for identical images; falls as noise grows |

LPIPS's backbone weights (AlexNet 233 MB, VGG-16 528 MB) are downloaded once
by PyTorch from download.pytorch.org and cached. The LPIPS tests run when
the weights are cached or `PSPLAT_LPIPS=1`. CI sets it on Linux, and the
Windows TD job runs `eval` with LPIPS from the installed wheel.

## Output

`results.parquet` has one row per budget and view. It uses the columns of
handoff §4.2 step 5 (`scene, trainer, ordering, terms, budget_frac, bytes,
format, psnr, ssim, lpips, flip, cvvdp, fps_web, wall_time, git_sha`) plus:

- `view` and `budget_splats`;
- `reference` (`full` or `images`);
- `backend`, `source_sha256` and `tool`.

Metrics not requested are null, and so is `fps_web` (the viewer measures it,
in P3.5). `bytes` is the PLY prefix's exact size. The packed formats' sizes
are in the manifest.

With `--write-manifest`, each cut's mean over views is also stored in the
manifest, as in handoff §3.4 (`"flip": 0.11, "lpips": 0.21`). Values that
are not finite, such as PSNR at a perfect budget, are left out, because
JSON has no infinity.
