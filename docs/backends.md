<!-- psplat-docs: audience=dev -->
# Rendering backends

`psplat analyze` needs a renderer to measure how much each splat contributes
to each view. Every renderer implements one protocol, so the rest of psplat
never knows which one ran (handoff §0.2).

| Backend | Runs on | Status | Role |
|---|---|---|---|
| `torch` | anything with PyTorch: CUDA, Apple MPS, CPU | implemented | always works; the correctness oracle |
| `gsplat` | NVIDIA GPU, Linux and Windows | planned, [#23](https://github.com/zach/psplat/issues/23) | production speed and benchmarks |
| `msplat` | Apple Silicon | planned, [#24](https://github.com/zach/psplat/issues/24) | Zach's local iteration |
| `null` | anything | implemented | commands on pre-analysed files; tests |

`--backend auto` picks gsplat when it is importable and CUDA is present,
else msplat on Apple Silicon, else torch. It prints one line naming the
choice and the reason. A backend that is installed but not implemented yet
is skipped, and the line says so.

## The protocol

Defined in `psplat/render/base.py`. Arrays cross it as NumPy.

```python
class Renderer(Protocol):
    name: str
    def describe(self) -> str: ...
    def render(self, scene, camera) -> RenderResult: ...        # (H, W, 3) image, (H, W) alpha
    def contributions(self, scene, cameras, progress=None): ... # (N, V) float32
    def estimate_seconds(self, scene, cameras) -> float | None: ...
```

A splat's contribution to a view, `c_iv`, is the sum over pixels of the
`alpha · T` it deposits, where `T` is the transmittance in front of it. This
is the rendering-contribution term of the perceptual weight (handoff §2.3).

Footprints and visibility do not come from backends. They come from the
shared projection in `psplat/render/project.py`, so they mean the same thing
whichever backend computed the contributions.

## Rasteriser conventions

The torch backend transcribes gsplat's classic 3DGS rasteriser. The
constants were read from the gsplat 1.5.3 source release rather than
recalled:

| Rule | gsplat (psplat default) | Inria reference |
|---|---|---|
| alpha cap | 0.999 | 0.99 |
| skip a splat at a pixel when | alpha < 1/255, or the quadratic form is negative | alpha < 1/255 |
| stop a pixel at the first splat that would make T | ≤ 1e-4, exclusive | < 1e-4, exclusive |
| 2D low-pass added to the covariance | 0.3 px² | 0.3 px² |
| near plane | 0.01 | 0.2 |
| pixel sampled at | its centre, +0.5 | its centre, via the NDC mapping |

The Inria alpha cap is one argument away: `TorchRenderer(alpha_max=0.99)`.

## Cross-backend tolerance

Once the gsplat backend exists, CI will assert on the synthetic fixture that
its contributions match the torch oracle. The starting tolerance, to be
tightened once measured, is a Spearman rank correlation of at least 0.99
and a maximum relative error of at most 5% on splats with non-trivial
contribution.

Expected sources of difference:

- float32 accumulation order: gsplat multiplies transmittance pixel by
  pixel in CUDA, the oracle uses a cumulative product per chunk;
- gsplat rounds screen radii up to whole pixels and caps the opacity-aware
  extent at 3.33σ; the oracle's extent is exact, and never smaller.

## The msplat gap

msplat 1.1.4 exposes `GaussianTrainer.render_from_pose(cam2world)`, which
returns an image, and no per-splat contribution or importance API. The
msplat backend therefore needs an approximation: projected footprint ×
opacity × a visibility estimate for iteration, validated against
leave-group-out re-renders. Its agreement with the oracle will be recorded
here when it lands. If it is too loose, Mac iteration uses the torch backend
on MPS instead.

## Speed of the torch backend

Measured in the remote development container on 4 CPU threads, with
300,000 splats at 1237×822:

| Scene type | Seconds per view |
|---|---|
| opaque surface | 3.2 |
| translucent clutter | 5.8 |

The cost is the alpha evaluation itself, not Python overhead. A 3M-splat
scene over about 160 views is therefore hours on a CPU. The handoff hoped
for tens of minutes; the gsplat backend is the fast path.

To try the torch backend on a GPU or on MPS, pass the device:

```bash
psplat analyze scene.ply --cameras cams.json --backend torch --device cuda
```

```powershell
psplat analyze scene.ply --cameras cams.json --backend torch --device cuda
```
