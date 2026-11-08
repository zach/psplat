<!-- psplat-docs: audience=td -->
# psplat

Perceptual, budget-scalable Gaussian splats. psplat analyses a trained 3D
Gaussian Splat scene once, then orders its splats so that the first N are
the best-looking N. Any engine can then truncate the file at whatever
budget it has, the way an MP3 decoder plays whatever bitrate it gets.

**Status: pre-alpha.** `psplat analyze`, `order`, `pack`, `eval` and the
web viewer work on any machine today. The Unreal Engine sample is next;
[`docs/plan.md`](docs/plan.md) has the roadmap and
[issues](https://github.com/zach/psplat/issues) track each piece.

## What you need

- Python 3.12 or 3.13.
- A trained splat scene as a `.ply` file, from any trainer.
- Its cameras, as a COLMAP model folder, a nerfstudio `transforms.json`, or
  psplat's own `cams.json`.
- For SOG, SPZ or glTF output only: [Node.js](https://nodejs.org) LTS. PLY
  output needs nothing extra.

You do not need a GPU, a compiler, Docker, WSL, a Mac, or any cloud account.
On Windows, see [`docs/windows.md`](docs/windows.md) for installing Python.

## Install

psplat is not on PyPI yet. Until v0.1 is published, replace `psplat` in the
`pip install` line with the path to a wheel from a GitHub release.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install psplat
psplat --version
```

```powershell
python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.venv\Scripts\Activate.ps1
pip install psplat
psplat --version
```

The `Set-ExecutionPolicy` line lets this one PowerShell window run the
virtual environment's activation script. It needs no administrator rights
and changes nothing once the window is closed.

## Analyse a scene

Run this in the folder that holds `scene.ply` and `cams.json`:

```bash
psplat analyze scene.ply --cameras cams.json
```

```powershell
psplat analyze scene.ply --cameras cams.json
```

psplat prints one line saying which renderer it picked and roughly how long
the analysis will take, then writes `scene.weights.npz` next to the scene.
With no NVIDIA GPU it uses its portable PyTorch renderer, which runs
anywhere but is slow on big scenes: expect hours for a multi-million splat
scene on a laptop CPU. A faster CUDA renderer is coming
([#23](https://github.com/zach/psplat/issues/23)).

If your cameras are a COLMAP model, pass the folder, or convert it once to
`cams.json`:

<!-- doctest: skip -->
```bash
psplat cameras path/to/garden/sparse/0 --downscale 4 --out cams.json
```

```powershell
psplat cameras path\to\garden\sparse\0 --downscale 4 --out cams.json
```

`--downscale 4` matches images that were resized to a quarter, as in the
Mip-NeRF 360 `images_4` folders. psplat refuses cameras with lens distortion
unless you undistort them first, because the result would silently be wrong.

## Order the splats

```bash
psplat order scene.ply scene.weights.npz
```

```powershell
psplat order scene.ply scene.weights.npz
```

This writes two files next to the scene:

- `scene.ordered.ply`, the same splats with the most important first. Any
  tool that reads splat PLY files can open it, and keeping only its first N
  splats gives the best N.
- `scene.psplat.json`, the manifest. Its `cuts` list says how many leading
  splats make up each budget: 5, 10, 25, 50, 75 and 100% by default, or
  your own with `--cuts 10,20,40`. An engine needs nothing else from it.

Today splats are ranked by how much each one contributes to the analysed
views. The perceptual ordering that psplat is named for comes in a later
release ([#35](https://github.com/zach/psplat/issues/35)). `--method` also
offers the pruning methods it will be compared against; see
[`docs/baselines.md`](docs/baselines.md).

## Pack it for an engine or the web

```bash
psplat pack scene.ordered.ply --ply --sog
```

```powershell
psplat pack scene.ordered.ply --ply --sog
```

This writes a `dist` folder holding:

- the ordered PLY, which Unreal and most tools import directly;
- a SOG file, PlayCanvas's compressed format for the web viewer;
- an updated `scene.psplat.json` that records how many bytes each budget
  takes in each format.

`--spz` and `--gltf` add those formats too.

SOG, SPZ and glTF are written by PlayCanvas's splat-transform. That tool
runs on Node.js, and psplat downloads it once, the first time you need it.
Without Node.js, psplat still writes the PLY, says what it skipped and why,
and exits with code 3.

Inside a SOG file, splats are only grouped by budget: each budget is
exactly the best splats, but order within a budget is not kept, so SOG
budgets snap to the cut points. PLY, SPZ and glTF keep the exact order, so
any splat count is a valid budget. SOG's view-dependent colour needs a GPU
to compress on scenes over about 33,000 splats. Until psplat can use one,
it writes those SOG files without it and tells you so;
[`docs/formats.md`](docs/formats.md) explains why.

## Look at it in a browser

<!-- doctest: skip -->
```bash
psplat viewer dist
```

```powershell
psplat viewer dist
```

This opens the web viewer in your default browser. Chrome, Edge and Firefox
all work; nothing is installed, and the viewer only listens on this machine.
Drag the **Budget** slider to see the scene with fewer splats, and tick
**Compare with full scene** for a split screen: the budget on the left, the
whole scene on the right. The panel shows how many splats are drawn, how
many bytes that budget takes and the frame time. **View** jumps to the
cameras in `cams.json`; drag to orbit, right-drag to pan, and scroll to zoom.
Press Ctrl+C in the terminal to stop the viewer.

A SOG keeps psplat's order only at the cut points, so its slider snaps to
them; a PLY takes any splat count.

## See what each budget costs in quality

```bash
psplat eval scene.ordered.ply --cameras cams.json
```

```powershell
psplat eval scene.ordered.ply --cameras cams.json
```

This renders each budget from the cameras set aside for evaluation and
compares it with the whole scene. It prints a table of PSNR, SSIM, LPIPS
and ꟻLIP per budget and writes every number to `scene.results.parquet`. The
first run downloads LPIPS's network weights (233 MB) once.
[`docs/metrics.md`](docs/metrics.md) explains each metric.

## Commands

`psplat --help` lists every command. Those not built yet say so and link to
the issue that tracks them.

| Command | Does | Status |
|---|---|---|
| `analyze` | per-splat contribution to every view | works |
| `cameras` | converts COLMAP or nerfstudio cameras to `cams.json` | works |
| `order` | sorts splats best first and writes the cut-point manifest | works |
| `pack` | writes PLY, SOG, SPZ or glTF, with per-budget sizes | works |
| `viewer` | web viewer with a budget slider and split-screen comparison | works |
| `eval` | renders each budget and scores it against the whole scene or photographs | works |
| `lod`, `bench`, `report`, `study` | level of detail and experiments | later |

## Licence

Apache-2.0. Third-party licences are listed in
[`THIRD_PARTY.md`](THIRD_PARTY.md).
