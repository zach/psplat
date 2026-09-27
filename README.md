<!-- psplat-docs: audience=td -->
# psplat

Perceptual, budget-scalable Gaussian splats. psplat analyses a trained 3D
Gaussian Splat scene once, then orders its splats so that the first N are
the best-looking N. Any engine can then truncate the file at whatever
budget it has, the way an MP3 decoder plays whatever bitrate it gets.

**Status: pre-alpha.** `psplat analyze` works on any machine today.
Ordering, packing, the web viewer and the Unreal Engine sample are next;
[`docs/plan.md`](docs/plan.md) has the roadmap and
[issues](https://github.com/zach/psplat/issues) track each piece.

## What you need

- Python 3.12 or 3.13.
- A trained splat scene as a `.ply` file, from any trainer.
- Its cameras, as a COLMAP model folder, a nerfstudio `transforms.json`, or
  psplat's own `cams.json`.

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

## Commands

`psplat --help` lists every command. Those not built yet say so and link to
the issue that tracks them.

| Command | Does | Status |
|---|---|---|
| `analyze` | per-splat contribution to every view | works |
| `cameras` | converts COLMAP or nerfstudio cameras to `cams.json` | works |
| `order` | sorts splats and writes cut points | next |
| `pack` | writes PLY, SOG, SPZ or glTF | next |
| `viewer` | web viewer with a budget slider | next |
| `eval`, `lod`, `bench`, `report`, `study` | evaluation and experiments | later |

## Licence

Apache-2.0. Third-party licences are listed in
[`THIRD_PARTY.md`](THIRD_PARTY.md).
