<!-- psplat-docs: audience=td -->
# Testing

The whole test suite runs on a CPU in a few seconds, on Windows, macOS and
Linux. No GPU, dataset or network is needed.

## Running the tests

```bash
uv venv -p 3.12 .venv
uv pip install --python .venv -e ".[dev]"
.venv/bin/pytest -m "not slow"
```

```powershell
uv venv -p 3.12 .venv
uv pip install --python .venv -e ".[dev]"
.venv\Scripts\pytest -m "not slow"
```

`-m "not slow"` skips the 3M-splat PLY parsing benchmark, which writes a
744 MB file and times itself, so it is only meaningful on an idle machine.

## The synthetic fixture

`tests/fixtures/synthetic/` holds a 200-splat scene and 8 cameras small
enough that analysing every view takes well under a second on a CPU. It is
built by `generate.py` in the same folder, which is deterministic; the
checked-in files are the source of truth and a test checks the generator
still reproduces them.

| File | Contents |
|---|---|
| `scene.ply` | 200 splats, SH degree 3, the standard 62-property layout |
| `cams.json` | 8 views, 64×48 px, on a circle of radius 4 at height 0.3, all looking at the origin; `view_00` and `view_04` are `eval`, the rest `train` |
| `oracles.json` | the splats whose behaviour is known by construction |
| `generate.py` | regenerates the three files above |

The splats are laid out in index order as follows.

| Indices | What | Known answer |
|---|---|---|
| 0–187 | the cloud: random splats within radius 0.8 of the origin | every one contributes to at least one view |
| 188–192 | off-screen: small splats far above every camera | invisible and zero contribution in all 8 views |
| 193 | the occluded target, above the cloud | exactly zero contribution in `view_01`, yet inside that view's frame; clearly positive in `view_05` |
| 194–197 | four wide, near-opaque occluders on `view_01`'s line of sight to the target | every pixel the target could reach stops before it |
| 198–199 | the overlap pair, front then back, on `view_02`'s line of sight | in `view_02`, front contributes more than back, and back more than zero |

The occluded case is exact rather than approximate because of how the
rasteriser stops. A pixel stops at the first splat that would take its
transmittance to 1e-4 or below, and neither that splat nor anything
behind it is composited. After two occluders the transmittance is already
about 1e-6, so the target never receives anything.

Regenerate the fixture after an intentional change with:

```bash
python tests/fixtures/synthetic/generate.py
```

```powershell
python tests\fixtures\synthetic\generate.py
```

## What the backend tests compare against

`tests/test_torch_backend.py` checks the torch backend against a literal
per-pixel transcription of gsplat's CUDA forward loop, with no tiles, no
chunks and no culling. Both translucent and dense opaque random scenes are
used; in the opaque ones a large share of pixels stop early. Projection and
spherical harmonics are also checked against gsplat's own pure-PyTorch
reference functions when gsplat is installed; those run on CPU.

## Interop test

`tests/test_interop_splat_transform.py` round-trips PLY files through
PlayCanvas splat-transform. It is opt-in because it fetches splat-transform
with `npx`:

<!-- doctest: skip -->
```bash
PSPLAT_INTEROP=1 .venv/bin/pytest tests/test_interop_splat_transform.py
```

```powershell
$env:PSPLAT_INTEROP = "1"; .venv\Scripts\pytest tests\test_interop_splat_transform.py
```
