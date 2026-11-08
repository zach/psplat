<!-- psplat-docs: audience=dev -->
# Packed formats and splat order

psplat's product is a file whose first N splats are its best N. That only
survives packing if a format keeps splats in the order they were given. This
page records which formats do, measured with PlayCanvas splat-transform
3.10.3 on 2026-11-07, and the decision that follows (D7 in `docs/plan.md`).
It answers spike P1.0 ([#19](https://github.com/zach/psplat/issues/19)).

## How it was measured

`scripts/spikes/order_preservation.py` places splats on a jittered 3D grid,
offset from the origin, and writes them in random order. Each splat's grid
cell identifies it even after lossy quantisation. After a round trip through a
format, the output order maps back to input indices exactly, so any spatial
re-sorting by a writer shows up as a permutation.

To rerun it:

<!-- doctest: skip -->
```bash
python scripts/spikes/order_preservation.py --n 4096 100000 --size 100000 --sh-degree 0
```

```powershell
python scripts\spikes\order_preservation.py --n 4096 100000 --size 100000 --sh-degree 0
```

## Which formats keep order

Results at 100,000 splats; 4,096 splats gave the same verdicts. "Prefixes
kept" lists the budgets, in percent, at which the first k output splats are
exactly the first k input splats.

| Format | Order kept | Prefixes kept | Why |
|---|---|---|---|
| PLY | yes | all | written as given |
| SPZ v4 | yes | all | written as given |
| glTF `KHR_gaussian_splatting` (.glb) | yes | all | written as given; positions are in glTF space, x and y negated relative to PLY |
| SOG, splat-transform command line | **no** | none | the writer Morton-sorts unless given an order |
| SOG, splat-transform library, `indices` = identity | yes | all | the supplied order replaces the sort |
| SOG, library, banded Morton | within bands | all cut points | see below |
| compressed PLY | **no** | none | always Morton-sorted |
| Streamed SOG (`lod-meta.json`) | **no** | none | split into spatial chunks, each Morton-sorted |

The source agrees. splat-transform's SOG writer sorts splats into Morton
(Z-curve) order "unless a caller-supplied order is used". That order is the
`indices` option of the library function `writeSog`, a full-length
permutation that sets which splat goes in which texture pixel. The command
line never passes it.

## What keeping order costs

SOG and SPZ compress better when spatial neighbours sit next to each other,
which is why the writers sort. psplat's order is perceptual, not spatial, so
keeping it exactly costs bytes. Random attributes compress equally badly in
any order, so this was measured on synthetic scenes whose colours, scales and
opacities vary smoothly over surfaces, as a real capture's do. Sizes are
relative to splat-transform's own Morton-sorted SOG of the same scene.

| Layout | 100k splats | 400k splats |
|---|---|---|
| SOG, Morton order (command line) | 1.00 | 1.00 |
| SOG, 6–7 bands, Morton within each | 1.14 | 1.11 |
| SOG, 20 bands (every 5%) | 1.26 | 1.23 |
| SOG, 100 bands (every 1%) | 1.42 | 1.39 |
| SOG, psplat order kept exactly | 1.59 | 1.74 |
| SPZ, psplat order kept exactly | 1.59 | 1.72 |
| SPZ, Morton order | 1.39 | 1.37 |

These are synthetic scenes with SH degree 0. The ratios must be re-measured on
a trained scene (garden) before they are quoted anywhere; P1.3
([#22](https://github.com/zach/psplat/issues/22)) owns that.

### Banded order

A banded order keeps which splats fall in each band between consecutive cut
points, and Morton-sorts within each band. The first k splats at any cut point
are then exactly the perceptual top k, and most of the compression comes back.

Between cut points the prefix is not the perceptual top k. It is the bands
below plus a spatially clustered part of the next band, because Morton order
fills one region first. A viewer showing a banded SOG must snap its budget to
cut points; a truncation between them would look patchy.

## SOG needs a GPU for spherical harmonics

SOG compresses the higher-order spherical-harmonic coefficients with k-means
over a palette of up to 64K entries. splat-transform runs it on the GPU
through WebGPU. With no usable adapter it fails outright, "Unable to retrieve
a WebGPU adapter", unless given `-g cpu`, and on the CPU it is only practical
for small scenes:

| Splats (SH degree 3) | SOG write on 4 CPU threads |
|---|---|
| 8,192 | 2.1 s |
| 32,768 | 4.7 s |
| 100,000 | over 25 min, stopped |

The palette grows with the splat count until it reaches 64K entries, so the
cost rises steeply past about 32K splats. A multi-million splat scene needs the
GPU path. SH degree 0 SOG and every other format here need no GPU.

## Decision (D7)

1. **SOG is written through splat-transform's library**, with `indices` set
   to a banded order whose bands are the manifest's cut points. Every cut
   point is an exact prefix. The web viewer's slider snaps to cut points for
   SOG. Default cuts stay those of the handoff, 5, 10, 25, 50, 75 and 100%,
   for about an 11–14% size cost on the synthetic scenes.
2. **PLY, SPZ and glTF are the exactly truncatable formats.** Any splat
   count is a valid budget. The Unreal sample consumes PLY.
3. **Compressed PLY and Streamed SOG are not offered as ordered outputs.**
   Both reorder unconditionally. Streamed SOG has its own spatial
   level-of-detail scheme, which may matter for P3 but is a different design.
4. **`psplat pack` must handle SOG without a GPU**: try the GPU, and on failure
   either drop the SH bands for the SOG output with a clear notice or skip SOG
   and say why. Never silently fall back to a 30-minute CPU run.
5. **The splat-transform version is pinned**, and an interop test asserts the
   behaviour above, so an upstream change to its writers fails CI instead of
   quietly breaking truncation.

## The manifest, `scene.psplat.json`

`psplat order` writes the manifest next to the ordered PLY. Its schema is
`psplat/io/manifest.schema.json` (JSON Schema 2020-12), `version: 1`, as in
handoff §3.4. It is frozen at the end of P1: any later change bumps
`version`.

Engines need only `cuts[].index`, the number of leading splats in each
budget. Everything else is there for reports and reproducibility:

| Field | Meaning |
|---|---|
| `source_sha256` | SHA-256 of the unordered input PLY |
| `count` | splats in the ordered file |
| `ordering` | `method`, the analysis `terms` it read, and `seed` for sampled methods; later `ppd` and `params` |
| `cuts[]` | `frac` in (0, 1], strictly increasing, ending at 1; `index` never decreasing, at most `count`; `pack` and `eval` add `bytes_<format>` and metric values per cut |
| `lod` | projected-size thresholds, from `psplat lod` (P3.3) |
| `provenance` | tool version, git state, hardware, date, and the backend and weights file the order came from |

A cut's `index` is `round(count · percent / 100)`, at least 1 for a non-empty
scene. That is the same rule the banded SOG layout uses for its band edges,
so every cut is also an exact prefix of a banded SOG.
