# psplat — Work plan

Derived from `docs/handoff.md` (v0.4, 2026-09-13). This file is the execution
breakdown: PR-sized items, dependencies, acceptance criteria, risks, and the
project's standing decisions. The handoff stays the source of truth for *what*
and *why*; this file is *in what order and how we know it is done*.

Every item below is mirrored as a GitHub issue (2026-09-20): six phase epics
([P0](https://github.com/zach/psplat/issues/1), [P1](https://github.com/zach/psplat/issues/2), [P2](https://github.com/zach/psplat/issues/3), [P3](https://github.com/zach/psplat/issues/4), [P4](https://github.com/zach/psplat/issues/5),
[P5](https://github.com/zach/psplat/issues/6)) with each work item as a sub-issue. The issue carries the
same acceptance criteria and is where status lives; this table is the map.

Conventions used below:

- **Surface** = where the item must work: `Win` (stock Windows TD, PowerShell),
  `Mac` (operator), `AWS` (Linux GPU box), `any` (pure-Python, no GPU).
- **Size** = S (≤ half a day), M (1–2 days), L (3–5 days). Sizes are for
  Claude Code doing the work with Zach reviewing; wall-clock for E-runs is
  extra and noted.
- Every item is one PR unless stated. Every PR touching `engines/unreal/`,
  `docs/README.md`, or packaging gets the §0.1 question in its description:
  *"Would a stock Windows TD with PowerShell and no Mac get through this?"*

---

## 0. Facts verified on 2026-09-13 that shape the plan

| Dependency | Verified state | Consequence |
|---|---|---|
| **gsplat** | PyPI is at 1.5.3 (May 2025). `main` carries 1.6.0: requires PyTorch ≥ 2.7; **"Gaussian ID rasterization — rasterize Gaussian IDs, counts, and top contributors per pixel"**, also in the sparse rasteriser. Prebuilt wheels are published for Linux **and Windows** for specific Python/Torch/CUDA triples via a wheel index. | `gsplat_backend` uses the Gaussian-ID / top-contributor op directly (no custom backward pass). Install from git at a pinned SHA; `[cuda]` extra pins one wheel-index triple (P1.10). API may still move before 1.6.0 ships → R5. |
| **msplat** | 1.1.4 (2026-08-25), Python 3.12/3.13, macOS 14+ arm64 only. `GaussianTrainer.render_from_pose(cam2world 4×4) → (H, W, 3) float32`. **No per-splat contribution API.** | Confirms the gap named in handoff §5.2. `msplat_backend` needs a contribution approximation (P1.5). Python floor is 3.12. |
| **XScene-UEPlugin** | Latest release v1.1.5.1 adds **UE 5.5** support; no mention of 5.6. Release zips attached (platform not stated on the page). | Path A targets 5.5 as planned. Confirm the zip contains Win64 binaries before relying on "nothing of ours to compile" (P1.9 spike step 1). |
| **splat-transform** | 3.4.2 on npm (3.10.3 by 2026-11-07; see D7). Reads PLY/compressed PLY/SOG/streamed SOG/SPZ/SPLAT/KSPLAT; writes PLY, SOG, streamed SOG, SPZ, GLB + `KHR_gaussian_splatting`, LOD, HTML viewer. Has `--morton-order` and `--decimate`. **No index-range filter and no stated guarantee that SOG preserves input order.** | **R1** (below): SOG lays splats out as image pixels; if the writer reorders, "first N" is lost in the compressed container. Spike before P1.3. |
| **This container** | Python 3.11 default, 3.13 present, uv can fetch 3.12. Node 22. No GPU, no torch. | P0.1–P0.7 (CPU-only) can be built and tested here. AWS/Mac/GPU items are executed by Zach or on the box. |

---

## 1. Work breakdown

### [P0](https://github.com/zach/psplat/issues/1) — Bootstrap (week 1)

Exit (handoff §7): AWS Linux + Mac envs reproducible from scripts; `torch_backend`
implemented and passing as oracle; E0 numbers match; CI green on macOS **and
Windows** (incl. `analyze --backend torch`) on the synthetic fixture.

| ID | Item | Acceptance | Depends | Surface | Size |
|---|---|---|---|---|---|
| [P0.1](https://github.com/zach/psplat/issues/7) | **Repo skeleton + packaging.** `pyproject.toml` (hatchling, `requires-python>=3.12`, pure wheel `py3-none-any`, extras `cuda`, `msplat`, `cvvdp`, `dev`, `aws`), `LICENSE` (Apache-2.0), `THIRD_PARTY.md` stub, `ruff` + `mypy --strict` config, `pytest` config, package layout from handoff §3.2 with empty modules. | `uv sync --extra dev`; `uv build` yields a `py3-none-any` wheel; `psplat --help` lists all §3.3 subcommands (stubs). | — | any | S |
| [P0.2](https://github.com/zach/psplat/issues/8) | **Camera + scene data model.** `psplat/io/cameras.py`: `cams.json` schema v1 (per-view `w2c` 4×4, `fx fy cx cy`, `width height`, `name`), converters from COLMAP `sparse/0` (binary + text) and nerfstudio `transforms.json`. `Scene` dataclass (means, scales, quats, opacities, sh0, shN, extra attrs) in `psplat/io/scene.py`. | Round-trip tests on hand-built cases; converter tests on a tiny checked-in COLMAP sample. | P0.1 | any | M |
| [P0.3](https://github.com/zach/psplat/issues/9) | **`io/ply.py`.** Read/write 3DGS PLY (all 62 standard attrs: xyz, normals, `f_dc_*`, `f_rest_*`, opacity, `scale_*`, `rot_*`), preserve unknown attributes, memory-mapped read for multi-million-splat files, deterministic write. | Byte-identical round-trip on fixture; round-trip on a real `garden` PLY (Zach, Mac; recorded in lab notebook). Reading 3M splats < 5 s. | P0.2 | any | M |
| [P0.4](https://github.com/zach/psplat/issues/10) | **Synthetic fixture.** Deterministic 200-splat scene generator + 8 cameras, checked in as `tests/fixtures/synthetic/` (PLY + `cams.json`), plus a generator script so it can be regenerated. Include known-occluded splats and known off-screen splats so contribution tests have oracles. | Fixture loads through `io/ply`; docs/testing.md describes it. | P0.3 | any | S |
| [P0.5](https://github.com/zach/psplat/issues/11) | **`Renderer` protocol + projection core + `null_backend`.** `render/base.py`: `Renderer.contributions(scene, cameras) -> (N, V)` and `Renderer.render(scene, camera) -> (H, W, 3)`. `render/project.py`: shared pure-Torch projection (means → pixels, 3D → 2D covariance, per-splat σ_px footprint, depth), used by every backend and by `csf.py`. `null_backend` raises a clear error for `analyze` and satisfies the protocol for tests. | Projection tested against hand-computed cases; `--backend null` path exercised by the CLI tests. | P0.2 | any | M |
| [P0.6](https://github.com/zach/psplat/issues/12) | **`torch_backend` (the oracle).** Pure PyTorch tile-sorted alpha accumulation: per-view depth sort, per-tile culling, front-to-back transmittance, accumulate `alpha·T` per splat into `c_iv`; also returns the rendered image. Correctness first, vectorise second. Runs on CUDA / MPS / CPU. Prints a runtime estimate before starting. | Analytic tests: single splat, two overlapping (ordering), full occluder → zero contribution, off-screen → zero. Renders match a reference image of the fixture within tolerance. Fixture `analyze` < 30 s on CPU. | P0.4, P0.5 | any | L |
| [P0.7](https://github.com/zach/psplat/issues/13) | **CLI wiring (typer).** All §3.3 subcommands exist; `analyze --backend torch` produces `scene.weights.npz` (`contribution (N,V)`, `sigma_px (N,V)`, `visible (N,V)`, meta). Backend auto-selection (`gsplat` if importable + CUDA → `msplat` on Mac → `torch`) with a one-line notice. | `psplat analyze fixture.ply --cameras cams.json --backend torch` succeeds in CI on all three OSes. | P0.6 | Win/any | S |
| [P0.8](https://github.com/zach/psplat/issues/14) | **CI.** GitHub Actions: `ubuntu-latest` (fast lint + tests), `macos-15` (tests + `msplat` import smoke; was `macos-14` until GitHub retired that image in 2026), **`windows-latest`** (build wheel → clean venv → `pip install` wheel → `order/pack/eval/lod/report` with `null_backend` on fixture → `analyze --backend torch` → PowerShell doc-block runner over `docs/`). Manual `workflow_dispatch` CUDA job targeting a self-hosted runner label, initially no-op. Windows job marked required in branch protection (Zach). | Green on all CPU jobs; README PowerShell blocks execute literally. | P0.7 | Win/Mac | M |
| [P0.9](https://github.com/zach/psplat/issues/15) | **Docs pass 1.** `docs/env.md` (Mac + AWS setup), `docs/windows.md` (pure-wheel install, `[cuda]` wheel triple placeholder, MSVC fallback, "nvcc not found"), `docs/backends.md` (protocol, tolerance, msplat gap), `docs/testing.md`. Every user command in bash **and** PowerShell fenced blocks. | Doc-block runner passes on Windows. | P0.8 | Win | S |
| [P0.10](https://github.com/zach/psplat/issues/16) | **AWS scripts.** `scripts/aws_provision.sh` (g6e.2xlarge / g6.xlarge, DLAMI AL2023 via SSM public parameter looked up at run time, 500 GB gp3, IAM role scoped to the bucket, `project=psplat` tag, SSM-only access), `scripts/aws_bootstrap.sh` (idempotent; from handoff §5.1), `scripts/aws_stop_idle.sh` + cron, `scripts/s3_sync.sh`. Budget alarm. | Zach provisions from the Mac; `python -c "import gsplat, torch"` succeeds on the box; versions recorded in `docs/env.md`. | P0.1 | Mac→AWS | M |
| [P0.11](https://github.com/zach/psplat/issues/17) | **E0 — reproduce baseline.** `psplat/bench/scenes.yaml` registry; `bench fetch` (Mip-NeRF 360 core to S3, never redistributed); `bench train` wrappers: gsplat MCMC `cap_max` 1e6 and 3e6, 30K iters, every-8th eval split, on AWS; msplat 7K on Mac. Saves PLY + `cams.json` + eval renders + `run.json`. | PSNR/SSIM within 0.3 dB / 0.02 of published gsplat numbers on garden/bicycle/counter/room; fixtures in `s3://…/trained/`. **Wall-clock: ~1–2 GPU-h per scene**; runs overlap with P1 work. | P0.10, P0.3 | AWS + Mac | M + runtime |
| [P0.12](https://github.com/zach/psplat/issues/18) | **Lab notebook.** `docs/lab-notebook.md` started (see today's entry). Append-only. | Each E-run appends an entry. | — | any | S |

### [P1](https://github.com/zach/psplat/issues/2) — Tools alpha (weeks 2–3)

Exit: `analyze/order/pack/eval` end-to-end on garden; web viewer slider works;
Blueprint-only Unreal sample authored on the Mac and confirmed on a Windows
machine; `psplat[cuda]` smoke-tested on Windows NVIDIA; E1 baseline zoo done.

| ID | Item | Acceptance | Depends | Surface | Size |
|---|---|---|---|---|---|
| [P1.0](https://github.com/zach/psplat/issues/19) | **Spike R1 — SOG order preservation.** Write a PLY with a known ordering, convert with `splat-transform` to SOG and back, check permutation. If SOG (or Streamed SOG) reorders: decide between (a) our own SOG writer that lays pixels out in our order (spec is open; WebP encode via Pillow), (b) SOG carries a per-splat `psplat_rank` side channel, (c) SOG is "full quality archive" only and progressive truncation lives in PLY/SPZ/glTF. Also check SPZ and GLB order preservation. | Written finding in `docs/formats.md`; decision recorded in §4 R1 below. **Half a day, do before P1.3.** | P0.3 | Mac/any | S |
| [P1.1](https://github.com/zach/psplat/issues/20) | **Baseline importance scores** `analyze/scores_baseline.py`: `opacity`, `lightgaussian` (opacity × accumulated contribution), `pup-simple` (Hessian sensitivity, simplest reading), `coreset-simple` (sensitivity sampling), `minisplatting-simple` (view-dependent). Each a pure function over `weights.npz` arrays + scene. `docs/baselines.md` records every ambiguity and the reading chosen. | Toy-case unit tests (e.g., duplicated splat gets half score; invisible splat scores 0). | P0.7 | any | M |
| [P1.2](https://github.com/zach/psplat/issues/21) | **`order` + manifest.** `order/order.py` (sort desc, tie-break deterministic, cut points), `io/manifest.py` (schema §3.4, `version: 1`, JSON-schema validated). `psplat order` writes `scene.ordered.ply` + `scene.psplat.json`. **Schema frozen at end of P1** (handoff §8.7). | Manifest validates; `cuts[].index` monotone; ordered PLY re-reads with identical attributes. | P1.1, P0.3 | any | S |
| [P1.3](https://github.com/zach/psplat/issues/22) | **`pack`.** Ordered PLY (native); SOG / SPZ / GLB-KHR via `splat-transform` shell-out (`npx` with Node check, PowerShell-friendly error) **or** own writer per R1 outcome; per-cut byte sizes into manifest. `--alloc uniform` only (perceptual in P3.1). | Outputs open in SuperSplat (manual, Mac) and re-import via `splat-transform`; truncation at `cuts[k].index` reproduces the same splats as PLY truncation. Windows CI runs the PLY path and skips SOG if Node is absent, with a clear message. | P1.0, P1.2 | Win | M |
| [P1.4](https://github.com/zach/psplat/issues/23) | **`gsplat_backend`.** Use Gaussian-ID / top-contributor rasterisation from gsplat `main` (pinned SHA) for `c_iv`; fall back to accumulated-alpha backward if the op is missing. Tolerance test vs `torch_backend` on the fixture in the CUDA job. Benchmark on garden (3M splats × all train views). | Contributions match oracle within documented tolerance (start: Spearman ≥ 0.99, max rel. err ≤ 5%); garden `analyze` < 5 min on L40S. | P0.6, P0.10 | AWS/Win | L |
| [P1.5](https://github.com/zach/psplat/issues/24) | **`msplat_backend`.** `render_from_pose` for images; contribution approximation: (a) projected-footprint × opacity × visibility estimate from `render/project.py` (cheap), (b) leave-group-out re-render deltas (expensive, validation only). Gap documented in `docs/backends.md`. | Agreement with oracle measured and recorded; used only for Zach's local iteration, never for headline numbers unless within tolerance. | P0.6 | Mac | M |
| [P1.6](https://github.com/zach/psplat/issues/25) | **`eval/metrics.py` + `psplat eval`.** PSNR, SSIM, LPIPS (BSD), ꟻLIP (NVIDIA BSD-3, via `flip-evaluator` or a small Torch port — D4), ColorVideoVDP behind `[cvvdp]` extra after licence check. Renders truncated scenes at each budget with the selected backend (torch on CPU for the Windows path) and writes `results.parquet` with provenance columns from handoff §4.2 step 5. | Metric unit tests against known image pairs; eval on fixture in CI (torch backend). | P0.7, P1.2 | Win | M |
| [P1.7](https://github.com/zach/psplat/issues/26) | **Web viewer** `engines/web/`. PlayCanvas engine (MIT), loads SOG (or PLY fallback), reads manifest, budget slider snapping to `cuts[]` or raw count, split-screen A/B vs full, FPS + bytes readout, projected-size threshold control (for E4). `psplat viewer dist/scene.sog` serves it locally (static files + `http.server`). No build step required for the TD; vendored minified engine or pinned CDN. | Opens in Chrome/Edge on Windows from `psplat viewer`; slider changes rendered splat count; works on the fixture and on garden. | P1.3 | Win | L |
| [P1.8](https://github.com/zach/psplat/issues/27) | **Unreal sample — Path A, two-day time-box** `engines/unreal/PsplatSample/`. Steps: (1) download XScene v1.1.5.1 release zip, confirm Win64 payload and UE 5.5 target; (2) open in UE 5.5 on the Mac, import a plain PLY — if Metal rendering fails, read the Niagara emitter's parameter surface from source and author against a stub emitter with identical parameter names; (3) identify the particle-count / spawn-count parameter and confirm the PLY is consumed in file order; (4) `PsplatBudget` Actor Component (Blueprint) reading `scene.psplat.json`, `SetBudgetFraction()`, `SetBudgetSplats()`, cut-point enum; (5) sample map: full vs budgeted actors side by side, UMG slider, `stat unit`; (6) `docs/unreal.md` with screenshots + a short C++ equivalent of each Blueprint; ≤10-line "drop in your own PLY". **If step 3 or 4 blows the time-box: stop and raise Path B with Zach; do not start it.** | A Windows contact opens the project from a fresh UE 5.5 Launcher install and confirms the slider drives particle count (screenshot in lab notebook). | P1.2 | Mac author → Win validate | L (boxed at 2 days) |
| [P1.9](https://github.com/zach/psplat/issues/28) | **Baseline `bench` + `report`.** `bench run` (scene × ordering × budget → analyze/order/pack/eval, idempotent, `run.json` each), `bench collect` → parquet, `report` → static HTML with rate–quality curves (PSNR/SSIM/LPIPS/ꟻLIP vs budget) and image strips (full / 25% / 10%) + ꟻLIP heatmaps. DuckDB for queries. | `psplat bench --scenes mipnerf360-core --orderings baselines --out bench/` then `psplat report bench/` reproduces E1 from one command. | P1.4, P1.6 | AWS | M |
| [P1.10](https://github.com/zach/psplat/issues/29) | **`psplat[cuda]` on Windows.** Pin the gsplat wheel-index triple (Python / Torch / CUDA) in the extra; document it in `docs/windows.md`; MSVC fallback section. One Windows contact runs `pip install psplat[cuda]` and `analyze --backend gsplat` on garden. | Their versions + wall time recorded in `docs/windows.md`. | P1.4 | Win | S + external |
| [P1.11](https://github.com/zach/psplat/issues/30) | **E1 — importance-score zoo.** Run P1.9 on the four core scenes at 5/10/25/50/100%. | Curves for ≥ 4 baselines; report committed under `bench/reports/e1/`; notebook entry. | P1.9, P0.11 | AWS | runtime |
| [P1.12](https://github.com/zach/psplat/issues/31) | **README = TD guide v1.** The 6-command afternoon path in bash + PowerShell, "what to expect" budget-vs-quality table (filled from E1), links to web viewer and Unreal sample. | Windows doc-block runner passes; a reader with no Mac can follow every step. | P1.3, P1.7, P1.8 | Win | S |

### [P2](https://github.com/zach/psplat/issues/3) — Perception (weeks 4–7)

Exit: E5 metric-calibration report; E2 + E2a results; headline metric chosen.

| ID | Item | Acceptance | Depends | Surface | Size |
|---|---|---|---|---|---|
| [P2.1](https://github.com/zach/psplat/issues/32) | **`csf.py`.** `f_iv = ppd / (2·σ_px)` from `render/project.py`; Barten CSF at configurable ppd (default 60) and mid luminance; castleCSF optional. | CSF peaks in the 3–5 cpd range; unit tests on tabulated values; ablation flag `--terms csf`. | P0.5 | any | S |
| [P2.2](https://github.com/zach/psplat/issues/33) | **`masking.py`.** Local RMS contrast of the reference render inside each splat's footprint (from the backend's rendered image); Daly-style threshold elevation as a power law with one exponent parameter. | Flat-region splats get weight 1; textured-region splats attenuated; tests on synthetic gratings. | P0.6 | any | M |
| [P2.3](https://github.com/zach/psplat/issues/34) | **`saliency.py`.** Centre-weighted prior by default; pluggable interface; optional person/face detector behind an extra (licence-checked); DeepGaze later. | Off by default; `--terms saliency` toggles it. | P0.5 | any | S |
| [P2.4](https://github.com/zach/psplat/issues/35) | **`weights.py` compose.** `w_i = Σ_v c·S·M·A·V`, `V` uniform; terms selectable; writes `ordering.method = perceptual-v1` + terms into the manifest. | Ablation is a flag flip; deterministic. | P2.1–P2.3, P1.2 | any | S |
| [P2.5](https://github.com/zach/psplat/issues/36) | **E5 — metric calibration.** Fetch 3DGS-QA and GScomp-QA (licence check each); compute PSNR/SSIM/LPIPS/ꟻLIP/CVVDP vs MOS; Spearman/PLCC table; `docs/metrics.md` names the headline metric. | Report committed; decision recorded in lab notebook. **Run before E2 headline numbers.** | P1.6 | Mac or AWS | M |
| [P2.6](https://github.com/zach/psplat/issues/37) | **E2 + E2a.** Perceptual ordering vs the E1 baselines on the core scenes + Tanks&Temples truck/train; per-term ablation table. | Success criterion from handoff §2.2 evaluated honestly and recorded either way. | P2.4, P2.5, P1.11 | AWS | runtime |

### [P3](https://github.com/zach/psplat/issues/4) — Allocation, LOD, acceptance (weeks 8–9)

| ID | Item | Acceptance | Depends | Surface | Size |
|---|---|---|---|---|---|
| [P3.1](https://github.com/zach/psplat/issues/38) | **`allocate.py` + `pack --alloc perceptual`.** Weight quantiles → SH band count + quantiser step per splat. Requires a container that supports per-splat variable precision — SOG/SPZ quantise uniformly, so this likely means our own SOG writer (R2) or a "banded" layout (splats grouped by allocation tier, each tier its own SOG chunk). | Equal-byte comparison vs uniform SOG and SPZ v4 is possible and reproducible. | P1.3, P2.4 | any | L |
| [P3.2](https://github.com/zach/psplat/issues/39) | **E3 — attribute allocation.** | ꟻLIP/LPIPS at matched bytes; criterion from §2.2. | P3.1 | AWS | runtime |
| [P3.3](https://github.com/zach/psplat/issues/40) | **`lod.py` + `psplat lod` + distance LOD in both engines.** Projected-size thresholds per viewing distance → `lod.json`; web viewer threshold control; optional Blueprint mapping camera distance → budget. | `lod.json` schema in `docs/formats.md`; both engines consume it. | P1.7, P1.8 | Win | M |
| [P3.4](https://github.com/zach/psplat/issues/41) | **E4 — distance robustness + frame time.** Zoom trajectories; quality vs threshold; rank-inversion stat; frame-time vs budget in the web viewer (Mac) and in Unreal (`stat unit` / Insights, captured during the TD run via a supplied script). | Curves in report; X for rank inversions set after pilot and recorded. | P3.3 | Mac + Win TD | runtime |
| [P3.5](https://github.com/zach/psplat/issues/42) | **Format validation.** SOG/SPZ/GLB outputs opened in SuperSplat; glTF validated; findings in `docs/formats.md`. | Screenshots + versions recorded. | P1.3, P3.1 | Mac | S |
| [P3.6](https://github.com/zach/psplat/issues/43) | **Release + acceptance.** GitHub Release zip of the Unreal sample; one-page TD checklist; `stat unit` capture script; **external Windows TD runs the afternoon test unassisted, including local `analyze`** (torch or `[cuda]`). | TD's checklist returned complete; frame-time curve received; blockers fixed and re-tested. | P1.12, P3.3 | Win (external) | M + external |

### [P4](https://github.com/zach/psplat/issues/5) — Study and write-up (weeks 10–12)

| ID | Item | Acceptance | Depends | Surface | Size |
|---|---|---|---|---|---|
| [P4.1](https://github.com/zach/psplat/issues/44) | **Study rig** `psplat study serve`: static full-screen 2AFC page, FastAPI backend, SQLite responses, S3 sync; stimuli as pre-rendered PNG/MP4 pairs; distance calibration instructions (60 ppd). Preregistration doc (`docs/study-prereg.md`) with design, n≈15, Bradley–Terry analysis plan. | Dry run with Zach; **ask before running with participants** (handoff §8.7). | P2.6 | Mac | M |
| [P4.2](https://github.com/zach/psplat/issues/45) | **E6 — human study.** 3 scenes × 3 budgets × 2 orderings. | Bradley–Terry scores + CIs; notebook entry. | P4.1 | Mac | runtime |
| [P4.3](https://github.com/zach/psplat/issues/46) | **Video (2–3 min).** Segments from handoff §6; Unreal segment via Movie Render Queue on the Mac if XScene renders on Metal, else screen capture supplied by the Windows TD. | Cut delivered; assets under `docs/media/` or a release. | P3.4, P3.6 | Mac | M |
| [P4.4](https://github.com/zach/psplat/issues/47) | **Paper sketch (4–6 pp, graphics venue)** with figures regenerated by `psplat report`. | Draft in `paper/`; figures reproducible from `bench/`. | P2.6, P3.2, P4.2 | Mac | L |
| [P4.5](https://github.com/zach/psplat/issues/48) | **v0.1 tag.** Changelog, `THIRD_PARTY.md` complete, wheel on PyPI (or TestPyPI first), Unreal sample release. | `pip install psplat==0.1.0` works in a clean Windows venv. | all | Win | S |

### [P5](https://github.com/zach/psplat/issues/6) — Stretch (open)

[P5.1](https://github.com/zach/psplat/issues/49) Unity (aras-p) and Godot (GDGS) samples ·
[P5.2](https://github.com/zach/psplat/issues/50) Splatwizard integration ·
[P5.3](https://github.com/zach/psplat/issues/51) E7 exploratory dynamic clip ·
[P5.4](https://github.com/zach/psplat/issues/52) gaze-prior design note (handoff §9.2).
None start before P3.6 passes.

---

## 2. Critical path and first moves

```
P0.1 → P0.2 → P0.3 → P0.4 → P0.5 → P0.6 → P0.7 → P0.8 → P0.9        (CPU, this week, buildable here)
P0.1 → P0.10 → P0.11 (E0, AWS/Mac, Zach)                            (parallel)
P1.0 spike (R1) as soon as P0.3 lands                                (parallel)
P1.1 → P1.2 → P1.3 → P1.7 (web) and P1.8 (Unreal)                   (the product path)
P1.4 → P1.9 → P1.11 (E1)                                             (the benchmark path)
P1.6 → P2.5 (E5) → P2.6 (E2)                                         (metrics before headline)
```

The long poles are `torch_backend` (P0.6), `gsplat_backend` (P1.4), the web
viewer (P1.7) and the Unreal sample (P1.8). Nothing perceptual (P2.x) starts
until `order`, `pack`, the web viewer and the Unreal sample exist, per
handoff §8.4.

**What Claude Code can do in this remote container right now:** P0.1–P0.9
(pure-Python, CPU, CI). **What needs Zach or the box:** P0.10, P0.11, P1.4
(GPU), P1.5 (Mac), P1.8 (UE on the Mac), every Windows check.

**On the "external TD" acceptance criteria** (D5, decided 2026-09-20): there
are no named external testers to coordinate with. The afternoon test stays the
design north star — it is what keeps the §0.1 discipline real — but it is a
directional bar, not a scheduling dependency. Wherever the handoff says "an
external Windows TD confirms X", the operative acceptance is: X is proven by
the Windows CI job and by a Windows run Zach drives himself, and anything not
actually verified is recorded as unverified rather than assumed. Items
affected: P1.8, P1.10, P3.4 (Unreal frame time), P3.6. If a real tester turns
up later, the stricter bar reapplies unchanged.

Proposed first PR series on `claude/upbeat-knuth-u1xp0b`:

1. `docs/`: this plan, the handoff, the lab notebook (this commit).
2. P0.1 skeleton + packaging + CI scaffold (CI runs lint only until P0.7).
3. P0.2 + P0.3 + P0.4: data model, PLY I/O, fixture.
4. P0.5 + P0.6: renderer protocol, projection core, torch oracle.
5. P0.7 + P0.8 + P0.9: CLI, full CI matrix, docs pass 1.
6. P0.10: AWS scripts (for Zach to run).

---

## 3. Schemas to fix early (changing them later is expensive)

- **`cams.json` v1** (P0.2): list of views with `name`, `width`, `height`,
  `fx fy cx cy`, `w2c` (row-major 4×4, OpenCV convention: +x right, +y down,
  +z forward), optional `split: train|eval`. Converters from COLMAP and
  nerfstudio provided; nothing else reads those formats directly.
- **`weights.npz`** (P0.7): `contribution (N,V) float32`, `sigma_px (N,V) float32`,
  `visible (N,V) bool`, `view_names (V)`, `meta` JSON string (backend, git sha,
  ppd, wall time). Per-term arrays added in P2 under their own keys.
- **`scene.psplat.json`** (P1.2): exactly handoff §3.4; `version: 1`; frozen at
  end of P1; any later change bumps `version` and is asked about first.
- **`lod.json`** (P3.3): `{"ppd": 60, "distances": [...], "sigma_px_min": [...],
  "budget_index": [...]}`.

---

## 4. Risks and spikes

| ID | Risk | Likelihood / impact | Mitigation |
|---|---|---|---|
| **R1** | SOG (via splat-transform) does not preserve splat order, so "first N" truncation is lost in the compressed container. | **Confirmed, then resolved (P1.0, 2026-11-07).** The command-line SOG, compressed-PLY and Streamed SOG writers all Morton-sort. | splat-transform's library `writeSog` takes an `indices` order; psplat passes a banded order (Morton within cut-point bands), so every cut point is an exact prefix at about 11–14% more bytes than Morton. See D7 and `docs/formats.md`. |
| R2 | SOG/SPZ quantise uniformly; per-splat variable precision (H2/E3) has no native home. | High / medium — affects E3 only. | Banded layout (tiers → separate chunks) or own writer; worst case E3 demonstrates on PLY-level quantisation plus a proposed extension note (handoff §9.2.4). |
| R3 | XScene does not render on the Mac (Metal). | Medium / low — plan already allows authoring against a stub emitter. | P1.8 step 2; do the Windows render check as early as the sample opens. |
| R4 | XScene's particle-count knob does not map to "first N in file order" (e.g., it spatially sorts on import). | Medium / high — breaks the Blueprint-only story. | P1.8 step 3 checks consumption order from source. If sorted on import, fall back to exporting per-cut PLYs (`scene.25.ply`, …) and switching assets, or raise Path B. **Do not start Path B without Zach.** |
| R5 | gsplat Gaussian-ID op is on unreleased `main`; API may change. | Medium / low. | Pin a SHA; keep the accumulated-alpha fallback in `gsplat_backend`; the oracle test catches drift. |
| R6 | msplat contribution approximation is too loose to be useful. | Medium / low — Mac is for iteration, not headline numbers. | Measure against the oracle (P1.5); if poor, Mac iteration uses `torch_backend` on MPS. |
| R7 | No prebuilt gsplat Windows wheel for the Torch/CUDA triple a TD has. | Medium / medium. | Pin one triple in `[cuda]` and document it; `torch_backend` is the guaranteed path; MSVC fallback documented. |
| R8 | Licences: ColorVideoVDP, DeepGaze, any face detector. | Low / medium. | Check before adding; keep behind extras; `THIRD_PARTY.md`; ask Zach for anything non-permissive (handoff §8.7). |
| R9 | Dataset licences (Mip-NeRF 360, T&T, 3DGS-QA, GScomp-QA). | Low / high if violated. | Fetch scripts only, never redistribute; S3 bucket private; nothing in the repo or video except our own captures and CC data. |
| R10 | Baseline-paper ambiguity (PUP Hessian, coreset sensitivity). | High / low. | `<method>-simple` naming + `docs/baselines.md` (handoff §8.9). |
| R11 | No external testers, so the Windows consumer surface is only ever exercised by us. Blind spots in the afternoon test (an assumed tool, an admin right, a path quirk) survive to release. | Medium / medium. | Windows CI does the literal work: clean-venv wheel install, the full `order/pack/eval/lod/report` path, `analyze --backend torch`, and the README's PowerShell blocks executed verbatim. Zach drives one manual Windows run per phase exit. Unverified steps are labelled unverified in the docs. |
| R12 | AWS spend. | Low / medium. | Spot for batch, idle-stop timer, budget alarm, ask before any single action > $50. |
| R14 | SOG's spherical-harmonic compression needs a GPU (WebGPU). With no adapter splat-transform fails outright, and with `-g cpu` a 100k-splat, SH degree 3 scene took over 25 minutes; the k-means palette grows with the scene up to 64K entries (measured 2026-11-07). | Medium / medium — a TD's machine may lack a working WebGPU adapter, and SOG is the web path. | `psplat pack` tries the GPU and, on failure, writes the SOG without higher SH bands or skips SOG with a clear notice; never a silent long CPU run. Owned by P1.3 ([#22](https://github.com/zach/psplat/issues/22)). |
| R13 | The portable torch backend is slow on CPU: 3–6 s per 1237×822 view at 300k splats, so a 3M-splat scene over ~160 views is hours, not the tens of minutes the handoff hoped for (measured 2026-09-27). | High / medium — weakens "analyze runs locally" for TDs without an NVIDIA GPU. | The cost is alpha evaluation, not Python overhead. Options: analyse a subset of views or at reduced resolution for CPU runs; vectorise across tiles; rely on `psplat[cuda]` (P1.10). Tracked in [#53](https://github.com/zach/psplat/issues/53). |

---

## 4a. P0 status (2026-09-27)

Built and pushed; details in `docs/lab-notebook.md`. P0.1, P0.2, P0.4–P0.9 meet
their acceptance. Still open, each for a reason outside the container:
P0.3 (round trip on a real garden PLY, needs the Mac), P0.10 (not yet run
against AWS), P0.11 (E0 needs the GPU box and the published reference
numbers), P0.12 (process, open by design).

Corrections to the rows above, learnt while building:

- P0.4 and P0.5 were built in the opposite order, because the fixture's
  oracles are defined through the projection conventions.
- P0.8's Windows job runs `cameras` and `analyze` today. `order`, `pack`,
  `eval`, `lod` and `report` do not exist until P1, so each P1 item that adds
  one must also add it to the `td-windows` job and the README.
- D2's AMI: the DLAMI SSM parameter name is discovered at run time from the
  public parameter tree rather than hard-coded.

## 5. Decisions

Resolved 2026-09-20 unless marked open.

| ID | Question | Decision |
|---|---|---|
| D1 | `cams.json` convention | **Own minimal schema v1** (OpenCV camera convention) plus COLMAP and nerfstudio converters. Nothing else reads those formats directly. |
| D2 | AWS region / bucket | **`us-east-1`**, bucket `psplat-<acct>`; both come from env vars so the scripts carry no account detail. |
| D3 | Python floor | **3.12.** msplat ships 3.12/3.13 wheels; ours is pure so 3.13 works too. |
| D4 | ꟻLIP implementation | **NVIDIA `flip-evaluator`** (BSD-3) if it installs cleanly on Windows CPU; otherwise a small Torch port under `eval/flip.py`. It has to run in CI either way. |
| D5 | External Windows contact (P1) and TD (P3) | **None — these are directional goals, not people to coordinate with.** The afternoon test stays the design bar; acceptance is Windows CI plus a manual Windows run Zach drives. Unverified steps are labelled unverified. See §2 and R11. |
| D6 | Mirror the plan to GitHub issues | **Done 2026-09-20.** Six phase epics, 46 sub-issues, same acceptance criteria as the rows here. GitHub issue types (Task / Feature) rather than milestones, since the repo has no milestones set up. |
| D7 | R1 outcome: does SOG preserve splat order? | **Resolved 2026-11-07 (P1.0).** No, not through splat-transform's command line. SOG is written through its library with a banded `indices` order aligned to the manifest's cut points, so every cut point is an exact prefix; the viewer snaps to cut points for SOG. PLY, SPZ and glTF keep order natively and are the arbitrarily truncatable formats; Unreal uses PLY. Compressed PLY and Streamed SOG are not offered as ordered outputs. splat-transform is pinned (3.10.3) and an interop test asserts all of this. Details: `docs/formats.md`. |

---

## 6. Standing rules (from handoff §8, restated so they are in the working doc)

- `torch_backend` is written before `gsplat_backend`; it is the oracle.
- `order` + `pack` + web viewer + Unreal sample before any perceptual experiment.
- The §0.1 test on every PR touching `engines/unreal/`, `docs/README.md`, or packaging.
- Every experiment writes `run.json`; no result without provenance.
- Ask before: > $50 in one AWS action; running the human study; a non-permissive dependency; changing the manifest after P1.
- `docs/lab-notebook.md` is append-only.
- Boring tech: parquet + DuckDB, static HTML, `typer`, `uv`.
