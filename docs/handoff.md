# psplat — Perceptual, Budget-Scalable Gaussian Splats

**Handoff document for Claude Code** · v0.4 · 2026-09-13
Owner: Zach · Working name `psplat` (rename freely; keep the import path short)
Changes in v0.2: the four open questions in §9 are resolved — graphics venue, no gaze data (noted as follow-up), no proprietary capture data (dynamic/human work demoted to exploratory), **Unreal Engine is the primary engine target**.
Changes in v0.3: **two audiences, two surfaces** (§0.1). Windows is a first-class *consumer* platform (TDs); the Mac is the *operator* platform (Zach).
Changes in v0.4: **the pipeline runs wherever the user is** (§0.2). Full `analyze` runs natively on Windows, Linux and Mac via a three-tier backend (gsplat CUDA / msplat Metal / portable `torch_backend`). No cloud is ever required of a TD, and **no Windows AWS image is used** — the Unreal sample is Blueprint-only over XScene's prebuilt releases, authored on the Mac, validated on an external TD's Windows machine.

---

## 0. One-paragraph brief

Build an open-source (Apache-2.0) toolkit that takes an already-trained 3D Gaussian Splat scene, does an expensive one-time **perceptual analysis** on a GPU box, and emits a **progressively decodable, perceptually ordered** splat file that any engine can truncate at an arbitrary budget (N splats or B bytes) and get the best-looking result for that budget. The analogy is MP3: analysis is slow and psychovisual, decode is trivial and scales to the client. The research question is whether ordering/allocating splats by a **human-visual-system model** (contrast sensitivity, masking, saliency, and for dynamic content, motion) beats the field's standard **contribution-only importance scores** at matched budgets — measured with perceptual metrics *and* a small human study. Deliverables, in priority order: (1) tools a studio TD can use in an afternoon — with **Unreal Engine 5** as the reference engine and a web viewer as the zero-install path, (2) a reproducible data pipeline, (3) a short video + paper-sketch aimed at a **graphics** venue. Gaze-contingent rendering and dynamic/human-performer content are explicitly out of scope for v0.1 and listed as follow-up work (§9).

Everything below is written so Claude Code can execute it with minimal back-and-forth. Where a decision is deferred, it says **DECIDE** and gives a default.

### 0.1 Two audiences, two surfaces (governs every platform decision below)

| | **Consumer surface — studio TD** | **Operator surface — Zach** |
|---|---|---|
| Machine | **Windows 11 workstation with an NVIDIA GPU**; assume no WSL, no admin rights beyond installing UE and Python | **M4 Max MacBook Pro** |
| Touches | `pip install psplat` (pure-Python wheel; `[cuda]` extra for local `analyze`), the Unreal sample project (Blueprint-only, over XScene's release zip), the README, the web viewer, `.ply/.sog` files | Claude Code, AWS control scripts, `msplat` for quick local scenes, the study rig, the web viewer, the Linux GPU box |
| Must never need | a Mac, Linux, Docker, **any cloud account**, a GPU for `order`/`pack`/`eval`, or anything beyond PowerShell | Windows locally, a Windows cloud instance |
| Shell | PowerShell (every README command has a PowerShell form) | zsh |
| Rendering backend | `gsplat_backend` (CUDA, prebuilt wheel) for full-speed local `analyze`; `torch_backend` (portable, slower) if no CUDA; `null_backend` if they only consume pre-analysed files | `msplat_backend` locally; `gsplat_backend` on AWS Linux; `torch_backend` as oracle |
| Bridge | The **web viewer** and the **SOG/PLY + manifest** are identical on both. That is what keeps the split cheap. | |

Rule of thumb for Claude Code: if a file lives under `engines/unreal/`, `docs/README.md`, or the wheel, ask "does this work on a stock Windows machine with PowerShell?" If it lives under `scripts/aws*`, `psplat/study/`, or `docs/lab-notebook.md`, ask "does this run from the Mac?" A change that breaks either answer is a regression.

### 0.2 The pipeline runs wherever the user already is

AWS is an **accelerator for Zach's benchmarks**, never a requirement for anyone. Every CLI subcommand — including `analyze` — runs natively on **Windows, Linux and macOS** through one `Renderer` protocol with three implementations:

| Backend | Platform | Speed | Role |
|---|---|---|---|
| `gsplat_backend` | Linux (AWS) and **native Windows** with an NVIDIA GPU; `pip install psplat[cuda]` pulls a prebuilt gsplat wheel matched to the installed Torch/CUDA (gsplat publishes these); MSVC Build Tools only as a documented fallback if no wheel matches | fast | production `analyze`; benchmarks |
| `msplat_backend` | macOS, Apple Silicon | fast | Zach's local iteration |
| `torch_backend` | any (CUDA / MPS / CPU) — pure PyTorch, tile-sorted alpha accumulation computing per-splat contribution | slow (expect tens of minutes for a 3M-splat scene on CPU; minutes on a GPU) | guarantee-it-runs path for any machine; **correctness oracle** the other two are tested against |
| `null_backend` | any | — | `order/pack/eval/lod/report` on pre-analysed inputs; unit tests |

Backend selection is automatic (`--backend auto`: gsplat if importable + CUDA, else msplat on Mac, else torch) with a one-line notice of what was picked and roughly how long it will take. Results must agree across backends within a documented tolerance (tested in CI on the synthetic fixture).

---

## 1. Landscape (what exists, what's open)

### 1.1 Compression & pruning — the baselines we must beat
- **Importance-score pruning**: LightGaussian (opacity × contribution), Mini-Splatting (view-dependent), PUP 3D-GS (Hessian sensitivity, prunes ~90%), REFINE (rendering-free Hessian importance, 2026), provable coresets (sensitivity sampling with guarantees, 2026), SA-3DGS (learned per-splat importance). None of these weight by *perceptual* visibility — they minimise L2/SSIM on training views.
- **Attribute compression**: quantisation + codebooks (Compact3D, Niedermayr et al.), anchor/hash-grid entropy models (HAC, Scaffold-GS/Octree-GS), image-codec layouts (Self-Organising Gaussians → the lineage that became **SOG**). **PCGS** (AAAI 2025) is progressive compression but rate-driven, not perception-driven.
- **LOD / hierarchy**: Hierarchical 3DGS (Kerbl 2024), Octree-GS, Frequency-Aware GS Decomposition (per-frequency splat groups; showed a 40% FPS win at 4K with gaze-weighting).
- **Surveys** to cite and mine for baselines: *Compression in 3DGS: A Survey of Methods, Trends, and Future Directions* (2025, IEEE), *3DGS.zip* (CGF 2025, living leaderboard at w-m.github.io/3dgs-compression-survey), *SUCCESS-GS* (2025, static+dynamic), and **Splatwizard** (Dec 2025) — a benchmark toolkit for 3DGS compression that we should plug into rather than rebuild.

### 1.2 Perception — the underexplored half
- **Closest prior art**: *Perceptually Guided 3DGS Streaming and Rendering for Mixed Reality* — hierarchical splats, continuous per-frustum LOD, FovVideoVDP as the metric, an MLP predicting an LOD map, 16-person user study. It is **gaze-contingent and runtime**. Our differentiation: **offline, gaze-agnostic (or gaze-prior) bit allocation baked into a static ordering** any engine can consume without a special renderer. Also Fov-GS, VR-Splatting, A3FR (all runtime foveation).
- **Quality assessment**: **3DGS-QA** (AAAI 2026) — 225 degraded reconstructions, 15 objects, MOS scores, and a no-reference metric operating on raw Gaussians (github.com/diaoyn/3DGSQA). **GScomp-QA** (2026) — subjective dataset specifically for *compressed* splats. Both are calibration gold for choosing which objective metric to trust.
- **Metrics available off the shelf**: PSNR/SSIM/LPIPS (standard), ꟻLIP (NVIDIA, perceptual per-pixel error, BSD), **ColorVideoVDP / FovVideoVDP** (Mantiuk lab — spatio-temporal + eccentricity-aware; check licence before shipping as a dependency), DISTS.
- **Vision-science foundations** for the paper's framing: Olshausen & Field sparse coding (Gaussian-derivative-like receptive fields), contrast sensitivity functions (Barten; Mantiuk's castleCSF), contrast masking (Daly VDP), saliency (Itti–Koch; modern DeepGaze).

### 1.3 The key observation that makes this tractable
A splat's **projected footprint in pixels is a spatial-frequency band**. A 2-pixel-wide Gaussian is a high-frequency component; a 40-pixel one is low frequency. That means a CSF can be applied *per splat per view* with almost no extra machinery — the projection pass already computes the 2D covariance. Combined with local contrast (from the rendered image) and a saliency prior, this gives a per-splat **perceptual weight** that can be folded into any existing importance score. This is the thesis of the project.

### 1.4 Formats & engines (so the output actually lands somewhere)
- **PLY** (lossless archive, ~248 B/splat), **SOG** (PlayCanvas, open spec, 15–20× smaller than PLY, WebP-image container; also *Streamed SOG* with spatial-tree LOD), **SPZ v4** (Niantic, ~10× smaller), **glTF + `KHR_gaussian_splatting`** (Khronos release candidate Feb 2026; SPZ sub-extension in progress).
- **Reference tooling**: `splat-transform` (PlayCanvas CLI/library, Node, reads/writes all the above, has `--reorder` and procedural JS generators), **SuperSplat** (browser editor/viewer, streams SOG).
- **Engines — Unreal (primary)**: the open option is XVERSE's **XScene-UEPlugin** (Apache-2.0; Niagara GPU-particle renderer; imports PLY; documented for UE 5.1–5.5 on Windows; automatic LOD; hybrid rendering with normal UE assets). **MLSLabs GaussianSplattingRenderer-UE** uses a custom render pipeline instead of Niagara (claims millions of splats, SOG support arriving; check licence). Closed/paid: Volinga, Luma's UE plugin, VIVE Mars Nova (private beta, PLY+SPZ), Akiya 3D Gaussians Plugin (Fab). **GSplatForUnrealEngine** (sudotman) is a smaller open project with in-editor COLMAP/training. Important property for us: **every Niagara-based renderer already has a particle-count knob**, and our ordered PLY means "first N splats" *is* the budget — so the Unreal sample may be a thin Blueprint/C++ wrapper rather than a renderer. Windows-only in practice, which matches the consumer surface (§0.1); our sample stays Blueprint-only on top of XScene's prebuilt releases so it can be authored on the Mac and needs no build of ours (§3.5, §5.3).
- **Other engines (secondary)**: Unity (official package loads PLY/SPZ; aras-p `UnityGaussianSplatting` MIT), Godot (`godot-gaussian-splatting` / GDGS, v3.3 Jul 2026, compute + raster backends, KHR glTF import), PlayCanvas (native SOG; used for our web viewer).

### 1.5 Trainers
- **gsplat** (Berkeley/NVIDIA, Apache-2.0). Very active: MCMC densification, 3DGUT, sparse rasterisation, **Gaussian-ID / top-contributor-per-pixel rasterisation (Jun 2026)** — directly useful for importance analysis. **PyPI is a year stale (1.1.5, Nov 2024); install from git.** Requires PyTorch ≥ 2.7, CUDA 12.8 / 13.x.
- **msplat** (Apache-2.0, PyPI 1.1.4 Aug 2026). Pure-Metal trainer for Apple Silicon: ~60–80 s for a 7K-iteration Mip-NeRF 360 scene at full res on M4 Max, ~350 FPS render, Python + Swift + C++ CLI. Quality within ~0.5 dB of gsplat at 7K. **This makes the Mac a real experiment box for small scenes, not just a viewer.**
- **Brush** (Apache-2.0, Rust/wgpu) — cross-platform trainer/viewer, runs on Metal and in browser. Good as a viewer and sanity check.
- Avoid as dependencies (licence): Inria reference code (non-commercial), OpenSplat (AGPL), LichtFeld (GPL).

---

## 2. Research plan

### 2.1 Hypotheses
- **H1 (ordering)**: At a fixed splat budget (10%, 25%, 50%), a perceptually weighted ordering (contribution × CSF × masking × saliency) yields higher perceptual quality (ꟻLIP, LPIPS, ColorVideoVDP, human 2AFC) than contribution-only ordering (LightGaussian / PUP / coreset scores), at a small PSNR cost.
- **H2 (allocation)**: Perceptual weights also improve *attribute* bit allocation — quantising SH/scale/rotation more coarsely for low-weight splats — versus uniform SOG/SPZ quantisation at equal file size.
- **H3 (scalability)**: A single ordering is robust across viewing distances: the same file truncated by projected-size threshold at render time degrades gracefully, without per-distance re-optimisation.
- **H4 (dynamic — exploratory only, not a v0.1 claim)**: For performer sequences, temporal masking lets the budget drop further for moving regions at equal perceived quality. No proprietary capture is available and public human datasets are licence-encumbered, so H4 is tested only qualitatively (E7) and never headlines.

### 2.2 Experiments

| ID | Name | Purpose | Data | Compute | Success criterion |
|----|------|---------|------|---------|-------------------|
| **E0** | Reproduce baseline | Establish trust + fixtures | Mip-NeRF 360: garden, bicycle, counter, room | AWS g6e (gsplat MCMC 30K), Mac (msplat 7K) | PSNR/SSIM within 0.3 dB / 0.02 of gsplat published numbers; fixtures saved to S3 |
| **E1** | Importance-score zoo | Implement + compare existing scores as ordering baselines | E0 scenes | AWS | Rate–quality curves (PSNR, SSIM, LPIPS, ꟻLIP) at 5/10/25/50/100% for ≥4 baseline scores; results reproducible from a single `psplat bench` command |
| **E2** | Perceptual weighting | Test H1 | E0 scenes + Tanks&Temples truck/train | AWS | Perceptual ordering dominates ≥3 of 4 baselines on ꟻLIP/LPIPS at 10–25% budget on ≥5 of 6 scenes |
| **E2a** | Ablation | Which perceptual terms matter | same | AWS | Per-term contribution table (CSF only / +masking / +saliency) |
| **E3** | Attribute allocation | Test H2 | same | AWS | At matched bytes vs SOG default and SPZ v4: ꟻLIP/LPIPS improvement ≥ 1 JND-equivalent on ≥4 scenes |
| **E4** | Distance robustness + frame time | Test H3; give the graphics audience a cost axis | same + rendered zoom trajectories | Mac (web viewer) + AWS; Unreal frame time from a Windows TD run | Quality vs projected-size threshold curve; no ordering rank inversions >X% (define X after pilot); **frame-time vs budget curve** in the web viewer and in Unreal (stat unit / Unreal Insights, captured during the TD's run) |
| **E5** | Metric calibration | Pick metrics we trust | 3DGS-QA, GScomp-QA | Mac or AWS | Spearman/PLCC of each metric vs MOS; document which to headline |
| **E6** | Human study (small) | Ground H1 | 3 scenes × 3 budgets × 2 orderings | Mac (study rig) | n≈15, 2AFC with reference, Bradley–Terry scores; preregister design before running |
| **E7** | Dynamic / human performer (exploratory, optional) | Probe H4 | DNA-Rendering or ActorsHQ subset (research licence; never redistributed; never in the demo video) | AWS | Qualitative clip + one temporal-metric curve; appendix material at most |

**Order of execution**: E0 → E1 → E5 (early, so metric choice is settled before E2 headline results) → E2/E2a → E3 → E4 → E6. E7 only if P0–P4 are complete.

### 2.3 Perceptual weight — first formulation (E2 starting point)
For splat *i* and training/eval view *v*:

```
w_i = Σ_v  c_iv · S(f_iv) · M(κ_iv) · A(p_iv) · V_iv
```
- `c_iv` — rendering contribution (accumulated alpha·transmittance; from gsplat top-contributor/ID rasterisation or a custom backward pass).
- `S(f_iv)` — CSF weight; `f_iv` = spatial frequency from projected footprint: `f ≈ (pixels-per-degree) / (2·σ_pixels)` cycles/deg. Start with Barten or castleCSF at 60 ppd, mid luminance.
- `M(κ_iv)` — contrast-masking attenuation from local RMS contrast of the *reference render* in the splat's footprint (Daly-style threshold elevation). Start with a simple power law; ablate.
- `A(p_iv)` — saliency prior at projected position (start: centre-weighted + face/person detector for humans; later DeepGaze). Make it optional/pluggable.
- `V_iv` — view weight. **Uniform over the evaluation camera set for v0.1.** No gaze or viewing-distribution data is available; a gaze prior is the natural follow-up (§9.2) and must not be assumed anywhere in the pipeline. The gaze-agnostic proxy for "where the viewer is" is E4's projected-size threshold, which is a function of camera distance only.

Ordering = sort by `w_i` descending. Progressive cut points every 5%. Attribute allocation (E3): map `w_i` quantiles → SH band count and quantiser step.

All terms are pure functions over per-view arrays; implement as a small pipeline of NumPy/Torch ops so ablations are flag flips.

---

## 3. Tool spec (primary deliverable)

### 3.1 Design principles
1. **Consume standard inputs, emit standard outputs.** In: PLY (any trainer). Out: reordered PLY/SOG/SPZ/glTF-KHR + a tiny JSON manifest. No custom renderer required to benefit.
2. **Analysis is optional-GPU; decode is nothing.** Core analysis needs a renderer (gsplat CUDA on AWS, msplat Metal on Mac). Packing, evaluation and manifest handling run anywhere.
3. **Afternoon test.** A TD **on a stock Windows machine** with a PLY and **Unreal Engine 5** must go `pip install psplat` → `psplat order` → `psplat pack --ply --sog` → import ordered PLY with the Unreal sample → drag the budget slider → done, following the README only, with no GPU required for the psplat steps. The web viewer is the same test with zero install. **Passing is defined as an external Windows TD doing it unassisted, not Zach doing it on the Mac.**
4. **Everything is a CLI subcommand and a Python function.** No notebooks required for the main path.

### 3.2 Package layout

```
psplat/
  pyproject.toml            # hatch or uv; py>=3.12; Apache-2.0
  psplat/
    __init__.py
    io/            # ply.py (read/write, all 62 attrs), sog.py, spz.py (via spz lib or splat-transform shell-out), gltf_khr.py, manifest.py
    render/        # Renderer protocol; gsplat_backend.py (CUDA, Linux+Windows), msplat_backend.py (Metal), torch_backend.py (portable oracle), null_backend.py
    analyze/       # contribution.py, csf.py, masking.py, saliency.py, weights.py (compose), scores_baseline.py (lightgaussian, pup, coreset, opacity)
    order/         # order.py (sort, cut points), allocate.py (attr bit allocation from weights)
    pack/          # pack.py: writes ordered outputs + manifest; lod.py: projected-size thresholds
    eval/          # metrics.py (psnr, ssim, lpips, flip, cvvdp optional), curves.py (rate–quality), report.py (HTML)
    bench/         # scene registry, dataset download, runners, result store (parquet)
    study/         # 2AFC web rig (static HTML + minimal FastAPI), analysis (Bradley–Terry)
    cli.py         # typer
  engines/
    unreal/        # UE5 sample project: PsplatBudget component/Blueprint over XScene (or own thin Niagara DI); reads manifest, sets particle count
    web/           # PlayCanvas-engine viewer: loads SOG, budget slider, A/B split view
    unity/         # (stretch) aras-p UnityGaussianSplatting sample
    godot/         # (stretch) GDGS sample
  scripts/         # aws bootstrap, s3 sync, dataset fetch
  docs/            # README (afternoon path), TD guide, format notes, experiment protocol
  tests/
```

### 3.3 CLI (v0.1 surface)

```
psplat analyze  scene.ply --cameras cams.json --backend gsplat|msplat --views train|all|N
               --terms contribution,csf,masking,saliency --out scene.weights.npz
psplat order    scene.ply scene.weights.npz --out scene.ordered.ply --cuts 5,10,25,50,75,100
psplat pack     scene.ordered.ply --sog|--spz|--gltf|--ply --alloc perceptual|uniform --out dist/
psplat lod      scene.ordered.ply --pixels-per-degree 60 --distances 1,2,4,8 --out lod.json
psplat eval     scene.ordered.ply --cameras cams.json --budgets 5,10,25,50,100
               --metrics psnr,ssim,lpips,flip[,cvvdp] --out results.parquet
psplat bench    --scenes mipnerf360-core --orderings all --out bench/   # runs E1/E2 end to end
psplat report   bench/ --out report.html
psplat study    serve --pairs pairs.json      # local 2AFC rig for E6
psplat viewer   dist/scene.sog                # opens web viewer with budget slider
```

### 3.4 Manifest (`scene.psplat.json`)
```json
{
  "version": 1,
  "source_sha256": "...",
  "count": 3512000,
  "ordering": {"method": "perceptual-v1", "terms": ["contribution","csf","masking"], "ppd": 60},
  "cuts": [{"frac": 0.05, "index": 175600, "bytes_sog": 812345, "flip": 0.11, "lpips": 0.21}, ...],
  "lod": {"projected_sigma_px_min": [0.5, 1.0, 2.0]},
  "provenance": {"tool": "psplat 0.1.0", "backend": "gsplat@<git-sha>", "date": "..."}
}
```
Engines only need `cuts[].index`. Everything else is for reports and reproducibility.

### 3.5 Engine samples (acceptance = the "afternoon test")
- **Unreal Engine 5 (primary)** — `engines/unreal/PsplatSample/`:
  - Target UE **5.5** (XScene's documented ceiling) and verify 5.6; note results in `docs/unreal.md`.
  - Path A (default): depend on **XScene-UEPlugin** (Apache-2.0). Add a `PsplatBudget` Actor Component + Blueprint that reads `scene.psplat.json`, exposes a 0–100% slider / `SetBudgetFraction()` / `SetBudgetSplats()` and drives the Niagara emitter's particle count so only the first N ordered splats render. Also expose "cut points" as an enum so a designer can pick 10/25/50%.
  - Path B (fallback if A's internals resist): our own minimal Niagara Data Interface / GPU buffer upload of the ordered PLY (positions, scales, quats, colour, opacity, SH0 only), rendered with a standard splat sprite material. Fewer features, fully ours, Apache-2.0.
  - Deliver: sample map with two actors (full vs budgeted) side by side, a UMG slider, `stat unit` overlay, and a 30-second Sequencer shot for the video. Document the "drop your own PLY in" step in ≤10 lines.
  - **Distance LOD**: optional Blueprint that maps camera distance → budget via `lod.json` (E4), so the TD sees the scaling story without touching code.
  - **Blueprint-only, nothing of ours to compile.** Path A ships as a UE 5.5 sample project whose only plugin dependency is XScene's **own prebuilt release zip** (downloaded by the TD or vendored under its Apache-2.0 licence). Our contribution is Blueprints + a UMG widget + data assets, which are platform-neutral, so the sample is **authored on the Mac** and needs no Windows build step. Distribution: a GitHub Release zip of the sample project that opens from a fresh Launcher install of UE 5.5 on Windows.
  - **Validation**: on an external TD's own Windows machine (the afternoon test). No Windows cloud instance is used anywhere in this project.
  - **Path B caveat**: a C++ Niagara Data Interface would require a Win64 build somewhere — a borrowed studio machine or a self-hosted GitHub Actions runner. **DECIDE (deferred)**: this conversation is triggered only if Path A blows its two-day time-box; until then, Path A is the plan.
- **`psplat` Python package on Windows**: pure-Python wheel (`py3-none-any`) so `pip install psplat` works with no compiler; `order`, `pack`, `eval` (PSNR/SSIM/LPIPS/ꟻLIP on CPU or CUDA if present), `lod`, `viewer`, `report` run with `null_backend`; **`analyze` runs locally too** — `pip install psplat[cuda]` for gsplat on an NVIDIA card, otherwise `torch_backend` with a time estimate printed up front. `splat-transform` shell-outs use `npx` with a Node check and a PowerShell-friendly error.
- **Web (PlayCanvas engine, MIT)**: load ordered SOG; budget slider selects `cuts[k].index` or a raw count; split-screen A/B vs full; FPS + byte readout. Zero-install path; also used for the E6 study rig.
- **Unity / Godot (stretch)**: only after Unreal + Web pass the afternoon test with an external tester.

### 3.6 Non-goals for v0.1
Training. Dynamic splats (E7 is exploratory only; no capture data). A custom renderer. **Gaze tracking / gaze-contingent LOD** (deliberate — see §9.2). Mobile. Unreal relighting or collision (use what the host plugin provides).

---

## 4. Data pipeline (secondary deliverable)

### 4.1 Scene registry (`psplat/bench/scenes.yaml`)
| Set | Scenes | Why | Licence note |
|-----|--------|-----|--------------|
| Mip-NeRF 360 core | garden, bicycle, counter, room | matches msplat + gsplat published tables → cross-platform sanity | research use; fetch script, don't redistribute |
| Mip-NeRF 360 rest | bonsai, kitchen, stump, flowers, treehill | full benchmark | same |
| Tanks & Temples | truck, train | outdoor, large | same |
| Deep Blending | drjohnson, playroom | indoor | same |
| Synthetic NeRF (Blender) | lego, ship | ground-truth camera, floaters-free | CC |
| 3DGS-QA / GScomp-QA | as provided | metric calibration (E5) | check each |
| Human (E7, optional) | DNA-Rendering part subset or ActorsHQ 1–2 sequences | exploratory only; no in-house capture exists | restrictive research licence; never shipped, never in video |
| Own captures | 2–3 phone captures (an object, a room, a consenting person if convenient) | demo footage we fully own; the "in-house asset" stand-in for the Unreal demo | ours |

### 4.2 Stages (each a `bench` function; each idempotent; each writes a provenance JSON)
1. `fetch` → `s3://<bucket>/raw/<set>/<scene>/`
2. `pose` → COLMAP (or GLOMAP for speed) → `sparse/0`. Skip for datasets shipping poses.
3. `train` → gsplat MCMC `cap_max=1e6` (fast) and `3e6` (quality); 30K iters; every-8th-view eval split (standard). Save PLY + cameras JSON + eval renders. Mac path: msplat 7K/30K for core scenes.
4. `analyze / order / pack / eval` → per ordering × budget.
5. `collect` → parquet table: `scene, trainer, ordering, terms, budget_frac, bytes, format, psnr, ssim, lpips, flip, cvvdp, fps_web, wall_time, git_sha`.
6. `report` → HTML with rate–quality curves and image strips (full / 25% / 10% for each ordering, plus ꟻLIP heatmaps).

Also integrate with **Splatwizard** if its API is stable: register `psplat` orderings as methods so numbers land on a comparable leaderboard.

### 4.3 Storage layout
```
s3://psplat-<acct>/raw/            datasets (immutable)
s3://psplat-<acct>/trained/<scene>/<trainer>/<sha>/   ply + cams + renders
s3://psplat-<acct>/runs/<date>-<tag>/                  weights, ordered, packed, results.parquet
s3://psplat-<acct>/study/                              stimuli + responses (anonymised)
```
Local mirror on the Mac: `~/psplat-data/` with `aws s3 sync --exclude renders/*` by default.

---

## 5. Environment spec

### 5.1 AWS (processing)
- **Instance**: `g6e.2xlarge` (1× L40S 48 GB, 8 vCPU, 64 GB) as the default trainer/analyser; `g6.xlarge` (L4 24 GB) for cheap CI/smoke runs. Use **Spot** for batch benches with checkpointing to S3; On-Demand for interactive sessions. **DECIDE**: region `us-east-1` unless data-residency says otherwise.
- **OS/AMI**: Amazon Linux 2023 — use the **AWS Deep Learning Base AMI (Amazon Linux 2023)** with a single CUDA stack, x86_64. Resolve the current AMI ID at provision time via the public SSM parameter under `/aws/service/deeplearning/ami/x86_64/...amazon-linux-2023/latest/ami-id` (Claude Code: look up the exact parameter name in the DLAMI docs rather than hard-coding; the docs list an SSM query per AMI flavour). Do not pin a kernel; the DLAMI version-locks it.
- **Storage**: 500 GB gp3 root+data (raise IOPS for COLMAP), S3 bucket above, lifecycle rule to IA after 30 days for `runs/`.
- **Access**: SSM Session Manager (no public SSH); VS Code Remote-SSH over SSM proxy; Claude Code runs *on the instance* for heavy tasks, on the Mac for everything else. IAM instance role with S3 RW to the bucket only.
- **Bootstrap** (`scripts/aws_bootstrap.sh`, idempotent):
  ```bash
  sudo dnf -y install git gcc gcc-c++ cmake ninja-build ffmpeg-free nodejs20 npm
  curl -LsSf https://astral.sh/uv/install.sh | sh
  uv python install 3.12
  uv venv ~/.venvs/psplat && source ~/.venvs/psplat/bin/activate
  uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128   # match CUDA on the AMI
  uv pip install "git+https://github.com/nerfstudio-project/gsplat.git"   # PyPI is stale
  uv pip install -e ".[dev,aws]"
  npm i -g @playcanvas/splat-transform
  # COLMAP/GLOMAP: build from source or use conda-forge; record chosen route in docs/env.md
  nvidia-smi && python -c "import gsplat, torch; print(gsplat.__version__, torch.version.cuda)"
  ```
- **Cost guardrails**: tag everything `project=psplat`; budget alarm; `scripts/aws_stop_idle.sh` cron that stops the instance after 45 min without a running `psplat` process or SSM session. Expect roughly 1–2 GPU-hours per scene for full train+analyse+eval.

### 5.2 Mac (operator surface: control, viewing, small experiments, user study)
- **M4 Max MacBook Pro, macOS 14+**, Xcode CLT, Homebrew, `uv`, Node 20+, AWS CLI v2 + Session Manager plugin, **Unreal Engine 5.5 (Apple Silicon) for authoring the Blueprint-only sample**.
- **Role**: this is where Claude Code and Zach live. All AWS provisioning, S3 sync, benchmark orchestration, report generation, paper figures, the E6 study rig, and Unreal sample authoring run from here. Heavy compute is dispatched to §5.1 when wanted, never required.
- **Trainer**: `pip install msplat` (Metal; Python 3.12/3.13 wheels). Use for E0 core scenes and for rapid iteration on the analysis pipeline via the `msplat` render backend (`trainer.render_from_pose`). Note msplat exposes rendering, not per-splat contribution — the `msplat_backend` may need a contribution approximation (render with and without a splat subset, or a Python-side projected-footprint estimate) until upstream exposes it. Log the gap in `docs/backends.md`.
- **Viewers**: Brush (Apache-2.0) for quick PLY checks; **SuperSplat** in Chrome for SOG/SPZ and format sanity; our own web viewer; MetalSplatter (MIT) if a native visionOS/macOS demo is wanted later.
- **Unreal on the Mac** is where the sample project is authored (Blueprints, UMG, data assets, sample map). XScene's Mac behaviour is unverified, so if it fails to render on Metal, author against a stub Niagara emitter with the same parameter names and rely on the Windows TD for the real render. Frame-time numbers for E4 come from the web viewer and from the TD's Windows run (§5.3), not from the Mac.
- **Metrics**: LPIPS, ꟻLIP, SSIM all run on MPS or CPU. ColorVideoVDP: try MPS; fall back to CPU or run on AWS.
- **Study rig (E6)**: `psplat study serve` — local static site, full-screen, calibrated distance (tape measure; 60 ppd target), stimuli as pre-rendered PNG/MP4 pairs, responses to a local SQLite then synced to S3.

### 5.3 Windows (consumer platform — the TD's own machine)
No Windows machine is provisioned by this project, on AWS or otherwise. Windows support is delivered by making the software run there natively and proving it in CI:
- **`windows-latest` GitHub Actions job (required to merge)**: clean venv, `pip install` the wheel, run `order/pack/eval/lod/report` on the synthetic fixture with `null_backend` and `analyze` with `torch_backend` on CPU; execute the README's PowerShell blocks literally.
- **gsplat on Windows**: document the tested Torch/CUDA/gsplat wheel triple in `docs/windows.md` and pin it in the `[cuda]` extra; include the MSVC Build Tools fallback and a "what to do if `nvcc` isn't found" section. Ask one Windows-using contact to smoke-test `psplat[cuda]` in P1.
- **Unreal acceptance** happens on an external TD's Windows workstation running the afternoon test unassisted (P3 exit). Provide them a one-page checklist and a screenshot/`stat unit` capture script so their run also yields the Windows frame-time curve for E4.
- If Path B ever becomes necessary (§3.5), revisit: borrowed machine or self-hosted runner — not a cloud Windows image.

### 5.4 Dev conventions
- `uv` for envs, `ruff` + `mypy --strict` on `psplat/`, `pytest` with a tiny synthetic scene fixture (200 splats) so tests run without GPU.
- **CI matrix**: `macos-14` (arm64, CPU tests + msplat import smoke), **`windows-latest` (CPU tests, wheel install from a clean venv, `psplat order/pack/eval` and `analyze --backend torch` on the fixture, PowerShell README commands executed literally)**, and a manual/self-hosted CUDA job on the AWS Linux box. The Windows job is required to merge.
- **PowerShell parity**: every user-facing command in `docs/` appears in both bash and PowerShell fenced blocks; a doc test extracts and runs the PowerShell blocks on the Windows runner. Paths in code go through `pathlib`; no hard-coded `/`.
- Pure-Python wheel for `psplat` (no compiled extensions in the core package); optional extras `[cuda]` (gsplat prebuilt wheel), `[msplat]`, `[cvvdp]`.
- Backends behind a `Renderer` protocol (§0.2); `torch_backend` is the oracle — CI asserts gsplat/msplat contributions match it within tolerance on the fixture; `null_backend` for pure I/O tests.
- Every experiment run writes `run.json` (git sha, args, hardware, wall time). No result without provenance.
- Licences: Apache-2.0 for code; MIT-compatible deps only in the runtime path. Document each third-party licence in `THIRD_PARTY.md`. GPL/AGPL tools may be *invoked* as external processes by users but are never imported or vendored.

---

## 6. Presentation (tertiary deliverable)

- **Video (2–3 min)**: (a) the MP3 analogy in one shot; (b) the same scene at 10% budget — contribution-only vs perceptual ordering, split screen, slow orbit; (c) **the Unreal sample: import ordered PLY, drag the slider, `stat unit` visible, ~30 s Sequencer shot**; (d) the web viewer for the zero-install story; (e) one rate–quality chart with frame time on a second axis. Record Unreal via Sequencer/Movie Render Queue, everything else with OBS on the Mac; offline renders from AWS.
- **Paper sketch (4–6 pages), graphics venue** — target one of **I3D, HPG, EGSR, or SIGGRAPH Asia Technical Communications**; a real-time-rendering audience. Consequences for content: headline **ꟻLIP and the 2AFC study**, treat LPIPS as supporting and PSNR/SSIM as a table in the appendix; always pair quality with **cost** (splat count, bytes, Unreal frame time); frame the contribution as *perceptual LOD/budget allocation for an existing primitive*, not as a codec paper; include an engine-integration section (which vision venues would cut). Sections: Intro (perception gap in splat LOD), Method (perceptual weight, ordering, allocation), Results (E1–E6 with cost axes), Engine integration (Unreal + web), Limitations & future work (static only, gaze-agnostic by design, metric validity), Release. Generate figures from `psplat report` so they regenerate on rerun.
- **README** doubles as the TD guide: the 6-command afternoon path, GIFs, "what to expect" table of budget vs quality.

---

## 7. Milestones

| Phase | Weeks | Exit criteria |
|-------|-------|---------------|
| P0 Bootstrap | 1 | AWS Linux + Mac envs reproducible from scripts; **`torch_backend` implemented and passing as oracle**; E0 numbers match; CI green on macOS **and Windows** (incl. `analyze --backend torch`) with the synthetic fixture |
| P1 Tools alpha | 2–3 | `analyze/order/pack/eval` work end-to-end on garden; web viewer slider works; **Blueprint-only Unreal sample authored on the Mac; a Windows contact confirms it opens and the slider drives particle count**; `psplat[cuda]` smoke-tested on a Windows NVIDIA machine; E1 baseline zoo done |
| P2 Perception | 3–4 | E5 metric calibration report; E2 + E2a results; decision on headline metric (expected: ꟻLIP) |
| P3 Allocation + LOD + acceptance | 2 | E3, E4 (web-viewer frame time + Windows frame time captured during the TD run); SOG/SPZ/glTF validated in SuperSplat; sample-project Release zip published; **an external Windows-using TD completes the afternoon test unassisted, running `analyze` locally** |
| P4 Study + write-up | 2–3 | E6 run and analysed; video with Unreal segment (Movie Render Queue on the Mac if XScene renders there, else screen capture supplied by the Windows TD); paper sketch; v0.1 tagged |
| P5 Stretch | open | Unity/Godot samples; Splatwizard integration; E7 exploratory; gaze-prior design note (§9.2) |

---

## 8. Instructions to Claude Code

1. **Start with P0.** Create the repo skeleton in §3.2, `pyproject.toml` (pure-Python wheel), CI (GitHub Actions: `macos-14` + `windows-latest` for CPU tests; CUDA job manual/self-hosted), `scripts/aws_bootstrap.sh` (Linux GPU), `docs/env.md`, `docs/windows.md`. Verify gsplat import on AWS Linux, msplat import on Mac, and `pip install psplat` + `analyze --backend torch` in a clean venv on the Windows runner; record versions.
1a. **Write `torch_backend` in P0**, before `gsplat_backend`. It is small, it is the oracle every other backend is tested against, and it is what makes the "runs anywhere" promise true.
2. **Implement `io/ply.py` first** with round-trip tests on a real PLY (garden). Then `render/torch_backend.py` implementing `contributions(scene, cameras) -> (N, V)` by tile-sorted alpha accumulation in pure PyTorch (correctness first, vectorise second). Then `render/gsplat_backend.py` using gsplat's Gaussian-ID / top-contributor rasterisation if available in the installed version, else an accumulated-alpha backward pass; assert it matches `torch_backend` on the fixture. Benchmark gsplat; it is the hot path.
3. **Implement baseline scores** (`scores_baseline.py`) faithfully from the papers; unit-test against toy cases; then `csf.py`, `masking.py`, `weights.py`.
4. **`order` + `pack` + web viewer + Unreal sample** before any perceptual experiments — the afternoon test is the product. For Unreal: clone XScene-UEPlugin as a submodule pinned to a tag, get a plain PLY rendering in a fresh UE 5.5 project **on the Mac** (or, if XScene won't render on Metal, confirm the emitter's parameter surface from its source and author against it), then find the spawn/particle-count parameter and wire the Blueprint-only `PsplatBudget` to it. Hand the project to a Windows contact for a render check as early as P1. If Path A takes more than two working days, **stop and raise Path B with Zach** (§3.5 — it needs a Win64 build somewhere) rather than starting it. Keep all Unreal assets text-serialisable where UE allows (Blueprints are binary — document them with screenshots + a short C++ equivalent).
4a. **Apply the §0.1 test to every PR touching `engines/unreal/`, `docs/README.md`, or packaging**: would a stock Windows TD with PowerShell and no Mac get through this? If any step says "on your Mac…" it is wrong.
5. **`eval` + `bench` + `report`**, then run E1 and E5. Only then E2.
6. Keep a running `docs/lab-notebook.md`: date, what ran, result, next. Append, don't rewrite.
7. **Ask before**: spending >$50 in a single AWS action; running any human study; adding a dependency with a non-permissive licence; changing the manifest schema after P1.
8. Prefer boring tech: parquet + DuckDB over an experiment-tracking service; static HTML over dashboards; `typer` over custom arg parsing.
9. When a paper's method is ambiguous, implement the simplest reading, name it `<method>-simple`, and note the ambiguity in `docs/baselines.md`.

---

## 9. Decisions and follow-up work

### 9.1 Decisions (resolved 2026-09-13)
| # | Question | Decision | Effect on this document |
|---|----------|----------|-------------------------|
| 1 | Paper audience | **Graphics** (I3D / HPG / EGSR / SIGGRAPH Asia TC), not vision/compression | ꟻLIP + user study headline; cost axes everywhere; engine-integration section; §6 |
| 2 | Gaze data in scope? | **No.** | `V_iv` uniform; gaze-contingent LOD is a non-goal; recorded as the primary follow-up in §9.2; stated limitation in the paper |
| 3 | Proprietary capture data? | **None available.** | Dynamic/human work (H4, E7) is exploratory-only and never headlines or appears in the video; demo uses public scenes + our own phone captures |
| 4 | Engine | **Unreal Engine 5** | Unreal is the primary sample and the afternoon-test target; web viewer is the zero-install path; Unity/Godot are stretch |
| 5 | Platforms | **Two surfaces: Windows for TDs (consume), Mac for Zach (operate)** | §0.1 governs; Windows CI required to merge; PowerShell parity in docs; pure-Python wheel; P3 exit = external Windows TD |
| 6 | Where does processing run? | **Wherever the user is; never require cloud; never a Windows AWS image** | §0.2 three-tier backend with portable `torch_backend`; `psplat[cuda]` on native Windows; Unreal sample Blueprint-only over XScene releases, authored on Mac, validated by a Windows TD; Path B deferred |

### 9.2 Follow-up work (explicitly deferred; design notes welcome, no implementation in v0.1)
1. **Gaze / viewing-distribution prior.** The weight formulation (§2.3) has a slot for it (`V_iv` and the eccentricity dimension of `S`). The natural next paper: measure or model where viewers look in these scenes, feed FovVideoVDP-style eccentricity sensitivity into the ordering, and compare gaze-agnostic vs gaze-prior orderings. This needs an eye tracker or a viewing study; flagged as the best follow-up question raised during planning.
2. **Dynamic / performer content** with temporal masking, contingent on obtaining licensable multi-view capture.
3. **Runtime re-ordering** (view-dependent budget) as an Unreal-native feature, once the static ordering is proven.
4. **Attribute-level progressive streams in SOG/glTF-KHR** (streaming quantiser refinement, not just splat count), possibly as a proposed extension.

---

## 10. References (verify links at implementation time)

**Core**
- Kerbl et al., *3D Gaussian Splatting for Real-Time Radiance Field Rendering*, TOG 2023.
- Ye et al., *gsplat: An Open-Source Library for Gaussian Splatting*, JMLR 2025 — github.com/nerfstudio-project/gsplat
- Hatout, *msplat* — pypi.org/project/msplat, github.com/rayanht/msplat
- Brussee, *Brush* — github.com/ArthurBrussee/brush

**Compression / pruning / LOD**
- Fan et al., *LightGaussian*, NeurIPS 2024 · Fang & Wang, *Mini-Splatting*, ECCV 2024 · Hanson et al., *PUP 3D-GS*, arXiv 2406.10219 · Niedermayr et al., *Compressed 3DGS*, CVPR 2024 · Chen et al., *HAC*, ECCV 2024; *PCGS*, AAAI 2025 (2503.08511) · Morgenstern et al., *Self-Organizing Gaussians* · Kerbl et al., *Hierarchical 3DGS*, SIGGRAPH 2024 · Lu et al., *Scaffold-GS*; Ren et al., *Octree-GS* · *Frequency-Aware Gaussian Splatting Decomposition*, arXiv 2503.21226 · *Provable Pruning via Coresets*, arXiv 2607.02721 · *REFINE*, arXiv 2606.09074 · *SA-3DGS*, arXiv 2508.03017 · *SpeeDe3DGS*, arXiv 2506.07917
- Surveys: *Compression in 3DGS: Survey of Methods, Trends, Future Directions*, arXiv 2502.19457 · Bagdasarian et al., *3DGS.zip*, CGF 2025 (leaderboard: w-m.github.io/3dgs-compression-survey) · *SUCCESS-GS*, arXiv 2512.07197 · *Splatwizard: A Benchmark Toolkit for 3DGS Compression*, arXiv 2512.24742 · *A Survey on 3D Gaussian Splatting*, ACM CSUR 2026

**Perception / quality**
- Wan et al., *Perceptual Quality Assessment of 3DGS: A Subjective Dataset and Prediction Metric* (3DGS-QA), AAAI 2026, arXiv 2511.08032, github.com/diaoyn/3DGSQA · *GScomp-QA*, arXiv 2605.26880 · *Perceptually Guided 3DGS Streaming and Rendering for Mixed Reality* (FovVideoVDP-driven continuous LOD) · *Fov-GS* · *VR-Splatting* · *A3FR*, ICS 2025 · *Adaptive 3DGS Video Streaming: Saliency-Aware Tiling*, arXiv 2507.14454 · Surace et al., *Gaze-Contingent Perceptual LOD Prediction*
- Metrics: Andersson et al., *ꟻLIP* (NVIDIA) · Mantiuk et al., *FovVideoVDP* (2021), *ColorVideoVDP* (2024) · Zhang et al., *LPIPS* · Ding et al., *DISTS*
- Foundations: Olshausen & Field 1996; Daly 1993 (VDP); Barten 1999 (CSF); Itti & Koch 1998

**Formats / engines**
- PlayCanvas SOG spec — developer.playcanvas.com/user-manual/gaussian-splatting/formats/sog · splat-transform — github.com/playcanvas/splat-transform · SuperSplat — superspl.at
- Niantic SPZ — github.com/nianticlabs/spz
- Khronos `KHR_gaussian_splatting` (RC, Feb 2026) — github.com/KhronosGroup/glTF/tree/main/extensions/2.0/Khronos/KHR_gaussian_splatting
- **Unreal**: XVERSE XScene-UEPlugin (Apache-2.0, Niagara) — github.com/xverse-engine/XScene-UEPlugin · MLSLabs GaussianSplattingRenderer-UE — github.com/mlslabs/MLSLabsGaussianSplattingRenderer-UE · GSplatForUnrealEngine — github.com/sudotman/GSplatForUnrealEngine · Volinga, Luma UE plugin, VIVE Mars Nova, Akiya 3D Gaussians (closed/paid; reference only)
- aras-p UnityGaussianSplatting (MIT) · godot-gaussian-splatting / GDGS — github.com/ReconWorldLab/godot-gaussian-splatting · MetalSplatter — github.com/scier/MetalSplatter

**Data**
- Mip-NeRF 360 (Barron et al. 2022) · Tanks & Temples · Deep Blending · NeRF-Synthetic · DNA-Rendering (ICCV 2023; 60 cams, restrictive licence) · ActorsHQ / HumanRF

**Infra**
- AWS Deep Learning AMI docs (Amazon Linux 2023 Base with single CUDA; SSM parameter lookup) — docs.aws.amazon.com/dlami
