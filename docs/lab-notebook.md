# psplat lab notebook

Append-only. One entry per working session or experiment run: date, what ran,
result, next. Newest at the bottom.

---

## 2026-09-13 — planning

**What ran:** No code. Read handoff v0.4; verified external dependencies from
the remote (CPU-only) container; wrote `docs/plan.md`.

**Findings:**

- gsplat `main` (1.6.0, unreleased on PyPI) has Gaussian-ID / top-contributor
  rasterisation and publishes Linux + Windows wheels for specific
  Python/Torch/CUDA triples. PyPI is at 1.5.3.
- msplat 1.1.4 (2026-08-25): Python 3.12/3.13, macOS 14+ arm64;
  `GaussianTrainer.render_from_pose(cam2world)`; no per-splat contribution API.
- XScene-UEPlugin v1.1.5.1 supports UE 5.5; no mention of 5.6.
- splat-transform 3.4.2: has `--morton-order`, `--decimate`; no index-range
  filter; no statement that SOG preserves input order. Logged as R1, spike P1.0.
- Container: Python 3.11 default (3.13 present, uv can fetch 3.12), Node 22,
  no GPU, no torch.

**Result:** `docs/plan.md` committed with P0–P5 breakdown, schemas to fix
early, risks R1–R12, decisions D1–D7.

**Next:** P0.1 skeleton + packaging + CI scaffold. Zach: answer D5 (Windows
contact / external TD) and D6 (mirror plan into GitHub issues?).

---

## 2026-09-20 — plan mirrored to GitHub issues

**What ran:** No code. Mirrored `docs/plan.md` into GitHub issues and closed out
the open decisions.

**Result:**

- Six phase epics (#1 P0, #2 P1, #3 P2, #4 P3, #5 P4, #6 P5) with 46 work items
  as sub-issues (#7–#52), each carrying the same acceptance criteria as its plan
  row. Issue types Task and Feature; no milestones, since the repo has none.
- Plan rows and phase headings now link to their issues.
- D1–D4 and D6 confirmed as planned. **D5 resolved: there are no external
  Windows testers to coordinate with.** The afternoon test remains the design
  bar, but acceptance for P1.8, P1.10, P3.4 and P3.6 is now Windows CI plus a
  manual Windows run Zach drives, with anything unverified labelled unverified
  rather than assumed. R11 rewritten accordingly.
- D7 (does SOG preserve splat order?) stays open until the P1.0 spike.

**Next:** P0.1 — repo skeleton, packaging, CI scaffold (#7).

---

## 2026-09-27 — P0 built

**What ran:** P0.1–P0.11 implemented in the remote container (Linux, 4 CPU
threads, no GPU) and pushed; CI on GitHub Actions. Commits on
`claude/upbeat-knuth-u1xp0b`, one per item.

**Results:**

- 117 tests pass locally and on ubuntu-latest, macos-14 and windows-latest.
- The Windows TD job, under stock Windows PowerShell 5.1 with a plain venv
  and pip, installs the built wheel and runs `psplat cameras` and
  `psplat analyze --backend torch` on the fixture, then every PowerShell
  block in the README and `docs/windows.md` literally. A self-test proves
  the doc runner fails on a failing native command, a failing cmdlet and an
  unknown command, and not on stderr output alone.
- The torch oracle matches a literal per-pixel transcription of gsplat's
  CUDA forward loop to rtol 1e-9 in float64, on translucent and on dense
  opaque random scenes; five injected bugs were each caught. Projection and
  SH match gsplat's own pure-PyTorch reference functions.
- PLY I/O agrees bit for bit with PlayCanvas splat-transform 3.4.2 in both
  directions; splat-transform keeps splat order PLY to PLY.
- Synthetic fixture oracles all hold: off-screen splats invisible; the
  occluded target contributes exactly 0 in view_01 while in frame, 2.08 in
  view_05; overlap pair front 20.2 > back 9.1 > 0.

| Measurement (this container) | Value |
|---|---|
| Fixture analyse, 8 views, CPU | 0.05 s |
| PLY parse, 3M splats, warm cache | about 1.9 s (cold adds disk time; 250 MB/s here) |
| torch backend, 300k splats, 1237×822, opaque surface | 3.2 s per view |
| torch backend, same, translucent clutter | 5.8 s per view |
| estimate_seconds vs actual, 300k splats, 3 views | 29 s vs 22 s |

**Findings:**

- gsplat 1.5.3 caps alpha at 0.999, not the Inria 0.99, and stops a pixel
  when transmittance would reach ≤ 1e-4. Read from its source release; the
  oracle follows gsplat, with the Inria cap one argument away.
- gsplat trains in a normalised world frame, so its PLY does not line up
  with raw COLMAP poses. `bench train` writes cameras through gsplat's own
  parser to match.
- Chunk size matters on CPU: 256 splats per step was 4× faster than 4096 on
  an opaque scene, because whole tiles stop early.
- CPU analysis of a 3M-splat scene over ~160 views extrapolates to hours,
  not the tens of minutes the handoff hoped for. Logged as R13, issue #53.
- GitHub's macOS runners report MPS available, then fail every allocation.
  Automatic device choice now probes and falls back.
- The idle-stop script's process check matched its own path ("psplat") and
  would never have stopped the instance; caught by a dry run, fixed.
- Unreachable from the dev container: download.pytorch.org,
  docs.aws.amazon.com, docs.gsplat.studio, Hugging Face. So the DLAMI SSM
  parameter is discovered at run time, and E0 reference numbers are left
  empty rather than recalled.
- `psplat` is currently unclaimed on PyPI.

**Not done, and why:**

- P0.3's round trip on a real garden PLY needs a trained scene (Mac).
- P0.10 has not been run against AWS; P0.11's E0 needs the GPU box and the
  published reference numbers.
- Branch protection (Windows jobs required) is a repository setting for Zach.

**Next:** Zach runs P0.10 and E0 (runbook in `docs/env.md`). P1 starts with
the P1.0 SOG-ordering spike (#19) and P1.1 baseline scores (#20).

---

## 2026-11-07 — P1.0 spike: does SOG keep splat order?

**What ran:** `scripts/spikes/order_preservation.py` against PlayCanvas
splat-transform 3.10.3 (3.4.2 in P0), at 4,096 and 100,000 splats, plus SOG
size experiments at 100k and 400k on spatially coherent synthetic scenes.
Splats sit on an off-centre jittered grid in random order, so each splat's
grid cell identifies it after lossy packing and any re-sorting shows up.

**Results:**

- Keep order: PLY, SPZ v4, glTF. Reorder completely (Spearman ≈ 0): SOG
  from the command line (bundled and unbundled), compressed PLY, Streamed
  SOG. The source agrees: the SOG writer Morton-sorts "unless a
  caller-supplied order is used", via the library's `writeSog({indices})`.
- `writeSog` with identity `indices` keeps order exactly; with a banded
  order (Morton within cut-point bands) every cut point is an exact prefix.
- Size cost against splat-transform's Morton SOG, coherent synthetic scenes:

| Layout | 100k | 400k |
|---|---|---|
| SOG, 6–7 bands | 1.14 | 1.11 |
| SOG, 20 bands (5%) | 1.26 | 1.23 |
| SOG, 100 bands (1%) | 1.42 | 1.39 |
| SOG, exact psplat order | 1.59 | 1.74 |
| SPZ, exact psplat order | 1.59 | 1.72 |

- SOG with SH degree ≥ 1 needs a GPU: no WebGPU adapter means an outright
  failure unless `-g cpu`, and on CPU 32k splats took 4.7 s but 100k ran
  past 25 min (palette up to 64K entries). Logged as R14.

**Mistake caught:** the first harness used a centred, symmetric grid, so
the glTF writer's PLY-to-glTF axis flip (x and y negated) looked like a
scramble. The grid is now off-centre, so a flip shows as unmatched splats,
and glTF is read back in PLY space; it keeps order exactly.

**Decision:** D7 in `docs/plan.md`; write-up in `docs/formats.md`. SOG via
the library with banded indices; PLY/SPZ/glTF for arbitrary truncation;
compressed PLY and Streamed SOG not offered as ordered outputs; version pinned
and asserted by `tests/test_interop_splat_transform.py`.

**Caveat:** size ratios are from synthetic scenes; re-measure on garden in
P1.3 before quoting them.

**Next:** P1.1 baseline scores (#20) and P1.2 order + manifest (#21).

**CI note, same day:** GitHub no longer runs the `macos-14` image ("The
configuration 'macos-14-arm64-us-default' is not supported"); its jobs
queued until cancelled, on every branch. CI now uses `macos-15`, still Apple
Silicon, so the msplat import check is unchanged.

---

## 2026-11-08 — P1.1 baseline scores, P1.2 order and manifest

**What ran:** P1.1 and P1.2 implemented in the remote container (Linux,
4 CPU threads, no GPU), as stacked PRs: P1.1 into `main`, P1.2 on top.

**Results:**

- Six baselines: `opacity`, `contribution`, `lightgaussian`, `pup-simple`,
  `coreset-simple` and `minisplatting-simple`. `docs/baselines.md` records,
  per paper, what is ambiguous and the reading taken.
- The torch backend gathers three new per-splat statistics, summed over
  views: transmittance over composited pixels, top-contributor pixel counts,
  and a 6×6 Fisher block over mean and log-scale. All three agree with a
  literal per-pixel loop. The Fisher also matches central finite
  differences to about 2e-5 relative. Injected bugs were each caught. One
  of them, a wrong gradient at capped alpha, survived the random scenes and
  needed a dedicated test.
- Cost on a CPU: `hits` and `top` add 10–40% to a view's time; `fisher`
  makes it 2–3 times as long.
- `psplat order` writes the ordered PLY and `scene.psplat.json`. The
  manifest is validated against a JSON Schema on write and on read. Each
  cut's index uses the banded SOG layout's band-edge rule. Windows TD CI
  runs `order` from the installed wheel.

**Findings:**

- **The plan's acceptance example, "a duplicated splat gets half the
  score", needed restating.** An exact copy doubles the alpha, so each copy
  keeps nearly its whole score. The test that means something is a split
  whose opacities compose to the original's. At o = 0.3, each copy gets
  0.54 / 0.49 of the contribution and 0.47 / 0.45 of the LightGaussian
  score. The PUP Fisher drops to about a quarter instead. Not exactly half,
  because:
  - the split holds only to first order in alpha;
  - the faint copies each lose more of their tails to the 1/255 threshold.
- LightGaussian's Eq. 1 includes transmittance but its Algorithm 2 does not.
  psplat follows Eq. 1; the released code was not checked.
- numpy's quicksort happened to keep tie order on small inputs, so a test
  of tie-breaking needed 200k tied scores to catch an unstable sort.

**Next:** P1.3 `pack` (#22), with the banded SOG writer from the spike.

---

## 2026-11-08 (later) — P1.3 pack

**What ran:** #55 (P1.1) and #56 (P1.2) merged; P1.3 `psplat pack`
implemented in the remote container.

**Results:**

- **What pack writes:** the ordered PLY, SOG (banded order), SPZ v4 and
  glTF `.glb`, plus the manifest with `bytes_<format>` for every cut.
  PLY prefix sizes are computed exactly. For the other formats, each cut's
  prefix is packed on its own and measured.
- **Interop checks:** packing a real grid scene through splat-transform
  3.10.3 and reading every output back confirms:
  - SPZ and glTF are in exact order;
  - the SOG's texels are in exactly the banded order, so each cut's leading
    texels are the same splats as the PLY truncated there;
  - SOG keeps SH at degree 1 and drops it when asked.
- **Without Node.js**, pack writes the PLY, names what it skipped and why,
  and exits 3. Windows CI checks this with Node removed from PATH, then
  packs every format with it.
- **Fixture timing:** 200 splats, all four formats with cut sizes, took
  about 10 s on this CPU, most of it SOG's SH clustering.

**Findings:**

- splat-transform's WebGPU device code lives in its CLI bundle and is not
  exported from the library. Driving the GPU from `write_sog.mjs` would mean
  copying that code, its DOM shims and the PlayCanvas engine version it
  bundles. It could not be verified anywhere available (no WebGPU adapter
  in the container or on GitHub runners). So SOG keeps SH only up to
  32,768 splats by default, and drops it above that with a notice. R14
  stays open until a GPU machine can verify a GPU path.
- **SOG output is not byte-deterministic:** the same input gave 26,809 and
  26,867 bytes in two runs, presumably from the SH clustering's random
  start. So `bytes_sog` varies by a fraction of a percent between runs, and
  tests do not compare SOG bytes.

**Not done, and why:** "outputs open in SuperSplat" is a manual check on the
Mac.

**Next:** P1.4 gsplat backend (#23) needs the GPU box; P1.6 metrics and
`psplat eval` (#25) and P1.7 web viewer (#26) can proceed here.

---

## 2026-11-08 (evening) — P1.6 metrics and eval

**What ran:** #57 (P1.3) merged. P1.6 `psplat eval` and `eval/metrics.py`
implemented in the remote container.

**Results:**

- **What eval does:** renders the first k splats for each budget from the
  eval views and scores each render. The reference is either the whole
  scene's render or, with `--images`, the training photographs. It writes
  `results.parquet` (one row per budget and view, with the columns of
  handoff §4.2 step 5) and can store each cut's mean in the manifest.
- **Metrics:** PSNR and SSIM are psplat's own; LPIPS uses `lpips` (AlexNet
  by default); ꟻLIP uses NVIDIA's `flip-evaluator`; ColorVideoVDP is
  behind `psplat[cvvdp]`.
- **Fixture, against the whole scene, on 2 eval views:**

| Budget | PSNR | SSIM | LPIPS | ꟻLIP |
|---|---|---|---|---|
| 5% | 17.0 | 0.736 | 0.135 | 0.124 |
| 25% | 22.0 | 0.876 | 0.053 | 0.078 |
| 50% | 28.0 | 0.964 | 0.020 | 0.041 |
| 75% | 34.2 | 0.989 | 0.005 | 0.017 |

**Checks against references:**

- SSIM equals scikit-image to 1e-6.
- ꟻLIP on NVIDIA's published pair: 0.159691, which is flip-evaluator's own
  result from the files; NVIDIA's test expects 0.159693 (made on Windows).
- LPIPS equals the `lpips` package's own image path.
- ColorVideoVDP gives 10 JOD for identical images.

**Findings:**

- A first ꟻLIP wrapper passed ppd = 67 and got 0.159717. ꟻLIP's default
  is 67.0206 pixels per degree, computed from its viewing condition, and
  that is enough to move the fifth decimal. Now exact.
- ColorVideoVDP 0.5.7 is MIT (its LICENSE file), which settles the licence
  check that #25 required.
  - It needs a `torch.device` object; a device string fails inside it.
  - Its ML variant downloads from Hugging Face, but psplat does not use it.
- torchvision 0.29 deprecates its image decoding. `--images` reads with
  Pillow instead, which torchvision already requires.
- LPIPS's AlexNet weights are a 233 MB one-time download; the README says
  so.
- The fixture's eval takes about 4 s on this CPU, rendering included.

**Next:** P1.7 web viewer (#26). P1.4 needs the GPU box.

---

## 2026-11-08 (night) — P1.7 web viewer

**What ran:** #58 (P1.6) merged. P1.7 `psplat viewer` and `engines/web/`
built in the remote container. The viewer was checked in headless Chromium
with software WebGL, on the fixture packed as PLY and as SOG.

**Results:**

- **Viewer:** the PlayCanvas engine 2.23.3, vendored and unmodified, plus
  one page and one module; no build step. `psplat viewer dist` serves it with
  the scene, manifest and cameras on 127.0.0.1.
- **Controls:**
  - a budget slider: cut points, or any count for a PLY;
  - a split-screen wipe against the whole scene;
  - readouts of splats, bytes and frame time;
  - the E4 projected-size threshold;
  - dataset camera views.
- **Checks in the browser:**
  - brightness rises with every cut, and the slider follows;
  - the wipe's halves equal the budget and full images exactly;
  - a 1000 px threshold hides everything.
- **SOG against PLY:** each budget adds the same brightness over the
  smallest, within 0.05 of the PLY. A Morton-sorted SOG, written by
  splat-transform's command line, misses by up to 40 and is kept in the
  tests as a negative control.

**Findings:**

- **PlayCanvas reorders splats on load.** A PLY is Morton-sorted unless the
  asset sets `reorder: false`, and the glTF loader always sorts, so the
  viewer refuses glTF. That would have silently broken every budget.
- **Ways PlayCanvas offers to draw fewer splats did not fit:**
  - `GSplatContainer` can draw a prefix, but forces SH bands to 0;
  - `splatBudget` applies only to its streamed spatial level-of-detail
    format;
  - shader-side hiding keeps every splat in the sort.

  So the viewer builds a resource from the first k rows of the loaded data
  (views, not copies). It keeps SH, and the frame time follows the budget.
- `scissorRect` on a camera component is ignored; the engine applies it only
  to shadow cameras. The wipe uses two half-canvas viewports, with the full
  canvas's projection shifted by ±1 in clip-space x.
- **A SOG needs two things:**
  - PlayCanvas's texture handler, since its images load as texture assets;
    without it the loader fails deep inside decompression;
  - `decompress: true`, so the loader decodes it while it still holds the
    textures.
- **A flaky SOG/PLY comparison was traced, not retried away.** One large
  splat's spherical-harmonic colour varies between SOG writes, because the
  clustering starts at random. Up to 3.7% on the left half was measured.
  The test now compares per-budget gains; a Morton-sorted SOG shows the
  comparison still catches a wrong order.

**Not done:** a person opening it in Chrome or Edge on Windows, and the
garden scene. CI drives the Windows wheel's viewer in the runner's Edge,
headless, which is the closest automated stand-in.

**Next:** P1.8, the Unreal sample (Mac; time-boxed), or P1.12, README as the
TD guide.
