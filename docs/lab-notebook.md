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
