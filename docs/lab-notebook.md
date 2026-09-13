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
