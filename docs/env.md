<!-- psplat-docs: audience=operator -->
# Operator environments

Where psplat is developed and benchmarked, as opposed to where it is used.
A technical director never needs anything on this page (handoff §0.1); their
path is the README and [`windows.md`](windows.md).

| Environment | Role |
|---|---|
| Mac (M4 Max) | Claude Code, orchestration, small scenes with msplat, the study rig, Unreal authoring |
| AWS Linux GPU box | training, analysis and benchmarks at full speed |
| Claude Code remote container | CPU-only development and tests |

## Mac

Needs macOS 14 or later, the Xcode command-line tools and Homebrew.

```bash
xcode-select --install
brew install uv node awscli
brew install --cask session-manager-plugin
```

Then, from a checkout of this repository:

```bash
uv venv -p 3.12 .venv
uv pip install --python .venv -e ".[dev,msplat]"
.venv/bin/pytest -m "not slow"
.venv/bin/python -c "import msplat; print(msplat.__version__)"
```

On macOS, PyTorch from PyPI includes MPS support, so
`psplat analyze --backend torch --device mps` uses the GPU.

The local data mirror lives at `~/psplat-data/`. Sync it from the bucket
without the bulky eval renders:

```bash
aws s3 sync "s3://$PSPLAT_BUCKET/" ~/psplat-data/ --exclude "*/renders/*"
```

## AWS

The GPU box is provisioned and bootstrapped by the scripts in `scripts/`;
see [P0.10](https://github.com/zach/psplat/issues/16). Record the versions
the bootstrap reports in the table below after the first run.

## Recorded versions

Versions psplat was last verified with, per environment. Update this table
when a bootstrap or upgrade changes them.

| Component | Remote container (2026-09-27) | Mac | AWS |
|---|---|---|---|
| OS | Linux x86_64, no GPU | not yet recorded | not yet recorded |
| Python | 3.12.11 | | |
| PyTorch | 2.14.0 (PyPI CUDA 13.0 build, running on CPU) | | |
| NumPy | 2.5.3 | | |
| typer | 0.27.2 | | |
| gsplat | 1.5.3 from PyPI, for its pure-PyTorch reference functions only | | |
| uv | 0.8.17 | | |
| Node | 22.22.2 | | |
| ruff / mypy / pytest | 0.16.9 / 2.3.1 / 9.1.1 | | |

### Remote container notes

The remote container's network policy blocks `download.pytorch.org`, so the
CPU-only PyTorch index cannot be used there; PyTorch comes from PyPI, which
on Linux is the CUDA build (about 5 GB installed) and runs fine on CPU.
Hugging Face is also blocked. PyPI and the npm registry are reachable.

There is no lockfile yet. On Linux, CI installs PyTorch from the CPU index
first so it does not download the CUDA build, which a single cross-platform
`uv.lock` cannot express cleanly. Pins are version floors in
`pyproject.toml` until the environments above are recorded.
