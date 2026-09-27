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

The GPU box is provisioned and bootstrapped by the scripts in `scripts/`
([P0.10](https://github.com/zach/psplat/issues/16)). From the Mac:

```bash
export PSPLAT_ALERT_EMAIL=you@example.com   # optional monthly budget alarm
scripts/aws_provision.sh                    # asks before launching; bills hourly
aws ssm start-session --target <instance-id>
```

Then on the instance:

```bash
git clone https://github.com/zach/psplat ~/psplat
PSPLAT_GSPLAT_REF=<sha> ~/psplat/scripts/aws_bootstrap.sh
```

Pin `PSPLAT_GSPLAT_REF` to a commit for reproducible runs. The bootstrap
prints the versions it installed; record them in the table below. The
instance stops itself after 45 idle minutes.

## E0: reproduce the baseline

E0 ([P0.11](https://github.com/zach/psplat/issues/17)) trains the four
Mip-NeRF 360 core scenes and checks them against published gsplat numbers:
PSNR within 0.3 dB and SSIM within 0.02. Wall clock is roughly 1–2 GPU-hours
per scene.

On the AWS box:

```bash
source ~/.venvs/psplat/bin/activate
psplat bench fetch mipnerf360-core --dest ~/data --s3-bucket "$PSPLAT_BUCKET"
for scene in garden bicycle counter room; do
  psplat bench train "$scene" --trainer gsplat-mcmc-1m \
    --data ~/data/raw/mipnerf360/$scene --out ~/trained
done
psplat bench e0 ~/trained
```

`bench train --dry-run` prints the exact `simple_trainer.py` command. Before
a real run it checks every flag against the installed trainer's `--help`,
so a renamed flag fails immediately rather than after an hour.

Each run folder holds `scene.ply`, `cams.json` in the same normalised world
frame gsplat trained in, `metrics.json`, `run.json` and the raw trainer
output. Push them with `scripts/s3_sync.sh push-trained`.

`bench e0` reports "no reference" until the published numbers are copied
into `reference:` in `psplat/bench/scenes.yaml`, with their source URL. They
were not recorded from memory: gsplat's benchmark page was unreachable from
the development container.

On the Mac, the msplat path is:

```bash
psplat bench fetch garden --dest ~/psplat-data
psplat bench train garden --trainer msplat-7k \
  --data ~/psplat-data/raw/mipnerf360/garden --out ~/psplat-data/trained
```

msplat does not document whether it renormalises the scene, so its
`cams.json` comes from the raw COLMAP model. On the first run, check that
`scene.ply` and `cams.json` line up before trusting later numbers.

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
