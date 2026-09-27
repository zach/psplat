# Third-party software

psplat is Apache-2.0. Runtime dependencies must be MIT-compatible
(handoff §5.4). GPL and AGPL tools may be *invoked* as external processes by
a user, but are never imported or vendored.

This file lists every third-party component psplat depends on, invokes, or
vendors, with its licence. Update it in the same PR that adds the dependency.

## Runtime dependencies (installed by `pip install psplat`)

| Package | Licence | Used for |
|---|---|---|
| numpy | BSD-3-Clause | arrays, PLY memory mapping |
| torch | BSD-3-Clause | `torch_backend`, projection, metrics |
| typer | MIT | command-line interface |
| PyYAML | MIT | scene registry (`psplat/bench/scenes.yaml`) |

## Optional extras

| Extra | Package | Licence | Status |
|---|---|---|---|
| `msplat` | msplat | Apache-2.0 | declared; backend lands in P1.5 (#24) |
| `cuda` | gsplat | Apache-2.0 | pin lands in P1.10 (#29) |
| `cvvdp` | ColorVideoVDP | to be checked | not added until licence review (#25) |

## Development only

| Package | Licence |
|---|---|
| pytest | MIT |
| ruff | MIT |
| mypy | MIT |
| hatchling | MIT |

## Invoked as external processes, never imported

| Tool | Licence | Used by |
|---|---|---|
| gsplat `examples/simple_trainer.py` | Apache-2.0 | `psplat bench train` on the AWS box |
| AWS CLI v2 | Apache-2.0 | `scripts/aws_*.sh` |

## Data

Datasets are fetched by script and never redistributed. See
`psplat/bench/scenes.yaml` for each set's licence note.
