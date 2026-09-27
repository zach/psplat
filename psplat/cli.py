"""The ``psplat`` command-line interface.

Every subcommand is also a Python function (handoff §3.1 principle 4). Commands
that have not landed yet are present as stubs so ``psplat --help`` shows the
whole v0.1 surface; each stub names the issue that tracks it.
"""

from __future__ import annotations

from pathlib import Path

import typer

from psplat import __version__

ISSUES = "https://github.com/zach/psplat/issues/"

app = typer.Typer(
    name="psplat",
    help="Perceptual, budget-scalable Gaussian splats: analyse once, truncate anywhere.",
    no_args_is_help=True,
    add_completion=False,
)

_STUB = {"allow_extra_args": True, "ignore_unknown_options": True}


def _not_yet(command: str, issue: int) -> None:
    typer.echo(
        f"psplat {command} is not implemented yet. Tracked in {ISSUES}{issue}",
        err=True,
    )
    raise typer.Exit(code=2)


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"psplat {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        False, "--version", callback=_version_callback, is_eager=True, help="Print version."
    ),
) -> None:
    """psplat command-line interface."""


@app.command(context_settings=_STUB)
def analyze() -> None:
    """Per-splat analysis over a camera set; writes scene.weights.npz."""
    _not_yet("analyze", 13)


@app.command(context_settings=_STUB)
def order() -> None:
    """Sort splats by weight and emit cut points plus the manifest."""
    _not_yet("order", 21)


@app.command(context_settings=_STUB)
def pack() -> None:
    """Write the ordered scene as PLY, SOG, SPZ or glTF, with per-cut byte sizes."""
    _not_yet("pack", 22)


@app.command(context_settings=_STUB)
def lod() -> None:
    """Projected-size thresholds per viewing distance; writes lod.json."""
    _not_yet("lod", 40)


@app.command("eval", context_settings=_STUB)
def eval_() -> None:
    """Render truncated scenes at each budget and score them; writes results.parquet."""
    _not_yet("eval", 25)


@app.command(context_settings=_STUB)
def bench() -> None:
    """Run experiments end to end over the scene registry."""
    _not_yet("bench", 28)


@app.command(context_settings=_STUB)
def report() -> None:
    """Build the static HTML report from bench results."""
    _not_yet("report", 28)


@app.command(context_settings=_STUB)
def study() -> None:
    """Serve the local 2AFC study rig."""
    _not_yet("study", 44)


@app.command(context_settings=_STUB)
def viewer() -> None:
    """Open the web viewer with a budget slider."""
    _not_yet("viewer", 26)


@app.command()
def cameras(
    source: Path = typer.Argument(
        ..., help="COLMAP model or dataset directory, transforms.json, or cams.json."
    ),
    out: Path = typer.Option(Path("cams.json"), "--out", "-o", help="Where to write cams.json."),
    downscale: float = typer.Option(
        1.0, help="Divide image size by this, e.g. 4 for Mip-NeRF 360 images_4."
    ),
    holdout: int = typer.Option(
        8, help="Mark every Nth view (by name) as eval; 0 marks all views train."
    ),
    allow_distortion: bool = typer.Option(
        False, help="Ignore lens distortion instead of refusing distorted cameras."
    ),
) -> None:
    """Convert COLMAP or nerfstudio cameras to psplat's cams.json."""
    from psplat.io.cameras import CameraFormatError, load_cameras, write_cameras

    try:
        cams = load_cameras(source, allow_distortion=allow_distortion, holdout_every=holdout)
    except CameraFormatError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from None
    if downscale != 1.0:
        cams = [cam.downscaled(downscale) for cam in cams]
    write_cameras(cams, out)
    n_eval = sum(cam.split == "eval" for cam in cams)
    typer.echo(f"wrote {len(cams)} views ({len(cams) - n_eval} train, {n_eval} eval) to {out}")
