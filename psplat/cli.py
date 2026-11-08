"""The ``psplat`` command-line interface.

Every subcommand is also a Python function (handoff §3.1 principle 4). Commands
that have not landed yet are present as stubs so ``psplat --help`` shows the
whole v0.1 surface; each stub names the issue that tracks it.
"""

from __future__ import annotations

from pathlib import Path
from typing import NoReturn

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


def _fail(message: str) -> NoReturn:
    typer.echo(f"error: {message}", err=True)
    raise typer.Exit(code=1)


def _duration(seconds: float) -> str:
    if seconds < 60:
        return "under a minute"
    if seconds < 3600:
        return f"about {round(seconds / 60)} min"
    return f"about {seconds / 3600:.1f} h"


@app.command()
def analyze(
    scene_path: Path = typer.Argument(..., metavar="SCENE.PLY", help="Trained splat scene."),
    cameras: Path = typer.Option(
        ..., "--cameras", "-c", help="cams.json, a COLMAP model directory, or transforms.json."
    ),
    backend: str = typer.Option(
        "auto", help="auto, torch, gsplat, msplat or null. auto picks the fastest available."
    ),
    views: str = typer.Option(
        "train", help="train, eval, all, or a number of evenly spaced views."
    ),
    terms: str = typer.Option(
        "contribution",
        help="Comma-separated analysis terms: contribution (always computed), hits, top, "
        "fisher, or baselines for all four. See docs/baselines.md.",
    ),
    out: Path | None = typer.Option(
        None, "--out", "-o", help="Output file. Default: SCENE.weights.npz next to the scene."
    ),
    device: str | None = typer.Option(None, help="Torch device: cpu, cuda or mps. Default: best."),
) -> None:
    """Per-splat analysis over a camera set; writes scene.weights.npz."""
    from psplat.analyze.contribution import TermUnavailable, check_terms
    from psplat.analyze.contribution import analyze as run_analysis
    from psplat.io.cameras import CameraFormatError, load_cameras, select_views
    from psplat.io.ply import PlyFormatError, read_ply
    from psplat.provenance import collect, sha256_file
    from psplat.render.base import BackendUnavailable
    from psplat.render.registry import get_backend, select_backend

    try:
        term_list = check_terms(terms.split(","))
        scene = read_ply(scene_path)
        all_cams = load_cameras(cameras)
        chosen = select_views(all_cams, views)
    except (TermUnavailable, PlyFormatError, CameraFormatError, ValueError, OSError) as exc:
        _fail(str(exc))
    if not chosen:
        _fail(f"--views {views} selects no cameras from {cameras}")

    try:
        if backend == "auto":
            selection = select_backend(device=device)
            renderer, reason = selection.renderer, selection.reason
        else:
            renderer, reason = get_backend(backend, device=device), "requested"
        estimate = renderer.estimate_seconds(scene, chosen)
    except BackendUnavailable as exc:
        _fail(str(exc))

    size = f"{chosen[0].width}x{chosen[0].height}"
    when = f"; {_duration(estimate)}" if estimate is not None else ""
    typer.echo(
        f"backend: {renderer.describe()} ({reason}); "
        f"{scene.count:,} splats x {len(chosen)} views at {size}{when}",
        err=True,
    )

    step = max(1, len(chosen) // 10)

    def progress(done: int, total: int) -> None:
        if done == total or done % step == 0:
            typer.echo(f"  view {done}/{total}", err=True)

    target = out or scene_path.with_name(scene_path.stem + ".weights.npz")
    provenance = collect(
        command="analyze",
        args={"scene": str(scene_path), "cameras": str(cameras), "backend": backend,
              "views": views, "terms": term_list, "device": device},
        source={"ply": scene_path.name, "sha256": sha256_file(scene_path)},
    )  # fmt: skip
    try:
        weights = run_analysis(scene, chosen, renderer, term_list, progress, provenance)
    except BackendUnavailable as exc:
        _fail(str(exc))
    weights.save(target)
    typer.echo(f"wrote {target} ({weights.meta['wall_seconds']:.1f} s)")


@app.command()
def order(
    scene_path: Path = typer.Argument(..., metavar="SCENE.PLY", help="Trained splat scene."),
    weights_path: Path | None = typer.Argument(
        None,
        metavar="[WEIGHTS.NPZ]",
        help="From psplat analyze. Default: SCENE.weights.npz next to the scene.",
    ),
    method: str = typer.Option(
        "contribution",
        help="Score to sort by: opacity, contribution, lightgaussian, pup-simple, "
        "coreset-simple or minisplatting-simple. See docs/baselines.md.",
    ),
    cuts: str = typer.Option("5,10,25,50,75,100", help="Budgets in percent; 100 is always added."),
    seed: int = typer.Option(0, help="Random seed for the sampled methods."),
    out: Path | None = typer.Option(
        None, "--out", "-o", help="Ordered PLY. Default: SCENE.ordered.ply next to the scene."
    ),
    manifest_path: Path | None = typer.Option(
        None, "--manifest", help="Default: SCENE.psplat.json next to the ordered PLY."
    ),
) -> None:
    """Sort splats best first; write the ordered PLY and its cut-point manifest."""
    from psplat.analyze.scores_baseline import BASELINES, MissingStatistic, score
    from psplat.io import manifest as mf
    from psplat.io.ply import PlyFormatError, read_ply, write_ply
    from psplat.io.weights import Weights, WeightsFormatError
    from psplat.order.order import apply_order, sort_order
    from psplat.provenance import collect, sha256_file

    if method not in BASELINES:
        _fail(f"unknown --method {method!r}; choose from {', '.join(BASELINES)}")
    baseline = BASELINES[method]
    try:
        cut_list = [float(c) for c in cuts.split(",") if c.strip()]
        mf.cuts_for(1, cut_list)
    except ValueError:
        _fail(f"--cuts must be percentages in (0, 100], comma-separated; got {cuts!r}")

    try:
        scene = read_ply(scene_path)
    except (PlyFormatError, OSError) as exc:
        _fail(str(exc))
    source_sha = sha256_file(scene_path)

    weights = None
    weights_sha = None
    if baseline.needs:
        weights_path = weights_path or scene_path.with_name(scene_path.stem + ".weights.npz")
        if not weights_path.exists():
            _fail(
                f"{weights_path} not found. --method {method} needs the output of "
                f"psplat analyze; run it first, or pass the weights file after the scene"
            )
        try:
            weights = Weights.load(weights_path)
        except (WeightsFormatError, OSError, ValueError) as exc:
            _fail(str(exc))
        analysed = weights.meta.get("provenance", {}).get("source", {}).get("sha256")
        if analysed is not None and analysed != source_sha:
            _fail(
                f"{weights_path} was computed from a different PLY than {scene_path} "
                "(SHA-256 differs); rerun psplat analyze on this scene"
            )
        weights_sha = sha256_file(weights_path)

    try:
        scores = score(method, scene, weights, seed=seed)
        permutation = sort_order(scores)
    except (MissingStatistic, ValueError) as exc:
        _fail(str(exc))

    target = out or scene_path.with_name(scene_path.stem + ".ordered.ply")
    if target.resolve() == scene_path.resolve():
        _fail("--out must differ from the input scene")
    manifest_target = manifest_path or target.with_name(scene_path.stem + ".psplat.json")
    provenance = collect(
        command="order",
        args={"scene": str(scene_path), "weights": str(weights_path) if weights else None,
              "method": method, "cuts": cut_list, "seed": seed},
    )  # fmt: skip
    provenance["date"] = provenance["created_utc"]
    if weights is not None:
        provenance["backend"] = str(weights.meta.get("backend_detail", weights.meta.get("backend")))
        provenance["analysis"] = {
            "weights_sha256": weights_sha,
            **{k: weights.meta[k] for k in ("terms", "views", "wall_seconds") if k in weights.meta},
        }
    manifest = mf.build(
        source_sha256=source_sha,
        count=scene.count,
        method=method,
        terms=list(baseline.needs),
        cuts_percent=cut_list,
        provenance=provenance,
        seed=seed if baseline.sampled else None,
    )
    try:
        for path in (target, manifest_target):
            path.parent.mkdir(parents=True, exist_ok=True)
        write_ply(apply_order(scene, permutation), target)
        mf.write(manifest, manifest_target)
    except OSError as exc:
        _fail(f"could not write output: {exc}")
    budgets = ", ".join(f"{c['frac']:.0%} {c['index']:,}" for c in manifest["cuts"])
    typer.echo(f"wrote {target} ({scene.count:,} splats by {method})")
    typer.echo(f"wrote {manifest_target} (cuts: {budgets})")


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


bench_app = typer.Typer(
    help="Datasets, training and experiments over the scene registry.", no_args_is_help=True
)
app.add_typer(bench_app, name="bench")


@bench_app.command("scenes")
def bench_scenes() -> None:
    """List the registry: sets, scenes and where their data comes from."""
    from psplat.bench.registry import load_registry

    reg = load_registry()
    for name, members in reg.sets.items():
        typer.echo(f"{name}: {', '.join(members)}")
    typer.echo("")
    for scene in reg.scenes.values():
        archive = reg.archives[scene.archive]
        checked = f"verified {archive.verified}" if archive.verified else "URL not verified"
        where = f"{scene.archive}/{scene.path}"
        typer.echo(f"{scene.name:10s} {where:26s} factor {scene.data_factor}  ({checked})")
    typer.echo("")
    typer.echo("trainers: " + ", ".join(reg.trainers))


@bench_app.command("fetch")
def bench_fetch(
    names: list[str] = typer.Argument(..., help="Scene or set names, e.g. mipnerf360-core."),
    dest: Path = typer.Option(Path("data"), help="Local data root; scenes go to DEST/raw/."),
    s3_bucket: str | None = typer.Option(None, help="Also sync each scene to s3://BUCKET/raw/."),
) -> None:
    """Download datasets and extract the requested scenes (idempotent)."""
    from psplat.bench.fetch import FetchError, fetch
    from psplat.bench.registry import RegistryError, load_registry

    try:
        fetch(load_registry(), names, dest, s3_bucket=s3_bucket, log=typer.echo)
    except (FetchError, RegistryError, OSError) as exc:
        _fail(str(exc))


@bench_app.command("train")
def bench_train(
    scene: str = typer.Argument(..., help="Scene name from the registry."),
    trainer: str = typer.Option("gsplat-mcmc-1m", help="Recipe from the registry."),
    data: Path = typer.Option(..., help="The scene's dataset folder (images + sparse/0)."),
    out: Path = typer.Option(Path("trained"), help="Output root."),
    gsplat_examples: Path = typer.Option(
        Path.home() / "gsplat" / "examples", help="gsplat checkout's examples folder."
    ),
    dry_run: bool = typer.Option(False, help="Print the command and stop."),
) -> None:
    """Train one scene with a registry recipe (E0); writes PLY, cameras, metrics, run.json."""
    from psplat.bench.registry import RegistryError, load_registry
    from psplat.bench.train import TrainError, train_gsplat, train_msplat

    try:
        reg = load_registry()
        entry, recipe = reg.scene(scene), reg.trainer(trainer)
        if recipe.kind == "gsplat":
            train_gsplat(recipe, entry, data, out, gsplat_examples, dry_run=dry_run, log=typer.echo)
        else:
            train_msplat(recipe, entry, data, out, dry_run=dry_run, log=typer.echo)
    except (RegistryError, TrainError, OSError) as exc:
        _fail(str(exc))


@bench_app.command("e0")
def bench_e0(
    trained: Path = typer.Argument(Path("trained"), help="Root written by bench train."),
) -> None:
    """Compare trained scenes' metrics with the published numbers in the registry."""
    import json

    from psplat.bench.registry import load_registry
    from psplat.bench.train import compare_to_reference

    results = []
    for run_json in sorted(trained.glob("*/*/*/run.json")):
        run = json.loads(run_json.read_text(encoding="utf-8"))
        m = run.get("metrics", {})
        if "psnr" in m and "ssim" in m:
            results.append({"scene": run["scene"], "trainer": run["trainer"]["name"],
                            "psnr": float(m["psnr"]), "ssim": float(m["ssim"])})  # fmt: skip
    if not results:
        _fail(f"no trained runs with metrics under {trained}")
    for row in compare_to_reference(results, load_registry().reference):
        ref = (
            f"ref {row['ref_psnr']:.2f} / {row['ref_ssim']:.3f}" if row["ref_psnr"] is not None
            else "ref -"
        )  # fmt: skip
        typer.echo(
            f"{row['scene']:10s} {row['trainer']:16s} PSNR {row['psnr']:.2f}  "
            f"SSIM {row['ssim']:.3f}  {ref}  {row['status']}"
        )


@bench_app.command("run", context_settings=_STUB)
def bench_run() -> None:
    """Run analyse, order, pack and eval over scenes and orderings (E1, E2)."""
    _not_yet("bench run", 28)


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
