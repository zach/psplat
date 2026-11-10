"""The ``psplat`` command-line interface.

Every subcommand is also a Python function (handoff §3.1 principle 4). Commands
that have not landed yet are present as stubs so ``psplat --help`` shows the
whole v0.1 surface; each stub names the issue that tracks it.
"""

from __future__ import annotations

import math
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
        # pack checks it is given the very file this manifest describes.
        manifest["provenance"]["output"] = {"ply": target.name, "ply_sha256": sha256_file(target)}
        mf.write(manifest, manifest_target)
    except OSError as exc:
        _fail(f"could not write output: {exc}")
    budgets = ", ".join(f"{c['frac']:.0%} {c['index']:,}" for c in manifest["cuts"])
    typer.echo(f"wrote {target} ({scene.count:,} splats by {method})")
    typer.echo(f"wrote {manifest_target} (cuts: {budgets})")


@app.command()
def pack(
    ordered: Path = typer.Argument(
        ..., metavar="SCENE.ORDERED.PLY", help="Ordered PLY from psplat order."
    ),
    ply: bool = typer.Option(False, "--ply", help="Ordered PLY: any prefix is a valid budget."),
    sog: bool = typer.Option(False, "--sog", help="PlayCanvas SOG; budgets snap to cut points."),
    spz: bool = typer.Option(False, "--spz", help="SPZ v4, in exact psplat order."),
    gltf: bool = typer.Option(
        False, "--gltf", help="glTF .glb with KHR_gaussian_splatting, in exact psplat order."
    ),
    out: Path | None = typer.Option(
        None, "--out", "-o", help="Output folder. Default: dist next to the input."
    ),
    manifest_path: Path | None = typer.Option(
        None, "--manifest", help="Default: SCENE.psplat.json next to the input."
    ),
    alloc: str = typer.Option("uniform", help="Attribute bit allocation; only uniform so far."),
    cut_sizes: bool = typer.Option(
        True, help="Pack each cut on its own to record its size in the manifest."
    ),
    sog_sh: str = typer.Option(
        "auto",
        help="SOG spherical harmonics: auto keeps them up to 32,768 splats, keep always "
        "(slow on CPU), drop never.",
    ),
) -> None:
    """Write the ordered scene as PLY, SOG, SPZ or glTF, with per-cut byte sizes.

    Exits 3 if a requested format had to be skipped (for example, no Node.js).
    """
    from psplat.io.manifest import ManifestError
    from psplat.io.ply import PlyFormatError
    from psplat.pack.pack import PackError, output_stem
    from psplat.pack.pack import pack as run_pack

    formats = [f for f, on in (("ply", ply), ("sog", sog), ("spz", spz), ("gltf", gltf)) if on]
    if not formats:
        _fail("choose at least one format: --ply, --sog, --spz or --gltf")
    if not ordered.exists():
        _fail(f"{ordered} not found")
    stem = output_stem(ordered)
    manifest_in = manifest_path or ordered.with_name(f"{stem}.psplat.json")
    if not manifest_in.exists():
        _fail(f"{manifest_in} not found; psplat order writes it next to the ordered PLY")
    target = out or ordered.parent / "dist"
    try:
        result = run_pack(
            ordered, manifest_in, formats, target, alloc=alloc, cut_sizes=cut_sizes,
            sog_sh=sog_sh, log=lambda m: typer.echo(m, err=True),
        )  # fmt: skip
    except (PackError, ManifestError, PlyFormatError, OSError, RuntimeError) as exc:
        _fail(str(exc))
    for notice in result.notices:
        typer.echo(f"note: {notice}", err=True)
    for path in result.outputs.values():
        typer.echo(f"wrote {path} ({path.stat().st_size:,} bytes)")
    typer.echo(f"wrote {result.manifest_path}")
    if result.skipped:
        for fmt, reason in result.skipped.items():
            typer.echo(f"skipped {fmt}: {reason}", err=True)
        raise typer.Exit(code=3)


@app.command(context_settings=_STUB)
def lod() -> None:
    """Projected-size thresholds per viewing distance; writes lod.json."""
    _not_yet("lod", 40)


@app.command("eval")
def eval_(
    ordered: Path = typer.Argument(
        ..., metavar="SCENE.ORDERED.PLY", help="Ordered PLY from psplat order."
    ),
    cameras: Path = typer.Option(
        ..., "--cameras", "-c", help="cams.json, a COLMAP model directory, or transforms.json."
    ),
    views: str = typer.Option("eval", help="eval, train, all, or a number of evenly spaced views."),
    budgets: str = typer.Option(
        "cuts", help="Percentages, e.g. 5,10,25,50,100, or cuts for the manifest's cut points."
    ),
    metrics: str = typer.Option(
        "psnr,ssim,lpips,flip", help="Any of psnr, ssim, lpips, flip, cvvdp."
    ),
    images: Path | None = typer.Option(
        None,
        help="Compare with these photographs, named as the cameras are, "
        "instead of with the whole scene's render.",
    ),
    manifest_path: Path | None = typer.Option(
        None, "--manifest", help="Default: SCENE.psplat.json next to the input, if present."
    ),
    write_manifest: bool = typer.Option(
        False, help="Also store each cut's mean metrics in the manifest."
    ),
    backend: str = typer.Option("auto", help="Renderer: auto, torch, gsplat or msplat."),
    device: str | None = typer.Option(None, help="Torch device: cpu, cuda or mps. Default: best."),
    lpips_net: str = typer.Option("alex", help="LPIPS backbone: alex or vgg."),
    flip_ppd: float | None = typer.Option(
        None, help="Pixels per degree for ꟻLIP. Default: its standard 67.02."
    ),
    scene_name: str | None = typer.Option(None, help="Scene label for the results table."),
    trainer: str = typer.Option("", help="Trainer label for the results table."),
    out: Path | None = typer.Option(
        None, "--out", "-o", help="Default: SCENE.results.parquet next to the input."
    ),
) -> None:
    """Render truncated scenes at each budget and score them; writes results.parquet."""
    from psplat.eval.evaluate import (
        DEFAULT_BUDGETS,
        ImageError,
        budgets_for,
        budgets_from_manifest,
        evaluate,
        summarise,
        write_results,
    )
    from psplat.eval.metrics import FLIP_PPD, MetricUnavailable, check_metrics, load_metrics
    from psplat.io import manifest as mf
    from psplat.io.cameras import CameraFormatError, load_cameras, select_views
    from psplat.io.ply import PlyFormatError, read_ply
    from psplat.pack.pack import output_stem, ply_prefix_bytes
    from psplat.render.base import BackendUnavailable
    from psplat.render.registry import get_backend, select_backend
    from psplat.render.torch_backend import default_device

    try:
        names = check_metrics(metrics.split(","))
        scene = read_ply(ordered)
        chosen = select_views(load_cameras(cameras), views)
    except (ValueError, PlyFormatError, CameraFormatError, OSError) as exc:
        _fail(str(exc))
    if not chosen:
        _fail(
            f"--views {views} selects no cameras from {cameras}; they may have no eval split. "
            "Use --views all or a number of views."
        )
    stem = output_stem(ordered)
    manifest_in = manifest_path or ordered.with_name(f"{stem}.psplat.json")
    manifest = None
    if manifest_in.exists():
        try:
            manifest = mf.read(manifest_in)
        except mf.ManifestError as exc:
            _fail(str(exc))
        if manifest["count"] != scene.count:
            _fail(f"{manifest_in} is for {manifest['count']:,} splats, not {scene.count:,}")
    elif manifest_path is not None:
        _fail(f"{manifest_path} not found")
    try:
        if budgets.strip() == "cuts":
            plan = (
                budgets_from_manifest(manifest)
                if manifest
                else budgets_for(scene.count, DEFAULT_BUDGETS)
            )
        else:
            plan = budgets_for(scene.count, [float(b) for b in budgets.split(",") if b.strip()])
    except ValueError:
        _fail(f"--budgets must be percentages in (0, 100] or cuts; got {budgets!r}")
    if write_manifest and manifest is None:
        _fail("--write-manifest needs a manifest; run psplat order first")

    try:
        if backend == "auto":
            selection = select_backend(device=device)
            renderer, reason = selection.renderer, selection.reason
        else:
            renderer, reason = get_backend(backend, device=device), "requested"
        if "lpips" in names:
            typer.echo(f"loading LPIPS ({lpips_net}); PyTorch downloads its weights once", err=True)
        metric_set = load_metrics(
            names, device=device or default_device(), lpips_net=lpips_net,
            flip_ppd=flip_ppd if flip_ppd is not None else FLIP_PPD,
        )  # fmt: skip
    except (BackendUnavailable, MetricUnavailable, ValueError) as exc:
        _fail(str(exc))

    typer.echo(
        f"backend: {renderer.describe()} ({reason}); {len(plan)} budgets x {len(chosen)} views; "
        f"reference: {'photographs in ' + str(images) if images else 'the whole scene'}",
        err=True,
    )

    def progress(done: int, total: int) -> None:
        if done == total or done % max(1, total // 10) == 0:
            typer.echo(f"  {done}/{total}", err=True)

    try:
        rows = evaluate(scene, chosen, renderer, metric_set, plan, images=images,
                        progress=progress)  # fmt: skip
    except (ImageError, BackendUnavailable) as exc:
        _fail(str(exc))

    sizes = ply_prefix_bytes(ordered, [b.splats for b in plan if b.splats > 0])
    provenance = (manifest or {}).get("provenance", {})
    common = {
        "scene": scene_name or stem,
        "trainer": trainer,
        "ordering": (manifest or {}).get("ordering", {}).get("method", "file order"),
        "terms": ",".join((manifest or {}).get("ordering", {}).get("terms", [])),
        "format": "ply",
        "reference": "images" if images else "full",
        "backend": renderer.describe(),
        "git_sha": (provenance.get("git") or {}).get("sha"),
        "source_sha256": (manifest or {}).get("source_sha256"),
        "tool": f"psplat {__version__}",
    }
    for row in rows:
        row.update(common)
        row["bytes"] = sizes.get(int(row["budget_splats"]))
    target = out or ordered.with_name(f"{stem}.results.parquet")
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        write_results(rows, target)
    except OSError as exc:
        _fail(f"could not write {target}: {exc}")

    summary = summarise(rows, names)
    typer.echo("budget      splats  " + "  ".join(f"{n:>8}" for n in names))
    for entry in summary:
        cells = "  ".join(f"{entry[n]:8.4f}" for n in names)
        typer.echo(f"{entry['budget_frac']:>6.0%}  {entry['budget_splats']:>10,}  {cells}")
    typer.echo(f"wrote {target} ({len(rows)} rows)")

    if write_manifest and manifest is not None:
        by_index = {e["budget_splats"]: e for e in summary}
        for cut in manifest["cuts"]:
            means = by_index.get(int(cut["index"]))
            for n in names:
                if means is not None and math.isfinite(means[n]):
                    cut[n] = round(float(means[n]), 6)
        manifest["provenance"]["eval"] = {
            "reference": common["reference"], "views": len(chosen), "metrics": list(names),
            "settings": metric_set.settings, "backend": renderer.describe(),
        }  # fmt: skip
        mf.write(manifest, manifest_in)
        typer.echo(f"updated {manifest_in}")


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


@app.command()
def viewer(
    scene_path: Path = typer.Argument(
        ..., metavar="SCENE", help="A .sog or .ply from psplat pack, or the folder holding it."
    ),
    manifest_path: Path | None = typer.Option(
        None, "--manifest", help="Default: SCENE.psplat.json beside the scene, if present."
    ),
    cameras: Path | None = typer.Option(
        None, "--cameras", "-c", help="cams.json for the view list. Default: found nearby."
    ),
    budget: str | None = typer.Option(
        None, help="Start at this budget: a fraction such as 0.25, or a splat count."
    ),
    port: int = typer.Option(8642, help="Local port; 0 picks a free one."),
    browser: bool = typer.Option(True, help="Open the default browser."),
) -> None:
    """Open the web viewer with a budget slider; serves on this machine only."""
    from psplat.viewer import ViewerError, open_browser, resolve, start

    try:
        served = resolve(scene_path, manifest_path, cameras)
    except ViewerError as exc:
        _fail(str(exc))
    if budget is not None:
        served.extra["budget"] = budget
    try:
        server = start(served, port=port, background=False)
    except OSError as exc:
        _fail(f"cannot listen on port {port} ({exc}); try --port 0")
    parts = [served.scene.name]
    parts.append(served.manifest.name if served.manifest else "no manifest")
    if served.cameras:
        parts.append(served.cameras.name)
    typer.echo(f"serving {', '.join(parts)}")
    typer.echo(f"open {server.url} in Chrome, Edge or Firefox; Ctrl+C stops the viewer")
    if browser:
        open_browser(server.url)
    try:
        server.httpd.serve_forever()
    except KeyboardInterrupt:
        typer.echo("stopped")
    finally:
        server.httpd.server_close()


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
