"""``psplat bench train``: E0 training wrappers for gsplat (AWS) and msplat (Mac).

Each run leaves, in ``<out>/<scene>/<trainer>/<code id>/``:

- ``scene.ply`` — the trained splats, standard 3DGS layout;
- ``cams.json`` — the cameras in the **same world frame** as ``scene.ply``;
- ``metrics.json`` — the trainer's own evaluation (PSNR, SSIM, ...);
- ``run.json`` — provenance: code versions, command, hardware, wall time;
- ``raw/`` — the trainer's untouched output.

The world frame matters. gsplat's trainer normalises the scene by default
(recentres and rotates it), so its splats do not line up with the raw COLMAP
poses. ``cams.json`` is therefore written from gsplat's own dataset parser,
with the same normalisation, split and image size the trainer used.

Neither trainer can run in the development container, so this module is
written against gsplat's documented example usage and msplat's documented
Python API, with a pre-flight check: every gsplat flag used is looked up in
the installed ``simple_trainer.py --help`` before training starts.
"""

from __future__ import annotations

import importlib
import json
import re
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from psplat.bench.registry import SceneEntry, Trainer
from psplat.io.cameras import Camera, read_colmap, write_cameras
from psplat.io.ply import read_ply_header, write_ply
from psplat.io.scene import Scene
from psplat.provenance import collect

Log = Callable[[str], None]


class TrainError(RuntimeError):
    pass


# gsplat ---------------------------------------------------------------------------


def gsplat_command(
    trainer: Trainer,
    scene: SceneEntry,
    data_dir: Path,
    result_dir: Path,
    examples: Path,
    python: str = sys.executable,
) -> list[str]:
    """The ``simple_trainer.py`` command for one scene and recipe."""
    if trainer.kind != "gsplat" or trainer.strategy is None:
        raise TrainError(f"{trainer.name} is not a gsplat recipe")
    return [
        python, str(examples / "simple_trainer.py"), trainer.strategy,
        "--data_dir", str(data_dir),
        "--data_factor", str(scene.data_factor),
        "--result_dir", str(result_dir),
        "--max_steps", str(trainer.steps),
        "--disable_viewer",
        *trainer.args,
    ]  # fmt: skip


def _flag_key(flag: str) -> str:
    return flag.lstrip("-").replace("_", "-").lower()


def missing_flags(command: list[str], help_text: str) -> list[str]:
    """Flags in ``command`` that ``help_text`` does not mention.

    tyro prints flags with hyphens and accepts underscores, so both spellings
    are normalised before comparing.
    """
    offered = {_flag_key(m) for m in re.findall(r"--[A-Za-z0-9_.\-]+", help_text)}
    used = [part for part in command if part.startswith("--")]
    return [flag for flag in used if _flag_key(flag) not in offered]


def gsplat_preflight(command: list[str]) -> None:
    help_cmd = [*command[:3], "--help"]
    result = subprocess.run(help_cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise TrainError(f"{' '.join(help_cmd)} failed:\n{result.stderr[-2000:]}")
    missing = missing_flags(command, result.stdout)
    if missing:
        raise TrainError(
            "the installed gsplat simple_trainer.py does not offer "
            f"{', '.join(missing)}; update the recipe in psplat/bench/scenes.yaml"
        )


def latest(folder: Path, pattern: str) -> Path:
    """The file in ``folder`` matching ``pattern`` with the highest step number."""
    found = []
    for path in folder.glob(pattern):
        numbers = re.findall(r"\d+", path.stem)
        if numbers:
            found.append((int(numbers[0]), path))
    if not found:
        raise TrainError(f"no {pattern} in {folder}")
    return max(found)[1]


def scene_from_gsplat_ckpt(path: Path) -> Scene:
    """Splats from a gsplat ``simple_trainer`` checkpoint.

    The checkpoint's ``splats`` hold the same storage conventions as a 3DGS
    PLY (log scales, logit opacity, wxyz quaternions); ``sh0`` is
    ``(N, 1, 3)`` and ``shN`` ``(N, K, 3)``.
    """
    import torch

    data = torch.load(path, map_location="cpu", weights_only=True)
    splats = data.get("splats", data)
    needed = ("means", "scales", "quats", "opacities", "sh0", "shN")
    absent = [k for k in needed if k not in splats]
    if absent:
        raise TrainError(
            f"{path.name} has no {', '.join(absent)}; keys are {sorted(splats)}. "
            "Appearance-optimised checkpoints (colors/features) are not supported."
        )

    def arr(key: str) -> NDArray[np.float32]:
        return np.ascontiguousarray(splats[key].detach().float().numpy())

    return Scene(
        means=arr("means"),
        scales=arr("scales"),
        quats=arr("quats"),
        opacities=arr("opacities").reshape(-1),
        sh0=arr("sh0").reshape(-1, 3),
        shN=arr("shN"),
    )


def gsplat_cameras(
    examples: Path, data_dir: Path, factor: int, test_every: int = 8
) -> list[Camera]:
    """Cameras exactly as gsplat's trainer saw them, in its normalised frame.

    Uses ``datasets/colmap.py``'s ``Parser`` from the gsplat examples, with
    ``normalize=True`` (simple_trainer's default), so poses, intrinsics after
    downscaling or undistortion, and the evaluation split all match.
    """
    sys.path.insert(0, str(examples))
    try:
        colmap = importlib.import_module("datasets.colmap")
    finally:
        sys.path.remove(str(examples))
    parser = colmap.Parser(data_dir=str(data_dir), factor=factor, normalize=True,
                           test_every=test_every)  # fmt: skip
    needed = ("image_names", "camtoworlds", "camera_ids", "Ks_dict", "imsize_dict")
    absent = [name for name in needed if not hasattr(parser, name)]
    if absent:
        raise TrainError(
            f"gsplat's Parser has no {', '.join(absent)}; its API changed. Update "
            "psplat/bench/train.py:gsplat_cameras."
        )
    cams = []
    for i, name in enumerate(parser.image_names):
        cam_id = parser.camera_ids[i]
        k = np.asarray(parser.Ks_dict[cam_id], dtype=np.float64)
        width, height = parser.imsize_dict[cam_id]
        w2c = np.linalg.inv(np.asarray(parser.camtoworlds[i], dtype=np.float64))
        w2c[3] = [0.0, 0.0, 0.0, 1.0]
        cams.append(
            Camera(
                name=str(name), width=int(width), height=int(height),
                fx=float(k[0, 0]), fy=float(k[1, 1]), cx=float(k[0, 2]), cy=float(k[1, 2]),
                w2c=w2c, split="eval" if i % test_every == 0 else "train",
            )
        )  # fmt: skip
    return cams


def git_sha(folder: Path) -> str | None:
    result = subprocess.run(
        ["git", "-C", str(folder), "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    )
    return result.stdout.strip() or None


def train_gsplat(
    trainer: Trainer,
    scene: SceneEntry,
    data_dir: Path,
    out_root: Path,
    examples: Path,
    *,
    dry_run: bool = False,
    log: Log = print,
) -> Path | None:
    if not (examples / "simple_trainer.py").exists():
        raise TrainError(f"{examples} has no simple_trainer.py; pass the gsplat examples folder")
    code = git_sha(examples) or "unknown"
    run_dir = out_root / scene.name / trainer.name / code[:12]
    command = gsplat_command(trainer, scene, data_dir, run_dir / "raw", examples)
    log("command: " + " ".join(command))
    if dry_run:
        return None
    gsplat_preflight(command)
    run_dir.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    subprocess.run(command, check=True)
    wall = time.perf_counter() - start

    raw = run_dir / "raw"
    scene_splats = scene_from_gsplat_ckpt(latest(raw / "ckpts", "ckpt_*_rank0.pt"))
    write_ply(scene_splats, run_dir / "scene.ply")
    write_cameras(gsplat_cameras(examples, data_dir, scene.data_factor), run_dir / "cams.json")
    metrics = json.loads(latest(raw / "stats", "val_step*.json").read_text(encoding="utf-8"))
    return _finish(run_dir, trainer, scene, command, code, wall, metrics, scene_splats.count, log)


# msplat ---------------------------------------------------------------------------


def train_msplat(
    trainer: Trainer,
    scene: SceneEntry,
    data_dir: Path,
    out_root: Path,
    *,
    dry_run: bool = False,
    log: Log = print,
) -> Path | None:
    """Train with msplat's documented Python API (Apple Silicon only).

    Its cameras are written from the raw COLMAP model at full resolution. Whether
    msplat renormalises the scene is not documented; check on the first run by
    rendering one view of ``scene.ply`` with ``cams.json`` before trusting it.
    """
    log(f"msplat: {trainer.steps} iterations on {data_dir}")
    if dry_run:
        return None
    try:
        msplat = importlib.import_module("msplat")
    except ImportError:
        raise TrainError("msplat is not installed; pip install 'psplat[msplat]' on a Mac") from None
    version = str(getattr(msplat, "__version__", "unknown"))
    run_dir = out_root / scene.name / trainer.name / f"msplat-{version}"
    run_dir.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    dataset = msplat.load_dataset(str(data_dir), eval_mode=True)
    config = msplat.TrainingConfig(iterations=trainer.steps)
    model = msplat.GaussianTrainer(dataset, config)
    model.train(lambda s: log(f"step {s.iteration} splats {s.splat_count:,}"), callback_every=1000)
    wall = time.perf_counter() - start
    model.export_ply(str(run_dir / "scene.ply"))
    write_cameras(read_colmap(data_dir), run_dir / "cams.json")
    metrics = {k: float(v) for k, v in dict(model.evaluate()).items()}
    count = read_ply_header(run_dir / "scene.ply").element("vertex").count
    command = ["msplat.GaussianTrainer", f"iterations={trainer.steps}", str(data_dir)]
    return _finish(run_dir, trainer, scene, command, f"msplat {version}", wall, metrics, count, log)


# Shared ---------------------------------------------------------------------------


def _finish(
    run_dir: Path,
    trainer: Trainer,
    scene: SceneEntry,
    command: list[str],
    code: str,
    wall: float,
    metrics: dict[str, Any],
    count: int,
    log: Log,
) -> Path:
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    record = collect(
        command="bench train",
        scene=scene.name,
        data_factor=scene.data_factor,
        trainer={"name": trainer.name, "kind": trainer.kind, "steps": trainer.steps,
                 "strategy": trainer.strategy, "args": list(trainer.args)},
        trainer_code=code,
        trainer_command=command,
        wall_seconds=round(wall, 1),
        splats=count,
        metrics=metrics,
    )  # fmt: skip
    (run_dir / "run.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    log(f"{scene.name}/{trainer.name}: {count:,} splats, {wall / 60:.1f} min -> {run_dir}")
    return run_dir


def compare_to_reference(
    results: list[dict[str, Any]], reference: dict[str, Any]
) -> list[dict[str, Any]]:
    """E0 check: PSNR within 0.3 dB and SSIM within 0.02 of the published number."""
    rows = []
    for r in results:
        ref = reference.get(r["trainer"], {}).get("scenes", {}).get(r["scene"])
        row: dict[str, Any] = {**r, "ref_psnr": None, "ref_ssim": None, "status": "no reference"}
        if ref:
            row["ref_psnr"], row["ref_ssim"] = ref.get("psnr"), ref.get("ssim")
            ok = abs(r["psnr"] - ref["psnr"]) <= 0.3 and abs(r["ssim"] - ref["ssim"]) <= 0.02
            row["status"] = "match" if ok else "MISMATCH"
        rows.append(row)
    return rows
