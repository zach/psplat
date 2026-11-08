"""``psplat eval``: render each budget and score it (P1.6, #25).

For every budget, the first ``k`` splats of an ordered scene are rendered from
each evaluation view and compared with a reference image:

- by default, the render of the **whole scene** from the same view. This
  isolates what truncation loses, which is what an ordering is judged on,
  and needs no dataset images. The 100% budget is then perfect by
  construction;
- with ``images``, the **photographs** the scene was trained on, named as
  the cameras are. That adds the reconstruction's own error, so the 100%
  budget shows the trainer's quality.

Results are one row per (budget, view), written as Parquet with the columns
of handoff §4.2 step 5 plus the view and run settings, so DuckDB or pandas can
average them however a report needs.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from psplat.eval.metrics import KNOWN_METRICS, MetricSet
from psplat.io.cameras import Camera
from psplat.io.scene import Scene
from psplat.render.base import Renderer

DEFAULT_BUDGETS = (5.0, 10.0, 25.0, 50.0, 100.0)
IMAGE_SUFFIXES = ("", ".png", ".jpg", ".jpeg", ".JPG", ".JPEG", ".PNG")

# Column order of results.parquet.
STRING_COLUMNS = (
    "scene", "trainer", "ordering", "terms", "format", "view", "reference", "backend",
    "git_sha", "source_sha256", "tool",
)  # fmt: skip
NUMBER_COLUMNS = ("budget_frac", *KNOWN_METRICS, "fps_web", "wall_time")
INTEGER_COLUMNS = ("budget_splats", "bytes")


@dataclass(frozen=True)
class Budget:
    frac: float
    splats: int


def budgets_for(count: int, percents: Sequence[float]) -> list[Budget]:
    """Budgets in percent to splat counts, by the same rule as the manifest's cuts."""
    out: dict[int, Budget] = {}
    for p in sorted({float(p) for p in percents}):
        if not 0 < p <= 100:
            raise ValueError("budgets must be percentages in (0, 100]")
        k = min(count, max(1, round(count * p / 100))) if count else 0
        out.setdefault(k, Budget(round(p / 100, 6), k))
    return list(out.values())


def budgets_from_manifest(manifest: Mapping[str, Any]) -> list[Budget]:
    return [Budget(float(c["frac"]), int(c["index"])) for c in manifest["cuts"]]


class ImageError(ValueError):
    """A reference photograph is missing or does not match its camera."""


def load_image(path: Path) -> NDArray[np.float32]:
    """An 8-bit image file (PNG, JPEG, ...) as ``(H, W, 3)`` float32 in [0, 1]."""
    from PIL import Image, UnidentifiedImageError

    try:
        with Image.open(path) as img:
            rgb = img.convert("RGB")
            array = np.asarray(rgb, dtype=np.float32) / 255.0
    except (OSError, UnidentifiedImageError) as exc:
        raise ImageError(f"{path}: cannot read image ({exc})") from None
    return np.ascontiguousarray(array)


def reference_photo(folder: Path, camera: Camera) -> NDArray[np.float32]:
    for suffix in IMAGE_SUFFIXES:
        path = folder / f"{camera.name}{suffix}"
        if path.is_file():
            image = load_image(path)
            if image.shape[:2] != (camera.height, camera.width):
                raise ImageError(
                    f"{path} is {image.shape[1]}x{image.shape[0]} but camera {camera.name} is "
                    f"{camera.width}x{camera.height}; use the matching resolution folder "
                    "(for example images_4 with cameras downscaled by 4)"
                )
            return image
    raise ImageError(f"no image for camera {camera.name!r} in {folder}")


ProgressFn = Callable[[int, int], None]


def evaluate(
    scene: Scene,
    cameras: Sequence[Camera],
    renderer: Renderer,
    metrics: MetricSet,
    budgets: Sequence[Budget],
    *,
    images: Path | None = None,
    progress: ProgressFn | None = None,
) -> list[dict[str, Any]]:
    """One row per (budget, view): budget, view and each metric's value."""
    rows: list[dict[str, Any]] = []
    total = len(cameras) * len(budgets)
    done = 0
    for camera in cameras:
        if images is not None:
            reference = reference_photo(images, camera)
        else:
            reference = renderer.render(scene, camera).image
        for budget in budgets:
            start = time.perf_counter()
            if budget.splats == scene.count and images is None:
                test = reference
            else:
                test = renderer.render(scene.take(np.arange(budget.splats)), camera).image
            values = metrics(test, reference)
            rows.append({
                "view": camera.name,
                "budget_frac": budget.frac,
                "budget_splats": budget.splats,
                **values,
                "wall_time": time.perf_counter() - start,
            })  # fmt: skip
            done += 1
            if progress is not None:
                progress(done, total)
    return rows


def summarise(rows: Sequence[Mapping[str, Any]], names: Sequence[str]) -> list[dict[str, Any]]:
    """Mean of each metric per budget, over views; ``inf`` PSNR stays ``inf``."""
    by_budget: dict[int, list[Mapping[str, Any]]] = {}
    for row in rows:
        by_budget.setdefault(int(row["budget_splats"]), []).append(row)
    out = []
    for k in sorted(by_budget):
        group = by_budget[k]
        entry: dict[str, Any] = {"budget_frac": group[0]["budget_frac"], "budget_splats": k}
        for name in names:
            values = [float(r[name]) for r in group]
            entry[name] = math.inf if any(math.isinf(v) for v in values) else float(np.mean(values))
        out.append(entry)
    return out


def write_results(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    """Write rows as Parquet with a fixed schema; absent metrics are null."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    fields = (
        [pa.field(c, pa.string()) for c in STRING_COLUMNS]
        + [pa.field("budget_frac", pa.float64())]
        + [pa.field(c, pa.int64()) for c in INTEGER_COLUMNS]
        + [pa.field(c, pa.float64()) for c in NUMBER_COLUMNS if c != "budget_frac"]
    )
    schema = pa.schema(fields)
    columns = {f.name: [row.get(f.name) for row in rows] for f in fields}
    table = pa.table(columns, schema=schema)
    tmp = path.with_name(path.name + ".tmp")
    pq.write_table(table, tmp)
    tmp.replace(path)
