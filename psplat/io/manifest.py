"""The ``scene.psplat.json`` manifest (handoff §3.4), version 1.

Written by ``psplat order`` next to the ordered PLY. Engines need only
``cuts[].index``: to show a budget, render the first ``index`` splats.
Everything else is for reports and reproducibility.

The schema is ``manifest.schema.json`` beside this module. It is frozen at
the end of P1 (handoff §8.7). Any later change bumps ``version`` and is asked
about first. :func:`validate` checks the schema plus what JSON Schema cannot
express: cuts strictly increasing in ``frac``, ``index`` never decreasing and
never beyond ``count``.

Example::

    {
      "version": 1,
      "source_sha256": "9f2c…",
      "count": 3512000,
      "ordering": {"method": "contribution", "terms": ["contribution"]},
      "cuts": [{"frac": 0.05, "index": 175600}, …, {"frac": 1.0, "index": 3512000}],
      "provenance": {"tool": "psplat 0.1.0", "backend": "torch on cuda", "date": "…"}
    }
"""

from __future__ import annotations

import functools
import json
from collections.abc import Sequence
from importlib import resources
from pathlib import Path
from typing import Any

import jsonschema

from psplat.pack.layout import band_bounds

VERSION = 1
SCHEMA_FILE = "manifest.schema.json"


class ManifestError(ValueError):
    """A manifest does not match the version 1 schema or its rules."""


@functools.cache
def schema() -> dict[str, Any]:
    text = resources.files("psplat.io").joinpath(SCHEMA_FILE).read_text(encoding="utf-8")
    loaded: dict[str, Any] = json.loads(text)
    return loaded


def cuts_for(count: int, cuts_percent: Sequence[float]) -> list[dict[str, Any]]:
    """Cut entries for budgets in percent, sorted, with a final 100% cut.

    ``index`` is ``round(count · percent / 100)``, at least 1 for a non-empty
    scene, exactly as the banded SOG layout draws its band edges
    (:func:`psplat.pack.layout.band_bounds`), so every cut is an exact prefix
    of a banded SOG too.
    """
    band_bounds(count, cuts_percent)  # validates the range, (0, 100]
    percents = sorted({float(c) for c in cuts_percent} | {100.0})
    out = []
    for p in percents:
        index = min(count, max(1, round(count * p / 100))) if count else 0
        out.append({"frac": round(p / 100, 6), "index": index})
    return out


def build(
    *,
    source_sha256: str,
    count: int,
    method: str,
    terms: Sequence[str],
    cuts_percent: Sequence[float],
    provenance: dict[str, Any],
    seed: int | None = None,
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    ordering: dict[str, Any] = {"method": method, "terms": list(terms)}
    if seed is not None:
        ordering["seed"] = seed
    if params:
        ordering["params"] = params
    manifest = {
        "version": VERSION,
        "source_sha256": source_sha256,
        "count": count,
        "ordering": ordering,
        "cuts": cuts_for(count, cuts_percent),
        "provenance": provenance,
    }
    validate(manifest)
    return manifest


def validate(manifest: Any) -> None:
    """Raise :class:`ManifestError` unless ``manifest`` is a valid version 1 manifest."""
    if isinstance(manifest, dict) and manifest.get("version") != VERSION:
        raise ManifestError(
            f"manifest version {manifest.get('version')!r}; this psplat reads version {VERSION}"
        )
    validator = jsonschema.Draft202012Validator(schema())
    errors = sorted(validator.iter_errors(manifest), key=lambda e: list(e.absolute_path))
    if errors:
        first = errors[0]
        where = "/".join(str(p) for p in first.absolute_path) or "(top level)"
        more = f" (and {len(errors) - 1} more)" if len(errors) > 1 else ""
        raise ManifestError(f"manifest {where}: {first.message}{more}")
    count = manifest["count"]
    previous_frac, previous_index = 0.0, 0
    for k, cut in enumerate(manifest["cuts"]):
        if cut["frac"] <= previous_frac:
            raise ManifestError(f"manifest cuts/{k}: frac must increase, got {cut['frac']}")
        if cut["index"] < previous_index:
            raise ManifestError(f"manifest cuts/{k}: index must not decrease, got {cut['index']}")
        if cut["index"] > count:
            raise ManifestError(f"manifest cuts/{k}: index {cut['index']} exceeds count {count}")
        previous_frac, previous_index = cut["frac"], cut["index"]


def write(manifest: dict[str, Any], path: str | Path) -> None:
    """Validate and write atomically, as UTF-8 with LF newlines."""
    validate(manifest)
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")
    tmp.replace(path)


def read(path: str | Path) -> dict[str, Any]:
    try:
        manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ManifestError(f"{path}: not JSON ({exc})") from None
    try:
        validate(manifest)
    except ManifestError as exc:
        raise ManifestError(f"{path}: {exc}") from None
    result: dict[str, Any] = manifest
    return result
