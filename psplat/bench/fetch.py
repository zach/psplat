"""``psplat bench fetch``: download datasets and extract the scenes asked for.

Local layout: ``<dest>/raw/<archive>/<scene>/``, the same shape as the
bucket's ``raw/`` prefix (handoff §4.3), so ``aws s3 sync`` maps one onto the
other. Downloads resume after an interruption, are checked against the size
the registry records, and each extracted scene gets a ``fetch.json``
provenance record. A scene whose ``fetch.json`` matches is skipped.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import urllib.request
import zipfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

from psplat.bench.registry import Archive, Registry, SceneEntry
from psplat.provenance import collect

Log = Callable[[str], None]
_CHUNK = 1 << 22


class FetchError(RuntimeError):
    pass


def download(archive: Archive, target: Path, log: Log = print) -> Path:
    """Download ``archive`` to ``target``, resuming a partial file."""
    target.parent.mkdir(parents=True, exist_ok=True)
    if archive.bytes is not None and target.exists() and target.stat().st_size == archive.bytes:
        log(f"{archive.name}: already downloaded")
        return target
    partial = target.with_name(target.name + ".part")
    have = partial.stat().st_size if partial.exists() else 0
    request = urllib.request.Request(archive.url)
    if have:
        request.add_header("Range", f"bytes={have}-")
        log(f"{archive.name}: resuming at {have:,} bytes")
    else:
        size = f" ({archive.bytes / 1e9:.1f} GB)" if archive.bytes else ""
        log(f"{archive.name}: downloading {archive.url}{size}")
    with urllib.request.urlopen(request) as response:
        if have and response.status != 206:
            have = 0  # server ignored the range; start again
        with partial.open("ab" if have else "wb") as f:
            while block := response.read(_CHUNK):
                f.write(block)
    got = partial.stat().st_size
    if archive.bytes is not None and got != archive.bytes:
        raise FetchError(
            f"{archive.name}: got {got:,} bytes, registry expects {archive.bytes:,}; "
            f"delete {partial} to start over"
        )
    partial.replace(target)
    return target


def _scene_members(zf: zipfile.ZipFile, path: str) -> tuple[str, list[zipfile.ZipInfo]]:
    """Members under ``path/``, allowing for one optional top-level folder in the zip."""
    marker = path.strip("/") + "/"
    for info in zf.infolist():
        name = info.filename
        at = name.find(marker)
        if at == 0 or (at > 0 and name[at - 1] == "/"):
            root = name[:at]
            if root.count("/") <= 1:
                members = [i for i in zf.infolist() if i.filename.startswith(root + marker)]
                return root + marker, members
    raise FetchError(f"no folder {marker!r} in {zf.filename}")


def extract_scene(zip_path: Path, scene: SceneEntry, out_dir: Path) -> int:
    """Extract one scene folder from the archive into ``out_dir``; returns file count."""
    with zipfile.ZipFile(zip_path) as zf:
        prefix, members = _scene_members(zf, scene.path)
        tmp = out_dir.with_name(out_dir.name + ".extracting")
        if tmp.exists():
            shutil.rmtree(tmp)
        count = 0
        for info in members:
            rel = info.filename[len(prefix) :]
            if not rel or info.is_dir():
                continue
            dest = tmp / rel
            if tmp.resolve() not in dest.resolve().parents:
                raise FetchError(f"refusing to extract {info.filename} outside the scene folder")
            dest.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as src, dest.open("wb") as dst:
                shutil.copyfileobj(src, dst, _CHUNK)
            count += 1
    if out_dir.exists():
        shutil.rmtree(out_dir)
    tmp.replace(out_dir)
    return count


def fetch(
    registry: Registry,
    names: list[str],
    dest: Path,
    *,
    s3_bucket: str | None = None,
    log: Log = print,
) -> list[dict[str, Any]]:
    """Fetch the scenes (or sets) in ``names`` into ``dest``; optionally mirror to S3."""
    records = []
    for scene in registry.resolve(names):
        archive = registry.archives[scene.archive]
        out_dir = dest / "raw" / archive.name / scene.name
        stamp = out_dir / "fetch.json"
        if stamp.exists():
            old = json.loads(stamp.read_text(encoding="utf-8"))
            if old.get("url") == archive.url and old.get("bytes") == archive.bytes:
                log(f"{scene.name}: already fetched")
                records.append(old)
                continue
        zip_path = download(archive, dest / "_downloads" / f"{archive.name}.zip", log)
        files = extract_scene(zip_path, scene, out_dir)
        record = collect(
            command="bench fetch",
            scene=scene.name,
            archive=archive.name,
            url=archive.url,
            bytes=archive.bytes,
            licence=archive.licence,
            files=files,
        )
        stamp.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        log(f"{scene.name}: {files} files in {out_dir}")
        records.append(record)
        if s3_bucket:
            uri = f"s3://{s3_bucket}/raw/{archive.name}/{scene.name}/"
            log(f"{scene.name}: syncing to {uri}")
            subprocess.run(
                ["aws", "s3", "sync", str(out_dir), uri, "--only-show-errors"], check=True
            )
    return records
