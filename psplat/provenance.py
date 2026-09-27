"""Provenance for every result: what ran, on what, from which code.

Handoff §5.4: no result without provenance. :func:`collect` is embedded in
``weights.npz`` metadata and written as ``run.json`` by experiment runners.
"""

from __future__ import annotations

import hashlib
import os
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from psplat import __version__

_REPO = Path(__file__).resolve().parent.parent


def sha256_file(path: str | Path, chunk: int = 1 << 22) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        while block := f.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def git_state() -> dict[str, Any] | None:
    """Commit and dirty flag of the psplat checkout, or ``None`` if not a checkout."""
    if not (_REPO / ".git").exists():
        return None
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=_REPO, capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain", "--untracked-files=no"],
                cwd=_REPO, capture_output=True, text=True, check=True,
            ).stdout.strip()
        )  # fmt: skip
    except (OSError, subprocess.CalledProcessError):
        return None
    return {"sha": sha, "dirty": dirty}


def hardware() -> dict[str, Any]:
    info: dict[str, Any] = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": sys.version.split()[0],
        "cpu_count": os.cpu_count(),
    }
    try:
        import torch

        info["torch"] = torch.__version__
        if torch.cuda.is_available():
            info["cuda"] = torch.version.cuda
            info["gpu"] = torch.cuda.get_device_name(0)
    except ImportError:  # pragma: no cover - torch is a core dependency
        pass
    return info


def collect(**extra: Any) -> dict[str, Any]:
    """Provenance record: tool version, git state, hardware, time, plus ``extra``."""
    return {
        "tool": f"psplat {__version__}",
        "git": git_state(),
        "hardware": hardware(),
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        **extra,
    }
