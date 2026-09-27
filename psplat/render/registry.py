"""Backend lookup and ``--backend auto`` selection."""

from __future__ import annotations

import importlib.util
import platform
import sys
from dataclasses import dataclass

from psplat.render.base import BackendUnavailable, Renderer

ISSUES = "https://github.com/zach/psplat/issues/"
BACKEND_NAMES = ("auto", "torch", "gsplat", "msplat", "null")

# Backends that are specified but not written yet, and the issue tracking each.
_PLANNED = {"gsplat": 23, "msplat": 24}


@dataclass(frozen=True)
class Selection:
    renderer: Renderer
    reason: str  # why this backend, for the one-line notice


def _importable(module: str) -> bool:
    return importlib.util.find_spec(module) is not None


def _cuda_available() -> bool:
    import torch

    return bool(torch.cuda.is_available())


def get_backend(name: str, *, device: str | None = None) -> Renderer:
    """Construct a backend by name. ``auto`` is resolved by :func:`select_backend`."""
    if name == "auto":
        return select_backend(device=device).renderer
    if name == "torch":
        from psplat.render.torch_backend import TorchRenderer

        return TorchRenderer(device=device)
    if name == "null":
        from psplat.render.null_backend import NullRenderer

        return NullRenderer()
    if name in _PLANNED:
        raise BackendUnavailable(
            f"the {name} backend is not implemented yet; it is tracked in "
            f"{ISSUES}{_PLANNED[name]}. Use --backend torch meanwhile."
        )
    raise BackendUnavailable(f"unknown backend {name!r}; choose from {', '.join(BACKEND_NAMES)}")


def select_backend(*, device: str | None = None) -> Selection:
    """Pick gsplat if usable with CUDA, else msplat on Apple Silicon, else torch.

    Planned backends that are installed but not implemented are skipped, and
    the reason says so, so nobody wonders why their GPU is idle.
    """
    notes = []
    if _importable("gsplat") and _cuda_available():
        if "gsplat" not in _PLANNED:
            return Selection(get_backend("gsplat", device=device), "gsplat with CUDA found")
        notes.append(f"gsplat is installed but its backend lands in {ISSUES}{_PLANNED['gsplat']}")
    if sys.platform == "darwin" and platform.machine() == "arm64" and _importable("msplat"):
        if "msplat" not in _PLANNED:
            return Selection(get_backend("msplat", device=device), "msplat on Apple Silicon")
        notes.append(f"msplat is installed but its backend lands in {ISSUES}{_PLANNED['msplat']}")
    reason = "portable PyTorch backend"
    if notes:
        reason += " (" + "; ".join(notes) + ")"
    return Selection(get_backend("torch", device=device), reason)
