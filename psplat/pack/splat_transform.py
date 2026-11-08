"""Running PlayCanvas splat-transform, the writer behind SOG, SPZ and glTF.

psplat shells out to splat-transform (MIT) under Node.js rather than
reimplementing those formats. The version is pinned
(:data:`psplat.pack.SPLAT_TRANSFORM_VERSION`), because whether a writer keeps
splat order is a property of the version (``docs/formats.md``).

The pinned package is installed once with npm into psplat's cache folder
and reused. ``PSPLAT_SPLAT_TRANSFORM_DIR`` can name an existing install of
the same version instead, for offline machines.

- SPZ and glTF go through splat-transform's command line, which keeps the
  input order for both.
- SOG goes through ``write_sog.mjs`` beside this module, which calls the
  library's ``writeSog`` with an explicit texel order. The command line
  would Morton-sort the splats and lose psplat's order.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from psplat.pack import SPLAT_TRANSFORM_VERSION

PACKAGE = "@playcanvas/splat-transform"
SOG_WRITER = Path(__file__).with_name("write_sog.mjs")
NODE_HELP = (
    "Node.js was not found. SOG, SPZ and glTF are written with PlayCanvas "
    "splat-transform, which runs on Node.js. Install the LTS release from "
    "https://nodejs.org (the Windows installer adds it to PATH), open a new "
    "terminal, and run this again. PLY needs no Node."
)


class ToolUnavailable(RuntimeError):
    """Node.js or splat-transform cannot be used; the message says what to do."""


def cache_dir() -> Path:
    """psplat's per-user cache folder."""
    if override := os.environ.get("PSPLAT_CACHE"):
        return Path(override)
    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
        return base / "psplat" / "cache"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Caches" / "psplat"
    return Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "psplat"


def find_node() -> tuple[str, str]:
    """Paths to ``node`` and ``npm``, or :class:`ToolUnavailable`."""
    node, npm = shutil.which("node"), shutil.which("npm")
    if node is None or npm is None:
        raise ToolUnavailable(NODE_HELP)
    return node, npm


def _installed_version(package_dir: Path) -> str | None:
    try:
        meta = json.loads((package_dir / "package.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    version = meta.get("version")
    return version if isinstance(version, str) else None


def package_dir(*, install: bool = True, notice: bool = True) -> Path:
    """The pinned splat-transform package folder, installing it if needed."""
    override = os.environ.get("PSPLAT_SPLAT_TRANSFORM_DIR")
    if override:
        path = Path(override)
        found = _installed_version(path)
        if found != SPLAT_TRANSFORM_VERSION:
            raise ToolUnavailable(
                f"PSPLAT_SPLAT_TRANSFORM_DIR={path} holds splat-transform {found or '(none)'}; "
                f"psplat needs {SPLAT_TRANSFORM_VERSION}"
            )
        return path
    prefix = cache_dir() / f"splat-transform-{SPLAT_TRANSFORM_VERSION}"
    path = prefix / "node_modules" / "@playcanvas" / "splat-transform"
    if _installed_version(path) == SPLAT_TRANSFORM_VERSION:
        return path
    if not install:
        raise ToolUnavailable(f"splat-transform {SPLAT_TRANSFORM_VERSION} is not installed")
    _, npm = find_node()
    if notice:
        print(
            f"installing splat-transform {SPLAT_TRANSFORM_VERSION} into {prefix} "
            "(once; needs internet)",
            file=sys.stderr,
        )
    prefix.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [npm, "install", "--prefix", str(prefix), "--no-audit", "--no-fund", "--silent",
         f"{PACKAGE}@{SPLAT_TRANSFORM_VERSION}"],
        capture_output=True, text=True, check=False,
    )  # fmt: skip
    if result.returncode != 0 or _installed_version(path) != SPLAT_TRANSFORM_VERSION:
        detail = (result.stderr or result.stdout).strip().splitlines()[-3:]
        raise ToolUnavailable(
            f"npm could not install {PACKAGE}@{SPLAT_TRANSFORM_VERSION}"
            + (": " + " / ".join(detail) if detail else "")
            + ". Check the internet connection, or set PSPLAT_SPLAT_TRANSFORM_DIR "
            "to an existing install of that version."
        )
    return path


def _run(cmd: list[str], what: str) -> None:
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip().splitlines()[-5:]
        raise RuntimeError(f"splat-transform failed writing {what}: " + " / ".join(detail))


def convert(source: Path, target: Path) -> None:
    """``source`` PLY to ``target`` with the command line; for SPZ and glTF (.glb)."""
    node, _ = find_node()
    cli = package_dir() / "bin" / "cli.mjs"
    _run([node, str(cli), "-w", "-q", str(source), str(target)], target.name)


def read_back(source: Path, target_ply: Path) -> None:
    """Decode any format splat-transform reads into a PLY; used by tests and checks."""
    convert(source, target_ply)


def write_sog(source: Path, target: Path, order_file: Path) -> None:
    """SOG with splats in texel order ``order_file`` (little-endian uint32)."""
    node, _ = find_node()
    _run([node, str(SOG_WRITER), str(package_dir()), str(source), str(target), str(order_file)],
         target.name)  # fmt: skip
