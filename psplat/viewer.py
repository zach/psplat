"""``psplat viewer``: serve the web viewer and one scene locally (P1.7, #26).

The viewer is static files (``engines/web`` in the repository, ``psplat/_web``
in the wheel) with the PlayCanvas engine vendored, so nothing is built or
downloaded. This module serves them with Python's ``http.server`` on
127.0.0.1, together with the scene, its manifest and cameras under
``/data/``, and opens the default browser.
"""

from __future__ import annotations

import http.server
import threading
import urllib.parse
import webbrowser
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from psplat.pack.pack import output_stem

# File types the browser must see exactly; Windows' registry can otherwise
# map .js to text/plain, which browsers refuse for module scripts.
MIME_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".mjs": "text/javascript; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".json": "application/json",
    ".css": "text/css; charset=utf-8",
    ".ply": "application/octet-stream",
    ".sog": "application/octet-stream",
    ".webp": "image/webp",
}
SCENE_SUFFIXES = (".sog", ".ply")


class ViewerError(ValueError):
    """The scene or its companions cannot be served; the message says why."""


def web_root() -> Path:
    """The viewer's static files: packaged in the wheel, or the repository's ``engines/web``."""
    packaged = Path(__file__).with_name("_web")
    if (packaged / "index.html").is_file():
        return packaged
    repo = Path(__file__).resolve().parents[1] / "engines" / "web"
    if (repo / "index.html").is_file():
        return repo
    raise ViewerError("the web viewer's files are missing from this installation")


@dataclass
class Served:
    """The files of one viewing session, as served under ``/data/``."""

    scene: Path
    manifest: Path | None = None
    cameras: Path | None = None
    extra: dict[str, str] = field(default_factory=dict)  # more query parameters

    def files(self) -> dict[str, Path]:
        out = {self.scene.name: self.scene}
        if self.manifest is not None:
            out["manifest.json"] = self.manifest
        if self.cameras is not None:
            out["cams.json"] = self.cameras
        return out

    def query(self) -> str:
        q = {"scene": f"data/{urllib.parse.quote(self.scene.name)}"}
        if self.manifest is not None:
            q["manifest"] = "data/manifest.json"
        if self.cameras is not None:
            q["cameras"] = "data/cams.json"
        q.update(self.extra)
        return urllib.parse.urlencode(q, safe="/")


def resolve(path: Path, manifest: Path | None = None, cameras: Path | None = None) -> Served:
    """Find the scene file and, when not given, its manifest and cameras nearby.

    ``path`` may be a scene file or a folder such as ``dist``, in which case its
    SOG is preferred over an ordered PLY. The manifest defaults to
    ``<stem>.psplat.json`` beside the scene; cameras to ``cams.json`` beside it
    or one folder up, where ``psplat pack`` leaves the source files.
    """
    if path.is_dir():
        found = sorted(path.glob("*.sog")) or sorted(path.glob("*.ordered.ply"))
        found = found or sorted(path.glob("*.ply"))
        if not found:
            raise ViewerError(f"no .sog or .ply scene in {path}")
        path = found[0]
    if not path.is_file():
        raise ViewerError(f"{path} not found")
    if path.suffix.lower() not in SCENE_SUFFIXES:
        raise ViewerError(
            f"{path.name}: the viewer shows SOG and PLY. SPZ and glTF keep psplat's order too, "
            "but PlayCanvas reorders glTF splats when it loads them."
        )
    if manifest is None:
        candidate = path.with_name(f"{output_stem(path)}.psplat.json")
        manifest = candidate if candidate.is_file() else None
    elif not manifest.is_file():
        raise ViewerError(f"{manifest} not found")
    if cameras is None:
        for folder in (path.parent, path.parent.parent):
            if (folder / "cams.json").is_file():
                cameras = folder / "cams.json"
                break
    elif not cameras.is_file():
        raise ViewerError(f"{cameras} not found")
    return Served(path.resolve(), manifest and manifest.resolve(), cameras and cameras.resolve())


class _Handler(http.server.SimpleHTTPRequestHandler):
    root: Path
    data: Mapping[str, Path]
    landing: str

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, directory=str(self.root), **kwargs)

    def do_GET(self) -> None:
        path = urllib.parse.urlsplit(self.path).path
        if path == "/":
            self.send_response(302)
            self.send_header("Location", self.landing)
            self.end_headers()
            return
        if path.startswith("/data/"):
            name = urllib.parse.unquote(path[len("/data/") :])
            target = self.data.get(name)
            if target is None:
                self.send_error(404, "not part of this viewing session")
                return
            self._send_file(target)
            return
        super().do_GET()

    def _send_file(self, target: Path) -> None:
        try:
            body = target.read_bytes()
        except OSError:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", self.guess_type(str(target)))
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def guess_type(self, path: str | Any) -> str:
        suffix = Path(str(path)).suffix.lower()
        return MIME_TYPES.get(suffix) or super().guess_type(path)

    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def log_message(self, format: str, *args: Any) -> None:
        pass  # quiet; errors still reach the browser


@dataclass
class ViewerServer:
    """A running viewer server; :meth:`close` stops it."""

    httpd: http.server.ThreadingHTTPServer
    url: str
    thread: threading.Thread | None = None

    def close(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()
        if self.thread is not None:
            self.thread.join(timeout=5)


def start(served: Served, *, port: int = 0, host: str = "127.0.0.1",
          background: bool = True) -> ViewerServer:  # fmt: skip
    """Start serving; ``port=0`` picks a free port. Background servers run in a thread."""
    handler = type(
        "PsplatViewerHandler",
        (_Handler,),
        {"root": web_root(), "data": served.files(), "landing": f"/index.html?{served.query()}"},
    )
    httpd = http.server.ThreadingHTTPServer((host, port), handler)
    url = f"http://{host}:{httpd.server_address[1]}/"
    server = ViewerServer(httpd, url)
    if background:
        server.thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        server.thread.start()
    return server


def open_browser(url: str) -> bool:
    try:
        return webbrowser.open(url)
    except webbrowser.Error:
        return False
