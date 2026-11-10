"""P1.7: ``psplat viewer``, its local server, and the viewer in a real browser.

The server tests run everywhere. The browser test drives the viewer in
headless Chromium through Playwright (``tests/web/check_viewer.mjs``); it is
opt-in, ``PSPLAT_BROWSER=1``, and needs Node with Playwright importable, plus
splat-transform for the SOG case (``PSPLAT_INTEROP=1``).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from psplat import viewer as vw
from psplat.cli import app

runner = CliRunner()
CHECK_SCRIPT = Path(__file__).parent / "web" / "check_viewer.mjs"


@pytest.fixture(scope="module")
def packed(tmp_path_factory: pytest.TempPathFactory, synthetic_dir: Path) -> Path:
    """Fixture analysed, ordered and packed as PLY: work/{scene.*, cams.json, dist/}."""
    work = tmp_path_factory.mktemp("viewer")
    for name in ("scene.ply", "cams.json"):
        shutil.copy(synthetic_dir / name, work / name)
    for args in (
        ["analyze", str(work / "scene.ply"), "--cameras", str(work / "cams.json"),
         "--backend", "torch", "--device", "cpu"],
        ["order", str(work / "scene.ply")],
        ["pack", str(work / "scene.ordered.ply"), "--ply"],
    ):  # fmt: skip
        result = runner.invoke(app, args)
        assert result.exit_code == 0, result.output
    return work


def get(url: str) -> tuple[int, dict[str, str], bytes]:
    try:
        with urllib.request.urlopen(url, timeout=10) as r:
            return r.status, {k.lower(): v for k, v in r.headers.items()}, r.read()
    except urllib.error.HTTPError as err:
        return err.code, {k.lower(): v for k, v in err.headers.items()}, b""


# Finding the files ----------------------------------------------------------------------------


def test_resolve_finds_scene_manifest_and_cameras(packed: Path) -> None:
    served = vw.resolve(packed / "dist")
    assert served.scene.name == "scene.ordered.ply"
    assert served.manifest is not None and served.manifest.name == "scene.psplat.json"
    assert served.cameras == (packed / "cams.json").resolve()  # one folder up from dist
    # A SOG in the folder is preferred over the PLY.
    sog = packed / "dist" / "scene.sog"
    sog.write_bytes(b"stand-in")
    try:
        assert vw.resolve(packed / "dist").scene.name == "scene.sog"
    finally:
        sog.unlink()
    assert "scene=data/scene.ordered.ply" in served.query()
    assert (
        "manifest=data/manifest.json" in served.query()
        and "cameras=data/cams.json" in served.query()
    )


def test_resolve_explains_what_it_cannot_show(packed: Path, tmp_path: Path) -> None:
    with pytest.raises(vw.ViewerError, match=r"no \.sog or \.ply"):
        vw.resolve(tmp_path)
    (tmp_path / "scene.glb").write_bytes(b"x")
    with pytest.raises(vw.ViewerError, match="reorders glTF"):
        vw.resolve(tmp_path / "scene.glb")
    with pytest.raises(vw.ViewerError, match="not found"):
        vw.resolve(packed / "dist", manifest=tmp_path / "nope.json")
    lone = tmp_path / "lone.ply"
    shutil.copy(packed / "dist" / "scene.ordered.ply", lone)
    served = vw.resolve(lone)
    assert served.manifest is None and served.cameras is None
    assert "manifest" not in served.query()


def test_cli_refuses_before_serving(tmp_path: Path) -> None:
    result = runner.invoke(app, ["viewer", str(tmp_path / "missing.sog"), "--no-browser"])
    assert result.exit_code == 1 and "not found" in result.output


# Serving ------------------------------------------------------------------------------------


@pytest.fixture
def server(packed: Path) -> Any:
    srv = vw.start(vw.resolve(packed / "dist"), port=0)
    yield srv
    srv.close()


def test_server_redirects_to_the_viewer_with_the_scene(server: Any) -> None:
    status, _, body = get(server.url)
    assert status == 200  # urllib follows the redirect
    assert b"psplat viewer" in body
    opener = urllib.request.build_opener(_NoRedirect)
    with pytest.raises(urllib.error.HTTPError) as err:
        opener.open(server.url, timeout=10)
    assert err.value.code == 302
    assert err.value.headers["Location"].startswith("/index.html?scene=data/scene.ordered.ply")


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args: Any, **kwargs: Any) -> None:
        return None


@pytest.mark.parametrize(
    ("path", "mime"),
    [
        ("viewer.mjs", "text/javascript"),
        ("vendor/playcanvas-2.23.3.min.mjs", "text/javascript"),
        ("index.html", "text/html"),
        ("data/manifest.json", "application/json"),
        ("data/cams.json", "application/json"),
        ("data/scene.ordered.ply", "application/octet-stream"),
    ],
)
def test_server_sends_files_with_types_browsers_accept(server: Any, path: str, mime: str) -> None:
    status, headers, body = get(server.url + path)
    assert status == 200 and headers["content-type"].startswith(mime) and body
    assert headers["cache-control"] == "no-store"


def test_server_data_is_exactly_the_session_files(server: Any, packed: Path) -> None:
    _, _, body = get(server.url + "data/scene.ordered.ply")
    assert body == (packed / "dist" / "scene.ordered.ply").read_bytes()
    manifest = json.loads(get(server.url + "data/manifest.json")[2])
    assert manifest["count"] == 200
    assert get(server.url + "data/scene.ply")[0] == 404  # not part of the session
    assert get(server.url + "data/..%2F..%2Fscene.ply")[0] == 404
    assert get(server.url + "../../pyproject.toml")[0] == 404


def test_vendored_engine_is_the_pinned_build() -> None:
    root = vw.web_root()
    engine = (root / "vendor" / "playcanvas-2.23.3.min.mjs").read_text(encoding="utf-8")[:400]
    assert "PlayCanvas Engine v2.23.3" in engine
    assert "MIT" in (root / "vendor" / "PLAYCANVAS-LICENSE").read_text(encoding="utf-8")
    assert "playcanvas-2.23.3.min.mjs" in (root / "viewer.mjs").read_text(encoding="utf-8")


# In a browser -------------------------------------------------------------------------------


def _browser_ready() -> bool:
    if os.environ.get("PSPLAT_BROWSER") != "1" or shutil.which("node") is None:
        return False
    probe = subprocess.run(
        ["node", "-e", "require('playwright')"], capture_output=True, check=False
    )
    return probe.returncode == 0


needs_browser = pytest.mark.skipif(
    not _browser_ready(), reason="set PSPLAT_BROWSER=1 with Node and Playwright to run"
)


def drive(url: str) -> dict[str, Any]:
    # The full Chromium build, not the headless shell: only a full browser
    # fetches /favicon.ico, which is how a 404 slipped past these tests once.
    out = subprocess.run(["node", str(CHECK_SCRIPT), url, "--channel", "chromium"],
                         capture_output=True, text=True, check=True, timeout=300)  # fmt: skip
    report: dict[str, Any] = json.loads(out.stdout)
    assert report.get("error") is None, report.get("error")
    assert not [line for line in report["logs"] if line.startswith(("error", "pageerror", "http "))]
    return report


def check_report(report: dict[str, Any], *, snaps: bool) -> None:
    cuts = [c["index"] for c in report["initial"]["cuts"]]
    assert report["initial"]["count"] == 200 and cuts == [10, 20, 50, 100, 150, 200]
    budgets = report["budgets"]
    assert [b["budget"] for b in budgets] == cuts
    assert [b["stats"]["slider"] for b in budgets] == list(range(len(cuts)))
    assert budgets[0]["stats"]["splats"] == "10 of 200"
    # More splats, more light: the slider visibly changes what is drawn.
    brightness = [b["image"]["mean"] for b in budgets]
    assert brightness == sorted(brightness) and brightness[-1] > 1.1 * brightness[0]
    # Between cut points: exact for PLY, snapped up for SOG.
    assert report["between"]["budget"] == (20 if snaps else report["between"]["requested"])
    # The wipe: left half is the budget, right half the whole scene, pixel for pixel.
    smallest, full = budgets[0]["image"], budgets[-1]["image"]
    assert report["split"]["meanLeft"] == pytest.approx(smallest["meanLeft"], abs=1e-9)
    assert report["split"]["meanRight"] == pytest.approx(full["meanRight"], abs=1e-9)
    assert report["splitFull"] == report["unsplitFull"]
    # The projected-size threshold hides more as it rises, and everything at 1000 px.
    lit = [t["image"]["lit"] for t in report["threshold"]]
    assert lit == sorted(lit, reverse=True) and lit[0] > 0.3 and lit[-1] == 0
    assert report["views"] == [True, "view_02"]


@needs_browser
def test_viewer_renders_budgets_from_a_ply(packed: Path) -> None:
    srv = vw.start(vw.resolve(packed / "dist" / "scene.ordered.ply"), port=0)
    try:
        check_report(drive(srv.url), snaps=False)
    finally:
        srv.close()


@needs_browser
@pytest.mark.skipif(os.environ.get("PSPLAT_INTEROP") != "1", reason="needs splat-transform")
def test_viewer_renders_a_banded_sog_like_the_ply(packed: Path, tmp_path: Path) -> None:
    out = tmp_path / "dist"
    result = runner.invoke(
        app, ["pack", str(packed / "scene.ordered.ply"), "--sog", "--out", str(out)]
    )
    assert result.exit_code == 0, result.output
    shutil.copy(packed / "cams.json", tmp_path / "cams.json")
    srv = vw.start(vw.resolve(out / "scene.sog"), port=0)
    try:
        sog = drive(srv.url)
    finally:
        srv.close()
    check_report(sog, snaps=True)
    assert sog["initial"]["format"] == "sog"
    srv = vw.start(vw.resolve(packed / "dist" / "scene.ordered.ply"), port=0)
    try:
        ply = drive(srv.url)
    finally:
        srv.close()
    # Each SOG cut holds the PLY prefix's splats, so what each budget adds over
    # the smallest is the same in both, and so is the left half's brightness.
    # Absolute brightness is not compared: SOG clusters spherical harmonics
    # from a random start, which shifts one large splat's colour by a few
    # levels from run to run (measured noise: under 4% on the left half).
    assert gains(sog) == pytest.approx(gains(ply), abs=0.5)
    assert left_means(sog) == pytest.approx(left_means(ply), rel=0.08, abs=0.15)

    # Negative control: the same scene as splat-transform's command line writes
    # SOG, Morton-sorted. Its budgets hold other splats, and the checks above
    # must tell (measured: gains off by up to 40, left half by up to 28%).
    morton = tmp_path / "morton"
    morton.mkdir()
    shutil.copy(out / "scene.psplat.json", morton / "scene.psplat.json")
    from psplat.pack import splat_transform

    node, _ = splat_transform.find_node()
    cli = splat_transform.package_dir() / "bin" / "cli.mjs"
    subprocess.run([node, str(cli), "-w", "-q", "-g", "cpu", str(packed / "scene.ordered.ply"),
                    str(morton / "scene.sog")], check=True, capture_output=True)  # fmt: skip
    srv = vw.start(vw.resolve(morton / "scene.sog", cameras=packed / "cams.json"), port=0)
    try:
        scrambled = drive(srv.url)
    finally:
        srv.close()
    assert gains(scrambled) != pytest.approx(gains(ply), abs=0.5)


def gains(report: dict[str, Any]) -> list[float]:
    means = [b["image"]["mean"] for b in report["budgets"]]
    return [m - means[0] for m in means]


def left_means(report: dict[str, Any]) -> list[float]:
    return [b["image"]["meanLeft"] for b in report["budgets"]]
