"""P0.1: the CLI exposes the whole v0.1 surface from handoff §3.3."""

from __future__ import annotations

from typer.testing import CliRunner

from psplat import __version__
from psplat.cli import app

runner = CliRunner()

V01_COMMANDS = [
    "analyze",
    "order",
    "pack",
    "lod",
    "eval",
    "bench",
    "report",
    "study",
    "viewer",
]


def test_help_lists_every_v01_subcommand() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for command in V01_COMMANDS:
        assert command in result.output


def test_version() -> None:
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert __version__ in result.output


def test_stub_names_its_tracking_issue_and_exits_2() -> None:
    result = runner.invoke(app, ["lod", "scene.ordered.ply", "--distances", "1,2,4"])
    assert result.exit_code == 2
    assert "issues/40" in result.output
