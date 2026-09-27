"""Shared test fixtures."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

SYNTHETIC = Path(__file__).parent / "fixtures" / "synthetic"


@pytest.fixture(scope="session")
def synthetic_dir() -> Path:
    return SYNTHETIC


@pytest.fixture(scope="session")
def oracles() -> dict[str, Any]:
    data: dict[str, Any] = json.loads((SYNTHETIC / "oracles.json").read_text(encoding="utf-8"))
    return data
