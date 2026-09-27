"""The scene registry in ``scenes.yaml``."""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import Any

import yaml


class RegistryError(ValueError):
    """The registry is malformed or a name is not in it."""


@dataclass(frozen=True)
class Archive:
    name: str
    url: str
    bytes: int | None
    verified: str | None
    licence: str


@dataclass(frozen=True)
class SceneEntry:
    name: str
    archive: str
    path: str
    data_factor: int


@dataclass(frozen=True)
class Trainer:
    name: str
    kind: str  # "gsplat" or "msplat"
    steps: int
    strategy: str | None = None
    args: tuple[str, ...] = ()


@dataclass(frozen=True)
class Registry:
    archives: dict[str, Archive]
    sets: dict[str, list[str]]
    scenes: dict[str, SceneEntry]
    trainers: dict[str, Trainer]
    reference: dict[str, Any] = field(default_factory=dict)

    def scene(self, name: str) -> SceneEntry:
        if name not in self.scenes:
            raise RegistryError(f"unknown scene {name!r}; known: {', '.join(sorted(self.scenes))}")
        return self.scenes[name]

    def trainer(self, name: str) -> Trainer:
        if name not in self.trainers:
            known = ", ".join(sorted(self.trainers))
            raise RegistryError(f"unknown trainer {name!r}; known: {known}")
        return self.trainers[name]

    def resolve(self, names: list[str]) -> list[SceneEntry]:
        """Scene entries for a mix of set names and scene names, in order, deduplicated."""
        out: list[SceneEntry] = []
        for name in names:
            members = self.sets.get(name, [name])
            for member in members:
                entry = self.scene(member)
                if entry not in out:
                    out.append(entry)
        return out


def load_registry(path: str | Path | None = None) -> Registry:
    """Load and validate the registry; defaults to the packaged ``scenes.yaml``."""
    if path is None:
        text = resources.files("psplat.bench").joinpath("scenes.yaml").read_text(encoding="utf-8")
    else:
        text = Path(path).read_text(encoding="utf-8")
    doc = yaml.safe_load(text) or {}
    try:
        archives = {
            k: Archive(k, v["url"], v.get("bytes"), v.get("verified"), v["licence"])
            for k, v in doc["archives"].items()
        }
        scenes = {
            k: SceneEntry(k, v["archive"], v["path"], int(v["data_factor"]))
            for k, v in doc["scenes"].items()
        }
        trainers = {
            k: Trainer(k, v["kind"], int(v["steps"]), v.get("strategy"), tuple(v.get("args", ())))
            for k, v in doc["trainers"].items()
        }
        sets = {k: list(v) for k, v in doc["sets"].items()}
    except (KeyError, TypeError) as exc:
        raise RegistryError(f"scenes.yaml is malformed near {exc}") from None
    for scene in scenes.values():
        if scene.archive not in archives:
            raise RegistryError(f"scene {scene.name} names unknown archive {scene.archive}")
    for set_name, members in sets.items():
        for member in members:
            if member not in scenes:
                raise RegistryError(f"set {set_name} names unknown scene {member}")
    for trainer in trainers.values():
        if trainer.kind not in ("gsplat", "msplat"):
            raise RegistryError(f"trainer {trainer.name} has unknown kind {trainer.kind}")
    return Registry(archives, sets, scenes, trainers, doc.get("reference") or {})
