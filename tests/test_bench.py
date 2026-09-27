"""P0.11: scene registry, dataset fetch and the E0 training wrappers.

The real trainers need a GPU (gsplat) or Apple Silicon (msplat), so the
wrappers are exercised against stand-ins that follow the documented
interfaces: a fake ``simple_trainer.py`` that prints tyro-style help and
writes a gsplat-shaped checkpoint and stats file, a fake gsplat ``Parser``,
and a fake ``msplat`` module with the API from msplat's PyPI page. These
test psplat's plumbing, not the trainers; the first real run on the AWS box
and the Mac is what confirms the interfaces themselves.
"""

from __future__ import annotations

import json
import sys
import textwrap
import types
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from typer.testing import CliRunner

from psplat.bench.fetch import FetchError, fetch
from psplat.bench.registry import Archive, Registry, RegistryError, SceneEntry, load_registry
from psplat.bench.train import (
    TrainError,
    compare_to_reference,
    gsplat_cameras,
    gsplat_command,
    missing_flags,
    scene_from_gsplat_ckpt,
    train_gsplat,
    train_msplat,
)
from psplat.cli import app
from psplat.io.cameras import read_cameras_json
from psplat.io.ply import read_ply, write_ply
from psplat.io.scene import Scene

COLMAP = Path(__file__).parent / "fixtures" / "colmap_tiny" / "bin"


# Registry ---------------------------------------------------------------------------------


def test_packaged_registry_is_consistent() -> None:
    reg = load_registry()
    assert [s.name for s in reg.resolve(["mipnerf360-core"])] == [
        "garden",
        "bicycle",
        "counter",
        "room",
    ]
    assert reg.scene("garden").data_factor == 4 and reg.scene("room").data_factor == 2
    assert reg.trainer("gsplat-mcmc-1m").args == ("--strategy.cap-max", "1000000")
    assert reg.archives["mipnerf360"].bytes == 12535427936
    # Sets and scene names mix, without duplicates.
    assert len(reg.resolve(["mipnerf360-core", "garden", "truck"])) == 5


def test_registry_errors(tmp_path: Path) -> None:
    reg = load_registry()
    with pytest.raises(RegistryError, match="unknown scene"):
        reg.resolve(["nowhere"])
    with pytest.raises(RegistryError, match="unknown trainer"):
        reg.trainer("nerf")
    bad = tmp_path / "bad.yaml"
    bad.write_text("archives: {}\nsets: {s: [x]}\nscenes: {}\ntrainers: {}\n")
    with pytest.raises(RegistryError, match="unknown scene x"):
        load_registry(bad)


# Fetch -------------------------------------------------------------------------------------


def _tiny_registry(zip_path: Path, *, size: int | None, path: str = "garden") -> Registry:
    archive = Archive("tiny", zip_path.as_uri(), size, None, "test data")
    scene = SceneEntry("garden", "tiny", path, 4)
    return Registry({"tiny": archive}, {"core": ["garden"]}, {"garden": scene}, {}, {})


def _zip(path: Path, members: dict[str, bytes]) -> Path:
    with zipfile.ZipFile(path, "w") as zf:
        for name, data in members.items():
            zf.writestr(name, data)
    return path


@pytest.mark.parametrize("root", ["", "360_v2/"])
def test_fetch_extracts_only_the_scene_and_is_idempotent(tmp_path: Path, root: str) -> None:
    z = _zip(tmp_path / "a.zip", {
        f"{root}garden/images_4/a.png": b"png", f"{root}garden/sparse/0/cameras.bin": b"c",
        f"{root}bicycle/images/b.png": b"other",
    })  # fmt: skip
    reg = _tiny_registry(z, size=z.stat().st_size)
    logs: list[str] = []
    fetch(reg, ["core"], tmp_path / "data", log=logs.append)
    scene_dir = tmp_path / "data" / "raw" / "tiny" / "garden"
    assert (scene_dir / "images_4" / "a.png").read_bytes() == b"png"
    assert not (tmp_path / "data" / "raw" / "tiny" / "bicycle").exists()
    record = json.loads((scene_dir / "fetch.json").read_text())
    assert record["files"] == 2 and record["licence"] == "test data"
    fetch(reg, ["garden"], tmp_path / "data", log=logs.append)
    assert logs[-1] == "garden: already fetched"


def test_fetch_restarts_when_the_server_ignores_ranges(tmp_path: Path) -> None:
    z = _zip(tmp_path / "a.zip", {"garden/x": b"12345"})
    partial = tmp_path / "data" / "_downloads" / "tiny.zip.part"
    partial.parent.mkdir(parents=True)
    partial.write_bytes(z.read_bytes()[:10])  # file:// ignores Range, so this must restart
    fetch(_tiny_registry(z, size=z.stat().st_size), ["garden"], tmp_path / "data", log=print)
    assert (tmp_path / "data" / "raw" / "tiny" / "garden" / "x").read_bytes() == b"12345"


def test_fetch_checks_size_and_refuses_path_escapes(tmp_path: Path) -> None:
    z = _zip(tmp_path / "a.zip", {"garden/x": b"1"})
    with pytest.raises(FetchError, match="registry expects"):
        fetch(_tiny_registry(z, size=1), ["garden"], tmp_path / "d1", log=print)
    evil = _zip(tmp_path / "evil.zip", {"garden/ok": b"1", "garden/../../escape": b"x"})
    with pytest.raises(FetchError, match="outside"):
        fetch(_tiny_registry(evil, size=None), ["garden"], tmp_path / "d2", log=print)
    with pytest.raises(FetchError, match="no folder"):
        fetch(_tiny_registry(z, size=None, path="stump"), ["garden"], tmp_path / "d3", log=print)


# gsplat wrapper --------------------------------------------------------------------------

TYRO_HELP = """usage: simple_trainer.py mcmc [-h] [OPTIONS]
  --data-dir STR  --data-factor INT  --result-dir STR  --max-steps INT
  --disable-viewer, --no-disable-viewer  --strategy.cap-max INT  --save-ply
"""


def test_gsplat_command_and_flag_check(tmp_path: Path) -> None:
    reg = load_registry()
    cmd = gsplat_command(reg.trainer("gsplat-mcmc-3m"), reg.scene("counter"),
                         Path("/d/counter"), Path("/r"), Path("/g/examples"), "python")  # fmt: skip
    assert cmd[:3] == ["python", str(Path("/g/examples/simple_trainer.py")), "mcmc"]
    assert cmd[cmd.index("--data_factor") + 1] == "2"
    assert cmd[-2:] == ["--strategy.cap-max", "3000000"]
    assert missing_flags(cmd, TYRO_HELP) == []
    assert missing_flags([*cmd, "--antialiased"], TYRO_HELP) == ["--antialiased"]
    with pytest.raises(TrainError, match="not a gsplat recipe"):
        gsplat_command(reg.trainer("msplat-7k"), reg.scene("garden"), Path(), Path(), Path())


def _ckpt_splats(n: int = 7) -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(0)
    return {
        "means": torch.randn(n, 3, generator=g),
        "scales": torch.randn(n, 3, generator=g),
        "quats": torch.randn(n, 4, generator=g),
        "opacities": torch.randn(n, generator=g),
        "sh0": torch.randn(n, 1, 3, generator=g),
        "shN": torch.randn(n, 15, 3, generator=g),
    }


def test_scene_from_gsplat_checkpoint(tmp_path: Path) -> None:
    splats = _ckpt_splats()
    torch.save({"step": 29999, "splats": splats}, tmp_path / "ckpt.pt")
    scene = scene_from_gsplat_ckpt(tmp_path / "ckpt.pt")
    assert scene.count == 7 and scene.sh_degree == 3
    np.testing.assert_array_equal(scene.sh0, splats["sh0"][:, 0].numpy())
    np.testing.assert_array_equal(scene.opacities, splats["opacities"].numpy())
    del splats["sh0"]
    torch.save({"splats": splats}, tmp_path / "bad.pt")
    with pytest.raises(TrainError, match="sh0"):
        scene_from_gsplat_ckpt(tmp_path / "bad.pt")


FAKE_PARSER = """
import numpy as np

class Parser:
    def __init__(self, data_dir, factor=1, normalize=False, test_every=8):
        assert normalize, "psplat must ask for the trainer's normalised frame"
        self.image_names = [f"im{i:02d}.png" for i in range(9)]
        self.camera_ids = [1] * 9
        self.Ks_dict = {1: np.array([[100.0 / factor, 0, 60], [0, 101.0 / factor, 40], [0, 0, 1]])}
        self.imsize_dict = {1: (120, 80)}
        self.camtoworlds = np.stack([np.eye(4) for _ in range(9)])
        for i in range(9):
            self.camtoworlds[i, :3, 3] = [i, 0.0, -3.0]
"""

FAKE_TRAINER = """
import json, sys
from pathlib import Path
import torch
args = sys.argv[1:]
if "--help" in args:
    print(HELP)
    sys.exit(0)
result = Path(args[args.index("--result_dir") + 1])
(result / "ckpts").mkdir(parents=True)
(result / "stats").mkdir(parents=True)
torch.save({"step": 6999, "splats": SPLATS(3)}, result / "ckpts" / "ckpt_6999_rank0.pt")
torch.save({"step": 29999, "splats": SPLATS(5)}, result / "ckpts" / "ckpt_29999_rank0.pt")
(result / "stats" / "val_step6999.json").write_text(json.dumps({"psnr": 20.0, "ssim": 0.5}))
final = {"psnr": 27.4, "ssim": 0.86, "lpips": 0.1}
(result / "stats" / "val_step29999.json").write_text(json.dumps(final))
"""


def _fake_examples(tmp_path: Path) -> Path:
    ex = tmp_path / "examples"
    (ex / "datasets").mkdir(parents=True)
    (ex / "datasets" / "__init__.py").write_text("")
    (ex / "datasets" / "colmap.py").write_text(FAKE_PARSER)
    splats_src = (
        "def SPLATS(n):\n"
        "    return {'means': torch.zeros(n, 3), 'scales': torch.zeros(n, 3),\n"
        "            'quats': torch.ones(n, 4), 'opacities': torch.zeros(n),\n"
        "            'sh0': torch.zeros(n, 1, 3), 'shN': torch.zeros(n, 15, 3)}\n"
    )
    body = FAKE_TRAINER.replace(
        "args = sys.argv", f"HELP = {TYRO_HELP!r}\n{splats_src}args = sys.argv"
    )
    (ex / "simple_trainer.py").write_text(textwrap.dedent(body))
    return ex


def test_gsplat_cameras_use_the_trainers_frame_and_split(tmp_path: Path) -> None:
    ex = _fake_examples(tmp_path)
    cams = gsplat_cameras(ex, tmp_path / "data", factor=4)
    assert len(cams) == 9 and [c.split for c in cams].count("eval") == 2
    np.testing.assert_allclose(cams[3].position, [3.0, 0.0, -3.0])
    assert cams[0].fx == pytest.approx(25.0) and (cams[0].width, cams[0].height) == (120, 80)
    sys.modules.pop("datasets.colmap", None)
    sys.modules.pop("datasets", None)


def test_train_gsplat_end_to_end_with_a_stand_in_trainer(tmp_path: Path) -> None:
    ex = _fake_examples(tmp_path)
    reg = load_registry()
    run_dir = train_gsplat(reg.trainer("gsplat-mcmc-1m"), reg.scene("garden"), tmp_path / "garden",
                           tmp_path / "trained", ex, log=print)  # fmt: skip
    sys.modules.pop("datasets.colmap", None)
    sys.modules.pop("datasets", None)
    assert run_dir is not None and run_dir.parent.name == "gsplat-mcmc-1m"
    assert read_ply(run_dir / "scene.ply").count == 5  # the latest checkpoint, not the 7k one
    assert len(read_cameras_json(run_dir / "cams.json")) == 9
    assert json.loads((run_dir / "metrics.json").read_text())["psnr"] == 27.4
    run = json.loads((run_dir / "run.json").read_text())
    assert run["splats"] == 5 and run["trainer"]["args"] == ["--strategy.cap-max", "1000000"]
    assert "--disable_viewer" in run["trainer_command"]


def test_train_gsplat_preflight_stops_on_a_renamed_flag(tmp_path: Path) -> None:
    ex = _fake_examples(tmp_path)
    trainer_py = ex / "simple_trainer.py"
    trainer_py.write_text(trainer_py.read_text().replace("--strategy.cap-max", "--strategy.max-gs"))
    reg = load_registry()
    with pytest.raises(TrainError, match=r"--strategy\.cap-max"):
        train_gsplat(reg.trainer("gsplat-mcmc-1m"), reg.scene("garden"), tmp_path,
                     tmp_path / "trained", ex, log=print)  # fmt: skip
    assert not (tmp_path / "trained").exists()


# msplat wrapper --------------------------------------------------------------------------


def test_train_msplat_with_the_documented_api(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: dict[str, Any] = {}

    class Config:
        def __init__(self, iterations: int) -> None:
            calls["iterations"] = iterations

    class Trainer:
        def __init__(self, dataset: object, config: Config) -> None:
            pass

        def train(self, callback: Any, callback_every: int) -> None:
            callback(types.SimpleNamespace(iteration=1000, splat_count=12))

        def export_ply(self, path: str) -> None:
            n = 4
            z = np.zeros
            write_ply(Scene(z((n, 3), np.float32), z((n, 3), np.float32),
                            np.ones((n, 4), np.float32), z(n, np.float32),
                            z((n, 3), np.float32), z((n, 15, 3), np.float32)), path)  # fmt: skip

        def evaluate(self) -> dict[str, float]:
            return {"psnr": 26.1, "ssim": 0.81}

    fake = types.ModuleType("msplat")
    fake.__version__ = "1.1.4"  # type: ignore[attr-defined]
    fake.load_dataset = lambda path, eval_mode: calls.setdefault("eval_mode", eval_mode)  # type: ignore[attr-defined]
    fake.TrainingConfig = Config  # type: ignore[attr-defined]
    fake.GaussianTrainer = Trainer  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "msplat", fake)
    reg = load_registry()
    run_dir = train_msplat(reg.trainer("msplat-7k"), reg.scene("garden"), COLMAP,
                           tmp_path / "trained", log=print)  # fmt: skip
    assert run_dir is not None and run_dir.name == "msplat-1.1.4"
    assert calls == {"iterations": 7000, "eval_mode": True}
    assert json.loads((run_dir / "run.json").read_text())["splats"] == 4
    assert len(read_cameras_json(run_dir / "cams.json")) == 10


# E0 comparison -----------------------------------------------------------------------------


def test_compare_to_reference() -> None:
    ref = {"gsplat-mcmc-1m": {"scenes": {"garden": {"psnr": 27.5, "ssim": 0.87}}}}
    rows = compare_to_reference([
        {"scene": "garden", "trainer": "gsplat-mcmc-1m", "psnr": 27.3, "ssim": 0.86},
        {"scene": "room", "trainer": "gsplat-mcmc-1m", "psnr": 31.0, "ssim": 0.92},
    ], ref)  # fmt: skip
    assert [r["status"] for r in rows] == ["match", "no reference"]
    far = compare_to_reference([{"scene": "garden", "trainer": "gsplat-mcmc-1m",
                                 "psnr": 26.9, "ssim": 0.86}], ref)  # fmt: skip
    assert far[0]["status"] == "MISMATCH"


def test_bench_cli(tmp_path: Path) -> None:
    runner = CliRunner()
    listing = runner.invoke(app, ["bench", "scenes"])
    assert listing.exit_code == 0 and "mipnerf360-core: garden" in listing.output
    run_dir = tmp_path / "garden" / "gsplat-mcmc-1m" / "abc"
    run_dir.mkdir(parents=True)
    (run_dir / "run.json").write_text(json.dumps({
        "scene": "garden", "trainer": {"name": "gsplat-mcmc-1m"},
        "metrics": {"psnr": 27.3, "ssim": 0.86},
    }))  # fmt: skip
    report = runner.invoke(app, ["bench", "e0", str(tmp_path)])
    assert report.exit_code == 0 and "no reference" in report.output
    stub = runner.invoke(app, ["bench", "run", "--scenes", "x"])
    assert stub.exit_code == 2 and "issues/28" in stub.output
