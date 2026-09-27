"""Generate the synthetic fixture: 200 splats, 8 cameras, and known answers.

Run from the repository root to regenerate the checked-in files::

    python tests/fixtures/synthetic/generate.py

Outputs, next to this script:

- ``scene.ply`` — 200 splats, SH degree 3, standard 62-property layout;
- ``cams.json`` — 8 views, 64×48 px, on a circle around the scene; every
  fourth view (view_00, view_04) is marked ``eval``;
- ``oracles.json`` — splats whose behaviour is known in advance, so
  contribution tests have ground truth. See docs/testing.md.

The layout, in splat index order:

- 0–187, the cloud: random splats within radius 0.8 of the origin;
- 188–192, off-screen: small splats far above every camera's field of view,
  so they are invisible and contribute nothing in every view;
- 193, the occluded target: a small splat at (0, 0, 1.3), above the cloud;
- 194–197, its occluders: four wide, near-opaque splats on the line from
  view_01 to the target. In view_01 every pixel the target could reach stops
  before it, so its contribution there is exactly zero; from view_05, on the
  opposite side, it is in plain sight;
- 198–199, the overlap pair: two identical splats one behind the other on
  view_02's line of sight to (0, 0, -1.3), below the cloud.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))  # the repository root, for `import psplat`

from psplat.io.cameras import Camera, look_at, write_cameras  # noqa: E402
from psplat.io.ply import write_ply  # noqa: E402
from psplat.io.scene import Scene  # noqa: E402

SEED = 20260927
N_CLOUD = 188
WIDTH, HEIGHT, FOCAL = 64, 48, 50.0
RADIUS, CAM_HEIGHT = 4.0, 0.3
OCCLUDING_VIEW, OPPOSITE_VIEW, OVERLAP_VIEW = 1, 5, 2


def logit(p: float) -> float:
    return math.log(p / (1 - p))


def cameras() -> list[Camera]:
    out = []
    for k in range(8):
        angle = k * math.pi / 4
        eye = (RADIUS * math.cos(angle), RADIUS * math.sin(angle), CAM_HEIGHT)
        out.append(
            Camera(
                name=f"view_{k:02d}",
                width=WIDTH,
                height=HEIGHT,
                fx=FOCAL,
                fy=FOCAL,
                cx=WIDTH / 2,
                cy=HEIGHT / 2,
                w2c=look_at(eye, (0.0, 0.0, 0.0)),
                split="eval" if k % 4 == 0 else "train",
            )
        )
    return out


def along(eye: np.ndarray, target: np.ndarray, t: float) -> list[float]:
    return list(eye + t * (target - eye))


def build() -> tuple[Scene, list[Camera], dict[str, object]]:
    rng = np.random.default_rng(SEED)
    cams = cameras()

    # Cloud: positions uniform in a ball of radius 0.8.
    direction = rng.normal(size=(N_CLOUD, 3))
    direction /= np.linalg.norm(direction, axis=1, keepdims=True)
    radius = 0.8 * rng.uniform(0, 1, (N_CLOUD, 1)) ** (1 / 3)
    means = [*(direction * radius).tolist()]
    sigmas = np.exp(rng.normal(-3.0, 0.4, (N_CLOUD, 3))).tolist()
    opacities = (1 / (1 + np.exp(-rng.normal(1.5, 1.5, N_CLOUD)))).tolist()

    # Off-screen: 5 small splats high overhead.
    offscreen = list(range(len(means), len(means) + 5))
    for i in range(5):
        means.append([0.5 * math.cos(i), 0.5 * math.sin(i), 7.0 + i])
        sigmas.append([0.02] * 3)
        opacities.append(0.9)

    # Occluded target and its four occluders along view_01's line of sight.
    eye = cams[OCCLUDING_VIEW].position
    target_pos = np.array([0.0, 0.0, 1.3])
    target = len(means)
    means.append(target_pos.tolist())
    sigmas.append([0.02] * 3)
    opacities.append(0.9)
    occluders = list(range(len(means), len(means) + 4))
    for t in (0.50, 0.53, 0.56, 0.59):
        means.append(along(eye, target_pos, t))
        sigmas.append([0.4] * 3)
        opacities.append(0.99999)

    # Overlap pair on view_02's line of sight, below the cloud.
    eye2 = cams[OVERLAP_VIEW].position
    below = np.array([0.0, 0.0, -1.3])
    front, back = len(means), len(means) + 1
    for t in (0.60, 0.70):
        means.append(along(eye2, below, t))
        sigmas.append([0.1] * 3)
        opacities.append(0.7)

    n = len(means)
    assert n == 200, n
    f = np.float32
    quats = rng.normal(size=(n, 4))
    quats[N_CLOUD:] = [1.0, 0.0, 0.0, 0.0]  # the constructed splats are isotropic anyway
    scene = Scene(
        means=np.asarray(means, f),
        scales=np.log(np.asarray(sigmas)).astype(f),
        quats=quats.astype(f),
        opacities=np.array([logit(min(o, 0.99999)) for o in opacities], f),
        sh0=rng.normal(0.0, 1.0, (n, 3)).astype(f),
        shN=rng.normal(0.0, 0.2, (n, 15, 3)).astype(f),
    )
    oracles: dict[str, object] = {
        "offscreen": {"splats": offscreen, "views": "all"},
        "occluded": {
            "splat": target,
            "occluders": occluders,
            "hidden_in": cams[OCCLUDING_VIEW].name,
            "visible_in": cams[OPPOSITE_VIEW].name,
        },
        "overlap": {"front": front, "back": back, "view": cams[OVERLAP_VIEW].name},
    }
    return scene, cams, oracles


def main(out_dir: Path = HERE) -> None:
    scene, cams, oracles = build()
    write_ply(scene, out_dir / "scene.ply", comments=["psplat synthetic fixture", f"seed {SEED}"])
    write_cameras(cams, out_dir / "cams.json")
    (out_dir / "oracles.json").write_text(json.dumps(oracles, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else HERE)
