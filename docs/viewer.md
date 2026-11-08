<!-- psplat-docs: audience=dev -->
# The web viewer

`psplat viewer` (P1.7, [#26](https://github.com/zach/psplat/issues/26))
serves `engines/web/` and one scene on 127.0.0.1 and opens the default
browser. The viewer is static files: an HTML page, `viewer.mjs`, and the
PlayCanvas engine 2.23.3 vendored unmodified (MIT). Nothing is built or
downloaded, and the wheel carries the files as `psplat/_web/`.

## How a budget is drawn

A budget is the first k splats of the file. The viewer:

1. Loads the scene once with PlayCanvas's loader.
   - **PLY:** loaded with `reorder: false`. PlayCanvas otherwise Morton-sorts
     a PLY on load, which would scramble every budget.
   - **SOG:** loaded with `decompress: true`, which decodes it to the same
     plain layout as a PLY.
2. For a budget k, builds a splat resource from the first k rows. The rows
   are views into the loaded arrays, not copies.
3. Keeps the whole scene's resource and up to two others, so going back and
   forth between budgets does not rebuild them every time.

Only the budget's splats are sorted and drawn, so the frame-time readout
reflects the budget. View-dependent colour (spherical harmonics) is kept.

PlayCanvas's own ways of drawing fewer splats did not fit:

- `GSplatContainer` can draw a prefix cheaply, but it renders without
  spherical harmonics.
- `splatBudget` applies only to PlayCanvas's streamed, spatial
  level-of-detail format.

| Format | Budgets | Why |
|---|---|---|
| PLY | any splat count | rows are psplat's order |
| SOG | cut points only; the slider snaps | texels are in the banded order (D7, `docs/formats.md`) |
| SPZ, glTF | not shown | glTF splats are reordered by PlayCanvas's loader; use the PLY |

## Controls

- **Budget:** steps through the manifest's cut points, or, for a PLY, any
  count.
- **Compare with full scene:** a wipe. Two cameras share one pose. Each draws
  half of the canvas, with the whole canvas's projection shifted so that the
  halves line up exactly. PlayCanvas's `scissorRect` is applied only to its
  shadow cameras, so it cannot do this.
- **Hide splats below N px:** hides a splat when `focal · largest scale /
  distance` is below N pixels. This is the projected-size threshold of
  experiment E4: a function of camera distance only, approximated by the
  largest axis and the distance rather than the projected depth. It runs in
  PlayCanvas's work-buffer modifier, refreshed every frame while it is on.
- **View:** places the camera at a pose from `cams.json`, converted from
  OpenCV to PlayCanvas (y and z flipped), with the vertical field of view
  from `fy`. The principal point is assumed central. Dragging switches back
  to orbiting.
- **Readout:** splats drawn; bytes for that budget from the manifest
  (`bytes_sog` or `bytes_ply`; for a PLY at any count, computed exactly from
  two cut sizes); and frame time averaged over half a second.

## Query parameters and the scripting API

`psplat viewer` passes `scene`, `manifest` and `cameras`. These parameters
are also accepted:

| Parameter | Meaning |
|---|---|
| `budget` | a fraction such as `0.25`, or a splat count |
| `split=1` | start in split screen |
| `view` | a camera name from `cams.json` |
| `minsigma` | projected-size threshold, pixels |
| `size=WxH` | fix the canvas size, for repeatable screenshots |

`window.psplat` exposes the same controls:

- `ready`: a promise that resolves once the scene is loaded;
- `setBudget(k)`, `setSplit(on)`, `setMinSigma(px)` and `setView(name)`;
- `state()`, the current settings;
- `settle(n)`, which waits for n frames to render.

The E6 study rig will drive the viewer through this API. The viewer always
uses WebGL 2, so one browser gives repeatable images.

## Tests

- **`tests/test_viewer.py`** checks how the scene and its companion files
  are found, and what the server sends: content types, only the session's
  files under `/data/`, and no path traversal. These run everywhere.
- **The browser test**, opt-in with `PSPLAT_BROWSER=1` (the Linux CI job sets
  it), runs `tests/web/check_viewer.mjs` in headless Chromium with software
  WebGL, on the fixture packed as PLY and as SOG. It checks:
  - every cut can be set, and the slider follows;
  - brightness rises with the budget;
  - between cut points, a PLY takes the exact count and a SOG snaps;
  - the wipe's halves equal the budget image and the full image exactly;
  - the threshold hides more as it rises.
- **SOG against PLY:** each budget adds the same brightness over the
  smallest in both formats, so each SOG cut holds the PLY prefix's splats.
  A Morton-sorted SOG from splat-transform's command line serves as a
  negative control and fails that check, with gains off by up to 40.
- **Windows:** the TD job starts `psplat viewer` from the installed wheel
  and drives it in the runner's Microsoft Edge with `--check`.

Not automated: a person opening it on Windows, and the garden scene. Both
are part of #26's acceptance and are run by hand.
