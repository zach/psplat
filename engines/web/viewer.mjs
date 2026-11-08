// psplat web viewer (P1.7, https://github.com/zach/psplat/issues/26).
//
// Shows an ordered splat scene at a chosen budget: the first k splats of the
// file. It reads the manifest written by `psplat order` for the cut points and
// the bytes each budget takes, and offers:
//   - a budget slider that snaps to the cut points, or takes any count (PLY);
//   - a split-screen wipe: budget on the left, the whole scene on the right;
//   - splat count, bytes and frame time;
//   - a projected-size threshold: hide splats whose on-screen size, estimated
//     from camera distance, is below a number of pixels (experiment E4).
//
// How a budget is drawn: the scene is loaded once (PLY with PlayCanvas's
// Morton reordering switched off, so rows stay in psplat's order; SOG
// decompressed to the same layout) and, for a budget k, a resource is built
// from the first k rows. Only those splats are sorted and drawn, so the frame
// time reflects the budget. A SOG's texels are in psplat's banded order
// (docs/formats.md), so its budgets are exact only at cut points and the
// slider snaps to them.
//
// Query parameters (all optional except scene):
//   scene=URL  manifest=URL  cameras=URL  budget=0.25|12345  split=1
//   view=NAME  minsigma=PX  size=WxH (fixed canvas size, for repeatable images)
//
// window.psplat exposes the same controls for tests and the E6 study rig.

import * as pc from './vendor/playcanvas-2.23.3.min.mjs';

const params = new URLSearchParams(location.search);
const $ = (id) => document.getElementById(id);
const statusEl = $('status');

function setStatus(text, isError = false) {
    statusEl.textContent = text;
    statusEl.classList.toggle('error', isError);
}

async function fetchJson(url) {
    const response = await fetch(url);
    if (!response.ok) throw new Error(`${url}: HTTP ${response.status}`);
    return response.json();
}

// Budgets ----------------------------------------------------------------------------------

const DEFAULT_CUTS = [0.05, 0.1, 0.25, 0.5, 0.75, 1.0];

// The manifest's rule: round(count * frac), at least 1 for a non-empty scene.
function cutIndex(count, frac) {
    return count ? Math.min(count, Math.max(1, Math.round(count * frac))) : 0;
}

function cutsFor(count, manifest) {
    if (manifest && Array.isArray(manifest.cuts) && manifest.cuts.length) {
        return manifest.cuts.map((c) => ({ ...c }));
    }
    return DEFAULT_CUTS.map((frac) => ({ frac, index: cutIndex(count, frac) }));
}

// Exact PLY size for any prefix, from two cut sizes: header + k * stride, where
// only the digits of the vertex count in the header change.
function plyBytesFor(cuts) {
    const known = cuts.filter((c) => Number.isInteger(c.bytes_ply) && c.index > 0);
    if (known.length < 2) return null;
    const a = known[0];
    const b = known[known.length - 1];
    if (b.index === a.index) return null;
    const stride = (b.bytes_ply - a.bytes_ply - (String(b.index).length - String(a.index).length)) /
        (b.index - a.index);
    if (!Number.isInteger(stride)) return null;
    const header = b.bytes_ply - b.index * stride - String(b.index).length;
    return (k) => header + String(k).length + k * stride;
}

// Scene data ---------------------------------------------------------------------------------

// The first n rows of a GSplatData; property arrays are views, not copies.
function prefixData(data, n) {
    const vertex = data.getElement('vertex');
    const properties = vertex.properties.map((p) => ({ ...p, storage: p.storage.subarray(0, n) }));
    const out = new pc.GSplatData([{ name: 'vertex', count: n, properties }], data.comments);
    out.activated = data.activated;
    return out;
}

function loadAsset(app, url) {
    return new Promise((resolve, reject) => {
        // reorder: false keeps the file's splat order; PlayCanvas would
        // otherwise Morton-sort a PLY on load and break every budget.
        // decompress: true makes the SOG loader decode to plain splat data
        // while it still holds the textures, so budgets can be cut from it.
        const asset = new pc.Asset('scene', 'gsplat', { url }, { reorder: false, decompress: true });
        asset.on('load', () => resolve(asset));
        asset.on('error', (err) => reject(new Error(`${url}: ${err}`)));
        app.assets.add(asset);
        app.assets.load(asset);
    });
}

async function sceneData(resource) {
    let data = resource.gsplatData;
    if (data.isSog) data = await data.decompress();
    else if (data.isCompressed) data = data.decompress();
    return data;
}

// Cameras --------------------------------------------------------------------------------------

// cams.json uses the OpenCV convention (x right, y down, z forward, w2c).
// A PlayCanvas camera looks down -z with +y up, so flip y and z.
function poseFromOpenCV(w2c) {
    const m = new pc.Mat4();
    // Mat4.data is column-major; w2c is row-major nested arrays.
    const d = m.data;
    for (let r = 0; r < 4; r++) for (let c = 0; c < 4; c++) d[c * 4 + r] = w2c[r][c];
    const c2w = m.clone().invert();
    const flip = new pc.Mat4().setScale(1, -1, -1);
    return c2w.mul(flip);
}

class Orbit {
    constructor(camera, canvas) {
        this.camera = camera;
        this.target = new pc.Vec3();
        this.distance = 5;
        this.yaw = 0;
        this.pitch = -15;
        this.fixed = null; // a dataset camera pose, until the user drags
        this.listeners = [];
        let dragging = null;
        canvas.addEventListener('pointerdown', (e) => {
            dragging = { x: e.clientX, y: e.clientY, button: e.button };
            canvas.setPointerCapture(e.pointerId);
        });
        canvas.addEventListener('pointerup', () => { dragging = null; });
        canvas.addEventListener('pointermove', (e) => {
            if (!dragging) return;
            const dx = e.clientX - dragging.x;
            const dy = e.clientY - dragging.y;
            dragging.x = e.clientX;
            dragging.y = e.clientY;
            this.leaveFixed();
            if (dragging.button === 0) {
                this.yaw -= dx * 0.3;
                this.pitch = Math.max(-89, Math.min(89, this.pitch - dy * 0.3));
            } else {
                const right = this.camera.right.clone().mulScalar(-dx * this.distance * 0.0015);
                const up = this.camera.up.clone().mulScalar(dy * this.distance * 0.0015);
                this.target.add(right).add(up);
            }
            this.apply();
        });
        canvas.addEventListener('wheel', (e) => {
            e.preventDefault();
            this.leaveFixed();
            this.distance *= Math.exp(e.deltaY * 0.001);
            this.apply();
        }, { passive: false });
        canvas.addEventListener('contextmenu', (e) => e.preventDefault());
    }

    frame(aabb) {
        this.target.copy(aabb.center);
        this.distance = Math.max(aabb.halfExtents.length() * 2.2, 1e-3);
        this.apply();
    }

    leaveFixed() {
        if (!this.fixed) return;
        // Orbit about the point in front of the dataset camera.
        const forward = this.camera.forward.clone();
        const position = this.camera.getPosition().clone();
        this.target.copy(position).add(forward.mulScalar(this.distance));
        const angles = this.camera.getEulerAngles();
        this.yaw = angles.y;
        this.pitch = angles.x;
        this.fixed = null;
        this.listeners.forEach((f) => f());
    }

    setFixed(pose, fovDeg) {
        this.fixed = pose;
        this.camera.setPosition(pose.getTranslation());
        this.camera.setRotation(new pc.Quat().setFromMat4(pose));
        this.camera.camera.fov = fovDeg;
    }

    apply() {
        if (this.fixed) return;
        const q = new pc.Quat().setFromEulerAngles(this.pitch, this.yaw, 0);
        const offset = q.transformVector(new pc.Vec3(0, 0, this.distance));
        this.camera.setPosition(this.target.clone().add(offset));
        this.camera.lookAt(this.target);
    }
}

// Projected-size threshold ---------------------------------------------------------------------

// Hides a splat when focal_px * largest_scale / distance < uPsplatMinSigma.
// It is a function of camera distance only, as experiment E4 specifies, and is
// applied while splats are written to PlayCanvas's work buffer.
const MIN_SIGMA_MODIFIER = {
    glsl: `
        uniform vec3 uPsplatCamera;
        uniform float uPsplatFocal;
        uniform float uPsplatMinSigma;
        bool psplatHidden = false;
        void modifySplatCenter(inout vec3 center) {}
        void modifySplatRotationScale(vec3 originalCenter, vec3 modifiedCenter, inout vec4 rotation, inout vec3 scale) {
            float d = max(distance(modifiedCenter, uPsplatCamera), 1e-6);
            float sigmaPx = uPsplatFocal * max(scale.x, max(scale.y, scale.z)) / d;
            psplatHidden = sigmaPx < uPsplatMinSigma;
            if (psplatHidden) scale = vec3(0.0);
        }
        void modifySplatColor(vec3 center, inout vec4 color) {
            if (psplatHidden) color.a = 0.0;
        }
    `,
};

// The viewer -----------------------------------------------------------------------------------

async function main() {
    const sceneUrl = params.get('scene');
    if (!sceneUrl) throw new Error('no scene: open this page through `psplat viewer SCENE`');
    const format = sceneUrl.toLowerCase().split('?')[0].endsWith('.sog') ? 'sog' : 'ply';
    $('title').textContent = decodeURIComponent(sceneUrl.split('/').pop());

    const [manifest, cameras] = await Promise.all([
        params.get('manifest') ? fetchJson(params.get('manifest')) : null,
        params.get('cameras') ? fetchJson(params.get('cameras')) : null,
    ]);

    const canvas = $('canvas');
    const size = params.get('size');
    if (size) {
        const [w, h] = size.split('x').map(Number);
        Object.assign(canvas.style, { width: `${w}px`, height: `${h}px`, right: 'auto', bottom: 'auto' });
    }
    // WebGL 2 everywhere, so the same browser gives the same pixels.
    const device = await pc.createGraphicsDevice(canvas, {
        deviceTypes: [pc.DEVICETYPE_WEBGL2], antialias: false, preserveDrawingBuffer: true,
    });
    const options = new pc.AppOptions();
    options.graphicsDevice = device;
    options.componentSystems = [pc.CameraComponentSystem, pc.GSplatComponentSystem];
    // A SOG's images load as texture assets, so the texture handler is needed too.
    options.resourceHandlers = [pc.GSplatHandler, pc.TextureHandler];
    const app = new pc.AppBase(canvas);
    app.init(options);
    app.setCanvasFillMode(pc.FILLMODE_NONE);
    app.setCanvasResolution(pc.RESOLUTION_AUTO);
    const resize = () => app.resizeCanvas(canvas.clientWidth, canvas.clientHeight);
    window.addEventListener('resize', resize);
    app.start();
    resize();

    setStatus(`loading ${format.toUpperCase()}…`);
    const asset = await loadAsset(app, sceneUrl);
    const data = await sceneData(asset.resource);
    const count = data.numSplats;
    if (manifest && manifest.count !== count) {
        setStatus(`manifest is for ${manifest.count} splats, scene has ${count}; ignoring it`, true);
    }
    const cuts = cutsFor(count, manifest && manifest.count === count ? manifest : null);
    const bytesKey = `bytes_${format}`;
    const plyBytes = format === 'ply' ? plyBytesFor(cuts) : null;

    // Two layers and two cameras with the same pose: the budget on the left of
    // the wipe, the whole scene on the right.
    const layers = app.scene.layers;
    const layerA = new pc.Layer({ name: 'psplat-budget' });
    const layerB = new pc.Layer({ name: 'psplat-full' });
    layers.push(layerA);
    layers.push(layerB);

    const makeCamera = (name, layer) => {
        const e = new pc.Entity(name);
        e.addComponent('camera', {
            clearColor: new pc.Color(0, 0, 0), layers: [layer.id], nearClip: 0.01, farClip: 1000,
        });
        app.root.addChild(e);
        return e;
    };
    const camA = makeCamera('camera-budget', layerA);
    const camB = makeCamera('camera-full', layerB);
    camB.enabled = false;

    const splatA = new pc.Entity('splats-budget');
    const splatB = new pc.Entity('splats-full');
    app.root.addChild(splatA);
    app.root.addChild(splatB);

    // Resources by splat count; the full scene's is kept, others are evicted.
    const resources = new Map();
    const resourceFor = (k) => {
        if (!resources.has(k)) {
            resources.set(k, new pc.GSplatResource(device, k === count ? data : prefixData(data, k)));
            for (const key of [...resources.keys()]) {
                if (resources.size <= 3) break;
                if (key !== k && key !== count && key !== state.budget) {
                    resources.get(key).destroy();
                    resources.delete(key);
                }
            }
        }
        return resources.get(k);
    };

    const state = {
        format, count, cuts, budget: count, split: false, minSigma: 0, view: '', frameMs: 0,
    };

    const attach = (entity, layer, k) => {
        if (entity.gsplat) entity.removeComponent('gsplat');
        if (k === 0) return;
        entity.addComponent('gsplat', { resource: resourceFor(k), layers: [layer.id] });
        applyMinSigma(entity);
    };

    const applyMinSigma = (entity) => {
        if (!entity.gsplat) return;
        if (state.minSigma > 0) {
            entity.gsplat.setWorkBufferModifier(MIN_SIGMA_MODIFIER);
            entity.gsplat.workBufferUpdate = pc.WORKBUFFER_UPDATE_ALWAYS;
        } else {
            entity.gsplat.setWorkBufferModifier(null);
            entity.gsplat.workBufferUpdate = pc.WORKBUFFER_UPDATE_AUTO;
        }
    };

    const orbit = new Orbit(camA, canvas);
    orbit.listeners.push(() => { state.view = ''; $('view').value = ''; });
    const aabb = new pc.BoundingBox();
    data.calcAabb(aabb);
    orbit.frame(aabb);

    // Controls -----------------------------------------------------------------------------

    const slider = $('budget');
    const snap = $('snap');
    if (format === 'sog') {
        snap.value = 'cuts';
        snap.disabled = true;
        snap.title = 'SOG budgets are exact only at cut points (docs/formats.md)';
    }

    const configureSlider = () => {
        if (snap.value === 'cuts') {
            slider.min = 0;
            slider.max = cuts.length - 1;
            slider.step = 1;
            syncSlider();
            $('ticks').innerHTML = cuts.map((c) => `<span>${Math.round(c.frac * 100)}%</span>`).join('');
        } else {
            slider.min = 1;
            slider.max = count;
            slider.step = 1;
            syncSlider();
            $('ticks').innerHTML = '<span>1</span><span>' + count.toLocaleString() + '</span>';
        }
    };

    // Keep the slider where the budget is, however the budget was set.
    const syncSlider = () => {
        if (snap.value === 'cuts') {
            const i = cuts.findIndex((c) => c.index >= state.budget);
            slider.value = i < 0 ? cuts.length - 1 : i;
        } else {
            slider.value = state.budget;
        }
    };

    const updateStats = () => {
        const cut = cuts.find((c) => c.index === state.budget);
        let bytes = cut && Number.isInteger(cut[bytesKey]) ? cut[bytesKey] : null;
        if (bytes === null && plyBytes) bytes = plyBytes(state.budget);
        $('stat-splats').textContent = `${state.budget.toLocaleString()} of ${count.toLocaleString()}`;
        $('stat-share').textContent = count ? `${(100 * state.budget / count).toFixed(1)}%` : '–';
        $('stat-bytes').textContent = bytes === null ? '–' : `${(bytes / 1048576).toFixed(2)} MB`;
    };

    const setBudget = (k) => {
        k = Math.max(0, Math.min(count, Math.round(k)));
        if (format === 'sog' && !cuts.some((c) => c.index === k)) {
            // Snap to the nearest cut at or above k.
            k = (cuts.find((c) => c.index >= k) || cuts[cuts.length - 1]).index;
        }
        state.budget = k;
        attach(splatA, layerA, k);
        updateStats();
        syncSlider();
        return k;
    };

    // Split screen as a wipe: each camera draws one half of the canvas, with the
    // projection of the whole canvas shifted so its half lines up with the
    // other's. (PlayCanvas applies scissorRect only to its own shadow cameras.)
    const halfProjection = (side) => (mat) => {
        const cam = camA.camera;
        mat.setPerspective(cam.fov, canvas.width / canvas.height, cam.nearClip, cam.farClip);
        // Clip-space x' = 2x + w (left half) or 2x - w (right half).
        const shift = new pc.Mat4();
        shift.data[0] = 2;
        shift.data[12] = side === 'left' ? 1 : -1;
        mat.copy(shift.mul(mat));
    };

    const setSplit = (on) => {
        state.split = !!on;
        $('split').checked = state.split;
        document.body.classList.toggle('split', state.split);
        camB.enabled = state.split;
        if (state.split && !splatB.gsplat) attach(splatB, layerB, count);
        if (!state.split && splatB.gsplat) splatB.removeComponent('gsplat');
        camA.camera.rect = new pc.Vec4(0, 0, state.split ? 0.5 : 1, 1);
        camA.camera.calculateProjection = state.split ? halfProjection('left') : null;
        camB.camera.rect = new pc.Vec4(0.5, 0, 0.5, 1);
        camB.camera.calculateProjection = halfProjection('right');
    };

    const setMinSigma = (px) => {
        state.minSigma = Math.max(0, Number(px) || 0);
        $('minsigma').value = state.minSigma;
        $('minsigma-value').textContent = state.minSigma > 0 ? `${state.minSigma} px` : 'off';
        applyMinSigma(splatA);
        applyMinSigma(splatB);
    };

    const setView = (name) => {
        const cam = (cameras && cameras.views || []).find((v) => v.name === name);
        state.view = cam ? name : '';
        $('view').value = state.view;
        if (!cam) return false;
        const fov = 2 * Math.atan(cam.height / (2 * cam.fy)) * 180 / Math.PI;
        orbit.setFixed(poseFromOpenCV(cam.w2c), fov);
        return true;
    };

    if (cameras && Array.isArray(cameras.views)) {
        const select = $('view');
        for (const v of cameras.views) {
            const option = document.createElement('option');
            option.value = v.name;
            option.textContent = v.split ? `${v.name} (${v.split})` : v.name;
            select.appendChild(option);
        }
        select.addEventListener('change', () => (select.value ? setView(select.value) : orbit.leaveFixed()));
    }

    slider.addEventListener('input', () => {
        const k = snap.value === 'cuts' ? cuts[Number(slider.value)].index : Number(slider.value);
        setBudget(k);
    });
    snap.addEventListener('change', configureSlider);
    $('split').addEventListener('change', (e) => setSplit(e.target.checked));
    $('minsigma').addEventListener('input', (e) => setMinSigma(e.target.value));

    // Per frame: keep camera B on camera A, feed the threshold uniforms, time frames.
    let frames = 0;
    let elapsed = 0;
    app.on('update', (dt) => {
        camB.setPosition(camA.getPosition());
        camB.setRotation(camA.getRotation());
        camB.camera.fov = camA.camera.fov;
        if (state.minSigma > 0) {
            const focal = (canvas.height / 2) / Math.tan(camA.camera.fov * Math.PI / 360);
            const p = camA.getPosition();
            for (const e of [splatA, splatB]) {
                if (!e.gsplat) continue;
                e.gsplat.setParameter('uPsplatCamera', [p.x, p.y, p.z]);
                e.gsplat.setParameter('uPsplatFocal', focal);
                e.gsplat.setParameter('uPsplatMinSigma', state.minSigma);
            }
        }
        frames += 1;
        elapsed += dt;
        if (elapsed >= 0.5) {
            state.frameMs = 1000 * elapsed / frames;
            $('stat-fps').textContent = `${state.frameMs.toFixed(1)} ms (${(1000 / state.frameMs).toFixed(0)} fps)`;
            frames = 0;
            elapsed = 0;
        }
    });

    // Initial state from the URL.
    const budgetParam = params.get('budget');
    let initial = count;
    if (budgetParam) {
        const v = Number(budgetParam);
        initial = v > 0 && v <= 1 && budgetParam.includes('.') ? cutIndex(count, v) : v;
    }
    setBudget(initial);
    configureSlider();
    setSplit(params.get('split') === '1');
    setMinSigma(params.get('minsigma') || 0);
    if (params.get('view')) setView(params.get('view'));
    else if (cameras && Array.isArray(cameras.views)) {
        const first = cameras.views.find((v) => v.split === 'eval') || cameras.views[0];
        if (first) setView(first.name);
    }
    setStatus(`${format.toUpperCase()}, ${count.toLocaleString()} splats` +
        (manifest ? `, ordered by ${manifest.ordering?.method ?? '?'}` : ', no manifest'));

    // Wait for frames to render after a change, for tests and screenshots.
    const settle = (n = 3) => new Promise((resolve) => {
        let left = n;
        const tick = () => { if (--left <= 0) { app.off('frameend', tick); resolve(); } };
        app.on('frameend', tick);
    });

    return {
        state: () => ({ ...state, cuts: state.cuts.map((c) => ({ ...c })), rendered: state.budget }),
        setBudget, setSplit, setMinSigma, setView, settle,
        app,
    };
}

window.psplat = {
    ready: main().then((api) => {
        Object.assign(window.psplat, api);
        return api;
    }).catch((err) => {
        console.error(err);
        setStatus(String(err.message || err), true);
        throw err;
    }),
};
