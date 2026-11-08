// Drive the psplat web viewer in headless Chromium and report what it renders.
//
//   node tests/web/check_viewer.mjs URL [--out report.json] [--shots DIR]
//                                       [--channel msedge|chrome] [--check]
//
// URL is the viewer's address, as `psplat viewer` prints it. Prints a JSON
// report; tests/test_viewer.py asserts on it in detail. --check applies the
// core assertions here and exits 1 on failure, for CI jobs without pytest
// (the Windows TD job runs it in Microsoft Edge with --channel msedge).
// Software WebGL is enough: the fixture has a few hundred splats.

import { createRequire } from 'node:module';
import { writeFileSync, mkdirSync } from 'node:fs';
import { join } from 'node:path';

// require() honours NODE_PATH, where a global Playwright install lives; import does not.
const { chromium } = createRequire(import.meta.url)('playwright');

const url = process.argv[2];
const outIdx = process.argv.indexOf('--out');
const shotsIdx = process.argv.indexOf('--shots');
const shots = shotsIdx > 0 ? process.argv[shotsIdx + 1] : null;
const channelIdx = process.argv.indexOf('--channel');
const channel = channelIdx > 0 ? process.argv[channelIdx + 1] : undefined;
const check = process.argv.includes('--check');
if (shots) mkdirSync(shots, { recursive: true });

const browser = await chromium.launch({
    channel, args: ['--use-angle=swiftshader', '--enable-unsafe-swiftshader'],
});
const page = await browser.newPage({ viewport: { width: 640, height: 480 } });
const logs = [];
page.on('console', (m) => logs.push(`${m.type()}: ${m.text()}`));
page.on('pageerror', (e) => logs.push(`pageerror: ${e.message}`));
// Name the URL of any failed request; the console only says "Failed to load resource".
page.on('response', (r) => { if (r.status() >= 400) logs.push(`http ${r.status()}: ${r.url()}`); });

const report = { url, logs };
try {
    await page.goto(url);
    await page.waitForFunction(() => window.psplat && window.psplat.ready, null, { timeout: 30000 });
    await page.evaluate(() => window.psplat.ready);

    // Fraction of canvas pixels that are not background, and their mean brightness.
    // Canvas pixels only (not the page, where the control panel overlays the
    // left side): the fraction that are not background, in total and per half,
    // and the mean brightness.
    const coverage = async (name) => {
        await page.evaluate(() => window.psplat.settle(6));
        const stats = await page.evaluate(async () => {
            const url = document.getElementById('canvas').toDataURL('image/png');
            const img = new Image();
            img.src = url;
            await img.decode();
            const c = document.createElement('canvas');
            c.width = img.width; c.height = img.height;
            const g = c.getContext('2d');
            g.drawImage(img, 0, 0);
            const d = g.getImageData(0, 0, c.width, c.height).data;
            let lit = 0, sum = 0, litLeft = 0, litRight = 0, sumLeft = 0, sumRight = 0;
            for (let i = 0; i < d.length; i += 4) {
                const v = d[i] + d[i + 1] + d[i + 2];
                const left = ((i / 4) % c.width) < c.width / 2;
                sum += v;
                if (left) sumLeft += v; else sumRight += v;
                if (v > 6) {
                    lit++;
                    if (left) litLeft++; else litRight++;
                }
            }
            const n = d.length / 4;
            return {
                png: url, lit: lit / n, litLeft: litLeft / (n / 2), litRight: litRight / (n / 2),
                mean: sum / (3 * n), meanLeft: sumLeft / (1.5 * n), meanRight: sumRight / (1.5 * n),
            };
        });
        if (shots) writeFileSync(join(shots, `${name}.png`), Buffer.from(stats.png.split(',')[1], 'base64'));
        delete stats.png;
        return stats;
    };

    const state0 = await page.evaluate(() => window.psplat.state());
    report.initial = state0;
    report.budgets = [];
    for (const cut of state0.cuts) {
        const k = await page.evaluate((k) => window.psplat.setBudget(k), cut.index);
        const s = await page.evaluate(() => window.psplat.state());
        report.budgets.push({ requested: cut.index, budget: k, stats: await page.evaluate(() => ({
            splats: document.getElementById('stat-splats').textContent,
            bytes: document.getElementById('stat-bytes').textContent,
            slider: Number(document.getElementById('budget').value),
        })), image: await coverage(`budget-${cut.index}`), state: { budget: s.budget } });
    }
    // A count between cuts: exact for PLY, snapped for SOG.
    const mid = Math.max(1, Math.floor((state0.cuts[0].index + state0.cuts[1].index) / 2));
    report.between = { requested: mid, budget: await page.evaluate((k) => window.psplat.setBudget(k), mid) };

    // Split screen at the smallest budget: left budget, right full scene.
    await page.evaluate((k) => window.psplat.setBudget(k), state0.cuts[0].index);
    await page.evaluate(() => window.psplat.setSplit(true));
    report.split = await coverage('split');
    await page.evaluate((k) => window.psplat.setBudget(k), state0.count);
    report.splitFull = await coverage('split-full');
    await page.evaluate(() => window.psplat.setSplit(false));
    report.unsplitFull = await coverage('unsplit-full');

    // Projected-size threshold at full budget.
    await page.evaluate((k) => window.psplat.setBudget(k), state0.count);
    report.threshold = [];
    for (const px of [0, 2, 8, 1000]) {
        await page.evaluate((v) => window.psplat.setMinSigma(v), px);
        report.threshold.push({ px, image: await coverage(`minsigma-${px}`) });
    }
    await page.evaluate(() => window.psplat.setMinSigma(0));
    report.views = await page.evaluate(() => [window.psplat.setView('view_02'), window.psplat.state().view]);
    report.final = await page.evaluate(() => window.psplat.state());
    delete report.final.cuts;
} catch (err) {
    report.error = String(err && err.stack || err);
} finally {
    await browser.close();
}
const text = JSON.stringify(report, null, 2);
if (outIdx > 0) writeFileSync(process.argv[outIdx + 1], text);
console.log(text);

if (check) {
    const failures = [];
    const expect = (ok, what) => { if (!ok) failures.push(what); };
    expect(!report.error, `no error (${report.error})`);
    expect(!report.logs.some((l) => /^(error|pageerror)/.test(l)), 'no page errors');
    if (!report.error) {
        const means = report.budgets.map((b) => b.image.mean);
        expect(means.every((m, i) => i === 0 || m >= means[i - 1]), `brightness rises with budget: ${means}`);
        expect(means[means.length - 1] > 1.1 * means[0], 'the slider visibly changes the image');
        expect(report.budgets.every((b, i) => b.stats.slider === i), 'slider follows the budget');
        expect(Math.abs(report.split.meanRight - report.budgets.at(-1).image.meanRight) < 1e-9,
            'split screen shows the full scene on the right');
        expect(report.threshold.at(-1).image.lit === 0, 'a 1000 px threshold hides everything');
    }
    if (failures.length) {
        console.error('viewer check failed:\n  ' + failures.join('\n  '));
        process.exit(1);
    }
    console.error('viewer check passed');
}
