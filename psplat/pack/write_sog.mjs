// Write a SOG whose texels hold splats in a given order (psplat pack, P1.3).
//
// splat-transform's command line Morton-sorts splats when it writes SOG, which
// would destroy psplat's "first N are the best N" order. Its library writer
// takes `indices`, a full-length ordering that replaces that sort
// (docs/formats.md). psplat passes a banded order: splats stay within their
// cut-point band and are Morton-sorted only inside it.
//
//   node write_sog.mjs <splat-transform package dir> in.ply out.sog order.u32
//
// order.u32 is little-endian uint32, one source row per texel. No GPU device
// is passed, so spherical-harmonic palettes are clustered on the CPU; psplat
// decides beforehand whether that is affordable (psplat/pack/pack.py).

import { readFileSync, writeFileSync, mkdirSync, renameSync } from 'node:fs';
import { basename, dirname, join, resolve } from 'node:path';
import { pathToFileURL } from 'node:url';

const [libDir, input, output, orderFile] = process.argv.slice(2);
if (!libDir || !input || !output || !orderFile) {
    console.error('usage: node write_sog.mjs <splat-transform dir> in.ply out.sog order.u32');
    process.exit(2);
}
const st = await import(pathToFileURL(join(resolve(libDir), 'dist', 'index.mjs')).href);

const readFs = new st.MemoryReadFileSystem();
readFs.set(basename(input), readFileSync(input));
const [source] = await st.readFile({
    filename: basename(input),
    inputFormat: st.getInputFormat(input),
    fileSystem: readFs,
});
const pool = st.createChunkDataPool();
const dataTable = await st.materializeToDataTable(source, pool);
source.close?.();

const raw = readFileSync(orderFile);
const order = new Uint32Array(raw.buffer, raw.byteOffset, raw.byteLength / 4).slice();
if (order.length !== dataTable.numRows) {
    console.error(`order has ${order.length} entries but the scene has ${dataTable.numRows} splats`);
    process.exit(1);
}

const writeFs = new st.MemoryFileSystem();
await st.writeSog({
    filename: basename(output),
    bundle: true,
    iterations: 10,
    indices: order,
    dataTable,
    logging: 'silent',
}, writeFs);

const outDir = dirname(resolve(output));
mkdirSync(outDir, { recursive: true });
for (const [name, bytes] of writeFs.results) {
    const target = join(outDir, basename(name));
    writeFileSync(target + '.tmp', bytes);
    renameSync(target + '.tmp', target);
}
