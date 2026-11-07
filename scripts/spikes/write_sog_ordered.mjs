// Write a SOG that keeps the input PLY's splat order (P1.0 spike, #19).
//
// splat-transform's command line Morton-sorts splats when it writes SOG, which
// destroys psplat's "first N are the best N" ordering. Its library writer
// takes an `indices` option, a full-length ordering that replaces that sort.
// Passing the identity keeps file order in the texture pixel order.
//
//   node write_sog_ordered.mjs <splat-transform package dir> in.ply out.sog
//
// The package directory is passed explicitly so this script does not depend
// on where npm put node_modules relative to it.

import { readFileSync, writeFileSync, mkdirSync } from 'node:fs';
import { basename, dirname, join, resolve } from 'node:path';
import { pathToFileURL } from 'node:url';

const [libDir, input, output] = process.argv.slice(2);
if (!libDir || !input || !output) {
    console.error('usage: node write_sog_ordered.mjs <splat-transform dir> in.ply out.sog|meta.json');
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

const identity = new Uint32Array(dataTable.numRows);
for (let i = 0; i < identity.length; i++) identity[i] = i;

const bundle = !output.endsWith('meta.json');
const writeFs = new st.MemoryFileSystem();
await st.writeSog({
    filename: basename(output),
    bundle,
    iterations: 10,
    indices: identity,
    dataTable,
    logging: 'silent',
}, writeFs);

mkdirSync(dirname(resolve(output)), { recursive: true });
for (const [name, bytes] of writeFs.results) {
    writeFileSync(join(dirname(resolve(output)), basename(name)), bytes);
}
console.log(`wrote ${dataTable.numRows} splats in input order to ${output}`);
