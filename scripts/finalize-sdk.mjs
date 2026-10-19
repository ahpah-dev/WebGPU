import { readFile, writeFile } from 'node:fs/promises';
const path = new URL('../dist/sdk/sdk.js', import.meta.url);
const source = await readFile(path, 'utf8');
if (!source.includes("new URL('./worker.ts', import.meta.url)")) throw new Error('Unexpected SDK worker entry.');
await writeFile(path, source.replace("new URL('./worker.ts', import.meta.url)", "new URL('./worker.js', import.meta.url)"));
