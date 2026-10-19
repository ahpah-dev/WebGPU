import { test } from 'node:test';
import assert from 'node:assert/strict';
import { makeSource } from '../src/core/source.ts';
import { parseGGUFHeader } from '../src/core/gguf.ts';

test('File source reads only the selected byte range', async () => {
  const file = new File([Uint8Array.from({ length: 100 }, (_, i) => i)], 'fixture.gguf');
  const source = makeSource(file);
  assert.deepEqual([...await source.read(75, 5)], [75, 76, 77, 78, 79]);
  await assert.rejects(source.read(99, 2), /Invalid model byte range/);
  await assert.rejects(source.read(0, 0), /Invalid model byte range/);
  await assert.rejects(source.read(-1, 1), /Invalid model byte range/);
});
test('HTTP source refuses servers that would download the entire model', async () => {
  const original = globalThis.fetch; let cancelled = false;
  globalThis.fetch = async () => new Response(new ReadableStream({ cancel() { cancelled = true; } }), { status: 200 });
  try { await assert.rejects(makeSource({ url: 'http://localhost/test.gguf', size: 100 }).read(0, 16), /Range/); assert.ok(cancelled); }
  finally { globalThis.fetch = original; }
});
test('HTTP source detects incorrect Content-Range and truncated responses', async () => {
  const original = globalThis.fetch;
  try {
    globalThis.fetch = async () => new Response(new Uint8Array(4), { status: 206, headers: { 'Content-Range': 'bytes 0-4/100' } });
    await assert.rejects(makeSource({ url: 'http://localhost/test.gguf', size: 100 }).read(0, 5), /Truncated/);
    globalThis.fetch = async () => new Response(new Uint8Array(5), { status: 206, headers: { 'Content-Range': 'bytes 1-5/100' } });
    await assert.rejects(makeSource({ url: 'http://localhost/test.gguf', size: 100 }).read(0, 5), /Range/);
  } finally { globalThis.fetch = original; }
});
test('reader rejects unrecognized files and unsupported GGUF versions', () => {
  assert.throws(() => parseGGUFHeader(new Uint8Array(24), 24), /GGUF/);
  const bytes = new Uint8Array(24), view = new DataView(bytes.buffer); view.setUint32(0, 0x46554747, true); view.setUint32(4, 99, true);
  assert.throws(() => parseGGUFHeader(bytes, 24), /version/);
});

test('removing a served model reports an unavailable file instead of a Range configuration error', async () => {
  const original = globalThis.fetch;
  globalThis.fetch = async () => new Response('', { status: 404 });
  try { await assert.rejects(makeSource({ url: 'http://localhost/test.gguf', size: 100 }).read(0, 16), /model file is no longer available/); }
  finally { globalThis.fetch = original; }
});
