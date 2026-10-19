import { test } from 'node:test';
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { mkdtemp, mkdir, writeFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join, dirname, basename, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { spawn } from 'node:child_process';
import { createServer } from 'node:net';
import { OllamaCatalog } from '../server/ollama.mjs';

async function fixture(t: { after: (f: () => Promise<void>) => void }) {
  const root = await mkdtemp(join(tmpdir(), 'webgpu-ollama-'));
  await mkdir(join(root, 'blobs'));
  t.after(async () => {
    assert.equal(dirname(resolve(root)), resolve(tmpdir()));
    assert.ok(basename(root).startsWith('webgpu-ollama-'));
    await rm(root, { recursive: true, force: true });
  });
  return root;
}
async function model(root: string, parts = ['registry.ollama.ai', 'library', 'qwen3.8', '27b'], family = 'qwen35', marker = 'test') {
  const blob = async (bytes: Buffer) => {
    const hash = createHash('sha256').update(bytes).digest('hex'); await writeFile(join(root, 'blobs', 'sha256-' + hash), bytes);
    return { digest: 'sha256:' + hash, size: bytes.length };
  };
  const bytes = Buffer.from('GGUF' + marker), weights = await blob(bytes), config = await blob(Buffer.from(JSON.stringify({ model_family: family, model_type: '27B', file_type: 'Q4_K_M' })));
  const path = join(root, 'manifests', ...parts); await mkdir(dirname(path), { recursive: true });
  await writeFile(path, JSON.stringify({ config, layers: [{ mediaType: 'application/vnd.ollama.image.projector', digest: 'sha256:' + '0'.repeat(64) }, { mediaType: 'application/vnd.ollama.image.model', ...weights }] }));
  return { path, bytes, weights };
}
test('Ollama catalog discovers namespaced models, labels unsupported models and hides filesystem paths', async t => {
  const root = await fixture(t);
  await model(root);
  await model(root, ['registry.ollama.ai', 'team', 'fine-tune', 'v1'], 'gemma3');
  await model(root, ['registry.ollama.ai', 'library', 'qwen2.5', '0.5b'], 'qwen2');
  const catalog = new OllamaCatalog(root), models = await catalog.refresh();
  assert.deepEqual(models.filter(m => m.supported).map(m => m.name), ['qwen3.8:27b', 'team/fine-tune:v1']);
  const unsupported = models.find(m => m.name === 'qwen2.5:0.5b')!;
  assert.equal(unsupported.supported, false); assert.match(unsupported.reason, /qwen2/);
  assert.ok(models.every(m => !('path' in m) && !JSON.stringify(m).includes(root)));
  assert.equal(catalog.resolve(unsupported.id), undefined);
  assert.deepEqual(await new OllamaCatalog(join(root, 'not-installed')).refresh(), []);
});
test('Ollama refresh tolerates broken manifests, reports missing weights, and changes URLs when a tag changes', async t => {
  const root = await fixture(t), first = await model(root), catalog = new OllamaCatalog(root);
  const before = (await catalog.refresh())[0];
  await writeFile(join(dirname(first.path), 'broken'), '{');
  const replacement = await model(root, undefined, 'qwen35', 'replacement');
  const after = await catalog.refresh(); assert.equal(after.length, 1); assert.notEqual(after[0].url, before.url);
  assert.equal(catalog.resolve(before.id), undefined);
  await rm(join(root, 'blobs', replacement.weights.digest.replace(':', '-')));
  const missing = (await catalog.refresh())[0]; assert.equal(missing.supported, false); assert.match(missing.reason, /missing or incomplete/);
});
test('Ollama HTTP API opens selected weights with bounded ranges and rejects cross-origin access and arbitrary IDs', async t => {
  const root = await fixture(t); const first = await model(root);
  const second = await model(root, ['registry.ollama.ai', 'library', 'gemma3', '27b'], 'gemma3', 'other-model');
  const reservation = createServer(); await new Promise<void>(r => reservation.listen(0, '127.0.0.1', r));
  const address = reservation.address(); assert.ok(address && typeof address !== 'string'); const port = address.port;
  await new Promise<void>(r => reservation.close(() => r()));
  const child = spawn(process.execPath, ['server/serve.mjs', '--ollama-dir', root, '--port', String(port)], { cwd: fileURLToPath(new URL('..', import.meta.url)), stdio: ['ignore', 'pipe', 'pipe'], windowsHide: true });
  const exited = new Promise<void>(r => child.once('exit', () => r()));
  t.after(async () => { child.kill(); await exited; });
  await new Promise<void>((resolve, reject) => {
    const timer = setTimeout(() => reject(new Error('Local byte server did not start.')), 15000);
    child.stdout.on('data', data => { if (String(data).includes('WebGPU:')) { clearTimeout(timer); resolve(); } });
    child.once('exit', code => { clearTimeout(timer); if (code) reject(new Error('Byte server exited: ' + code)); });
  });
  const base = `http://127.0.0.1:${port}`;
  const response = await fetch(base + '/api/ollama/models'); assert.equal(response.status, 200);
  const { models } = await response.json(); assert.equal(models.length, 2);
  for (const [name, expected] of [['qwen3.8:27b', first], ['gemma3:27b', second]] as const) {
    const entry = models.find((m: { name: string }) => m.name === name);
    const weights = await fetch(base + entry.url, { headers: { Range: 'bytes=0-3' } });
    assert.equal(weights.status, 206); assert.equal(weights.headers.get('content-range'), `bytes 0-3/${expected.bytes.length}`);
    assert.equal(Buffer.from(await weights.arrayBuffer()).toString(), 'GGUF');
    assert.equal((await fetch(base + entry.url)).status, 400);
    assert.equal((await fetch(base + entry.url, { headers: { Range: 'bytes=0-999' } })).status, 416);
  }
  assert.equal((await fetch(base + '/api/ollama/models', { headers: { Origin: 'https://unrelated.example' } })).status, 403);
  assert.equal((await fetch(base + '/api/ollama/models', { method: 'POST' })).status, 405);
  assert.equal((await fetch(base + '/ollama/models/' + '0'.repeat(64) + '/weights.gguf', { headers: { Range: 'bytes=0-3' } })).status, 404);
});
