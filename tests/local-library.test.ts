import { test } from 'node:test';
import assert from 'node:assert/strict';
import { registerHooks } from 'node:module';

// Node cannot execute a stylesheet import. The browser build receives the real CSS
// through Vite; the test runner only needs an inert module for that specifier.
registerHooks({
  load(url, context, nextLoad) {
    if (url.endsWith('.css')) return { format: 'module', source: 'export default undefined;', shortCircuit: true };
    return nextLoad(url, context);
  },
});

const hadBrowserGlobals = typeof document !== 'undefined' || typeof window !== 'undefined';
const globalsBeforeImport = new Set(Object.keys(globalThis));
const library = await import('../src/local-library.ts');
const addedGlobals = Object.keys(globalThis).filter(key => !globalsBeforeImport.has(key));

const {
  modelNameFromManifestPath,
  manifestModelLayer,
  blobNameForDigest,
  isSupportedArchitecture,
  mergeModelEntries,
  createLibraryPicker,
} = library;

const DIGEST_A = 'sha256:' + 'a'.repeat(64);
const DIGEST_B = 'sha256:' + 'b'.repeat(64);
const DIGEST_C = 'sha256:' + 'c'.repeat(64);
const DIGEST_D = 'sha256:' + 'd'.repeat(64);

test('manifest paths omit the default registry and library namespace', () => {
  assert.equal(modelNameFromManifestPath(['registry.ollama.ai', 'library', 'qwen2.5', '0.5b']), 'qwen2.5:0.5b');
  assert.equal(modelNameFromManifestPath(['hf.co', 'bartowski', 'Vendor', 'Q4_K_M']), 'hf.co/bartowski/Vendor:Q4_K_M');
  assert.equal(modelNameFromManifestPath(['registry.ollama.ai', 'team', 'fine-tune', 'v1']), 'team/fine-tune:v1');
  assert.equal(modelNameFromManifestPath(['registry.ollama.ai', 'library', 'org', 'model', 'latest']), 'org/model:latest');
  assert.equal(modelNameFromManifestPath(['example.com', 'library', 'model', 'q4']), 'example.com/model:q4');
  assert.throws(() => modelNameFromManifestPath(['registry.ollama.ai', 'library']), /Invalid Ollama manifest path/);
  assert.throws(() => modelNameFromManifestPath(['registry.ollama.ai', '', 'model', 'tag']), /Invalid Ollama manifest path/);
});

test('a realistic Ollama manifest exposes its local weight layer', () => {
  const manifest = {
    schemaVersion: 2,
    mediaType: 'application/vnd.docker.distribution.manifest.v2+json',
    config: { mediaType: 'application/vnd.ollama.image.config', digest: DIGEST_A, size: 767 },
    layers: [
      { mediaType: 'application/vnd.ollama.image.license', digest: DIGEST_B, size: 1234 },
      { mediaType: 'application/vnd.ollama.image.model', digest: DIGEST_C, size: 123456789 },
    ],
  };
  assert.deepEqual(manifestModelLayer(manifest), { digest: DIGEST_C, size: 123456789 });
  assert.equal(manifestModelLayer({ layers: [{ mediaType: 'application/vnd.ollama.image.projector', digest: DIGEST_C, size: 1 }] }), undefined);
  assert.equal(manifestModelLayer({ layers: [{ mediaType: 'application/vnd.ollama.image.model', digest: 'sha256:xyz', size: 1 }] }), undefined);
  assert.equal(manifestModelLayer({ layers: [{ mediaType: 'application/vnd.ollama.image.model', digest: DIGEST_C, size: -1 }] }), undefined);
  assert.equal(manifestModelLayer({ layers: [{ mediaType: 'application/vnd.ollama.image.model', digest: DIGEST_C }] }), undefined);
  assert.equal(manifestModelLayer({ layers: 'not-a-list' }), undefined);
  assert.equal(manifestModelLayer(null), undefined);
  assert.equal(manifestModelLayer('{}'), undefined);
});

test('blob names follow the Ollama digest format and reject anything else', () => {
  const hex = '0123456789abcdef'.repeat(4);
  assert.equal(blobNameForDigest(`sha256:${hex}`), `sha256-${hex}`);
  assert.throws(() => blobNameForDigest(`sha256:${'A'.repeat(64)}`), /Invalid Ollama model digest/);
  assert.throws(() => blobNameForDigest(`sha256:${'a'.repeat(63)}`), /Invalid Ollama model digest/);
  assert.throws(() => blobNameForDigest(`sha256-${'a'.repeat(64)}`), /Invalid Ollama model digest/);
  assert.throws(() => blobNameForDigest(`sha256:${'a'.repeat(64)}\n`), /Invalid Ollama model digest/);
  assert.throws(() => blobNameForDigest('sha256:not-hex'), /Invalid Ollama model digest/);
});

test('only the architectures implemented by the browser engine are supported', () => {
  for (const architecture of ['gemma3', 'qwen35', 'llama', 'qwen2', 'qwen3']) assert.equal(isSupportedArchitecture(architecture), true);
  for (const architecture of ['gemma2', 'phi3', 'mixtral', 'QWEN3', '']) assert.equal(isSupportedArchitecture(architecture), false);
});

function entry(digest, name, supported = true) {
  return { digest, name, size: 1024, architecture: '', quantization: '', supported, reason: supported ? '' : 'The weights are missing.' };
}

test('entries dedupe by digest, keep the least-qualified name and sort supported models first', () => {
  const handle = { kind: 'file', name: 'sha256-' + '1'.repeat(64) };
  const longer = { ...entry(DIGEST_A, 'hf.co/bartowski/vendor:q4_k_m'), handle };
  const shorter = entry(DIGEST_A, 'vendor:q4_k_m');
  const merged = mergeModelEntries([
    longer,
    shorter,
    entry(DIGEST_B, 'zeta'),
    entry(DIGEST_C, 'alpha', false),
    entry(DIGEST_D, 'alpha'),
  ]);
  assert.equal(merged.length, 4);
  assert.deepEqual(merged.map(model => model.name), ['alpha', 'vendor:q4_k_m', 'zeta', 'alpha']);
  assert.equal(merged.filter(model => model.digest === DIGEST_A).length, 1);
  const kept = merged.find(model => model.digest === DIGEST_A);
  assert.equal(kept.name, 'vendor:q4_k_m');
  assert.equal(kept.handle, handle, 'the usable file source survives when the shorter entry had none');
  assert.equal(merged[merged.length - 1].supported, false);
});

test('importing the module in Node does not touch document or window', () => {
  assert.equal(hadBrowserGlobals, false);
  assert.equal(typeof document, 'undefined');
  assert.equal(typeof window, 'undefined');
  assert.equal(typeof createLibraryPicker, 'function');
  assert.deepEqual(addedGlobals, []);
});
