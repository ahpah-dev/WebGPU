import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { modelConfig, type GGUF, type Tensor } from '../src/core/gguf.ts';
const fixture = JSON.parse(readFileSync(new URL('./fixtures/qwen38-shapes.json', import.meta.url), 'utf8'));
function model(): GGUF {
  const tensors = new Map<string, Tensor>(fixture.tensors.map(([name, dims]: [string, number[]]) => [name, { name, dims, type: 0, offset: 0, bytes: 0, rowBytes: 0 }]));
  return { metadata: new Map(Object.entries(fixture.metadata)), tensors, dataOffset: 0, parameters: 0 };
}
test('real Qwen3.8 GGUF shapes select 64 text layers and exclude the MTP prediction layer', () => {
  const c = modelConfig(model());
  assert.equal(c.architecture, 'qwen35'); assert.equal(c.layers, 64);
  assert.equal(c.parameters, 26895998464); assert.equal(c.rotaryDim, 64);
  assert.equal(c.recurrent!.layers.filter(Boolean).length, 48);
  assert.deepEqual(c.recurrent!.layers.slice(0, 8), [true, true, true, false, true, true, true, false]);
  assert.equal(c.attentionScale, 1 / 16);
});
test('Qwen model validation rejects missing recurrent matrices and invalid MTP metadata', () => {
  const missing = model(); missing.tensors.delete('blk.0.ssm_out.weight');
  assert.throws(() => modelConfig(missing), /ssm_out/);
  const invalid = model(); invalid.metadata.set('qwen35.nextn_predict_layers', -1);
  assert.throws(() => modelConfig(invalid), /prediction layer/);
  const dimensions = model(); dimensions.metadata.set('qwen35.rope.dimension_count', 257);
  assert.throws(() => modelConfig(dimensions), /dimensions/);
});
test('Qwen layer map validates explicit recurrent flags instead of guessing unknown layouts', () => {
  const g = model(); g.metadata.set('qwen35.attention.recurrent_layers', [true]);
  assert.throws(() => modelConfig(g), /layer map/);
  const unknown = model(); unknown.metadata.set('general.architecture', 'qwen3');
  assert.throws(() => modelConfig(unknown), /Supported text architectures/);
});
