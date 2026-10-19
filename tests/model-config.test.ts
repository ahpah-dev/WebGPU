import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { modelConfig, type GGUF, type Tensor } from '../src/core/gguf.ts';
const fixture = JSON.parse(readFileSync(new URL('./fixtures/qwen38-shapes.json', import.meta.url), 'utf8'));
function model(): GGUF {
  const tensors = new Map<string, Tensor>(fixture.tensors.map(([name, dims]: [string, number[]]) => [name, { name, dims, type: 0, offset: 0, bytes: 0, rowBytes: 0 }]));
  return { metadata: new Map(Object.entries(fixture.metadata)), tensors, dataOffset: 0, parameters: 0 };
}
export function denseGGUF(architecture: string, options: { bias?: boolean; qkNorm?: boolean; gate?: boolean; fullKv?: boolean } = {}): GGUF {
  const hidden = 64, heads = 4, kvHeads = options.fullKv ? 4 : 2, headDim = 16, layers = 2, intermediate = 128, vocab = 96;
  const tensors: [string, number[]][] = [['token_embd.weight', [hidden, vocab]], ['output_norm.weight', [hidden]], ['output.weight', [hidden, vocab]]];
  for (let l = 0; l < layers; l++) {
    const prefix = `blk.${l}.`;
    tensors.push([prefix + 'attn_norm.weight', [hidden]], [prefix + 'ffn_norm.weight', [hidden]]);
    tensors.push([prefix + 'attn_q.weight', [hidden, heads * headDim]]);
    tensors.push([prefix + 'attn_k.weight', [hidden, kvHeads * headDim]]);
    tensors.push([prefix + 'attn_v.weight', [hidden, kvHeads * headDim]]);
    tensors.push([prefix + 'attn_output.weight', [heads * headDim, hidden]]);
    if (options.bias) tensors.push([prefix + 'attn_q.bias', [heads * headDim]], [prefix + 'attn_k.bias', [kvHeads * headDim]], [prefix + 'attn_v.bias', [kvHeads * headDim]]);
    if (options.qkNorm) tensors.push([prefix + 'attn_q_norm.weight', [headDim]], [prefix + 'attn_k_norm.weight', [headDim]]);
    if (options.gate !== false) tensors.push([prefix + 'ffn_gate.weight', [hidden, intermediate]]);
    tensors.push([prefix + 'ffn_up.weight', [hidden, intermediate]], [prefix + 'ffn_down.weight', [intermediate, hidden]]);
  }
  return {
    metadata: new Map<string, unknown>(Object.entries({
      'general.architecture': architecture, 'general.name': `${architecture} test`,
      [`${architecture}.block_count`]: layers, [`${architecture}.embedding_length`]: hidden,
      [`${architecture}.attention.head_count`]: heads, [`${architecture}.attention.head_count_kv`]: kvHeads,
      [`${architecture}.attention.key_length`]: headDim, [`${architecture}.feed_forward_length`]: intermediate,
      [`${architecture}.context_length`]: 8192,
    })),
    tensors: new Map(tensors.map(([name, dims]) => [name, { name, dims, type: 0, offset: 0, bytes: 0, rowBytes: 0 }])),
    dataOffset: 0, parameters: 0,
  };
}
test('real Qwen3.8 GGUF shapes select 64 text layers and exclude the MTP prediction layer', () => {
  const c = modelConfig(model());
  assert.equal(c.architecture, 'qwen35'); assert.equal(c.family, 'qwen35'); assert.equal(c.layers, 64);
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
  const unknown = denseGGUF('falcon');
  assert.throws(() => modelConfig(unknown), /Supported text architectures/);
});
test('Llama 3 GGUF selects the dense decoder with full-context attention and no Q/K norm', () => {
  const c = modelConfig(denseGGUF('llama'));
  assert.equal(c.family, 'llama'); assert.equal(c.layers, 2); assert.equal(c.heads, 4); assert.equal(c.kvHeads, 2);
  assert.equal(c.headDim, 16); assert.equal(c.rotaryDim, 16); assert.equal(c.theta, 10_000);
  assert.equal(c.attentionScale, 0.25); assert.equal(c.activation, 'silu');
  assert.equal(c.qkNorm, false); assert.deepEqual(c.bias, { q: false, k: false, v: false, output: false });
  assert.equal(c.ffnGate, true);
  // period 0 means every layer attends over the whole context; no sliding ring.
  assert.equal(c.period, 0);
  // Llama GGUF files store Q/K permuted for adjacent-pair rotation.
  assert.equal(c.rope, 'norm'); assert.equal(c.slidingWindow, 0);
});
test('RoPE scaling and sliding windows are read from the GGUF metadata', () => {
  const mistral = denseGGUF('llama');
  mistral.metadata.set('llama.attention.sliding_window', 4096);
  mistral.metadata.set('llama.rope.scaling.type', 'linear');
  mistral.metadata.set('llama.rope.scaling.factor', 4);
  const c = modelConfig(mistral);
  assert.equal(c.slidingWindow, 4096); assert.equal(c.ropeScale, 0.25);
  const llama31 = denseGGUF('llama');
  llama31.metadata.set('llama.rope.scaling.type', 'llama3');
  llama31.metadata.set('llama.rope.scaling.factor', 8);
  // Frequency-band scaling is not a plain divisor; positions stay unscaled.
  assert.equal(modelConfig(llama31).ropeScale, 1);
  const qwen = denseGGUF('qwen2'); qwen.metadata.set('qwen2.rope.scaling.type', 'linear');
  qwen.metadata.set('qwen2.rope.scaling.factor', 2);
  assert.equal(modelConfig(qwen).ropeScale, 0.5);
});
test('Qwen2 attention biases and Qwen3 Q/K normalization are detected from the tensors', () => {
  const qwen2 = modelConfig(denseGGUF('qwen2', { bias: true }));
  assert.deepEqual(qwen2.bias, { q: true, k: true, v: true, output: false });
  assert.equal(qwen2.qkNorm, false);
  assert.equal(qwen2.rope, 'neox');
  const qwen3 = modelConfig(denseGGUF('qwen3', { qkNorm: true }));
  assert.equal(qwen3.qkNorm, true);
  assert.equal(qwen3.rope, 'neox');
  assert.deepEqual(qwen3.bias, { q: false, k: false, v: false, output: false });
});
test('Llama 2 style files without ffn_gate use the ungated MLP path', () => {
  assert.equal(modelConfig(denseGGUF('llama', { gate: false })).ffnGate, false);
});
test('Dense model validation rejects MoE weights, malformed shapes, and missing layers', () => {
  const moe = denseGGUF('llama'); moe.tensors.set('blk.0.ffn_gate_inp.weight', { name: 'blk.0.ffn_gate_inp.weight', dims: [64, 4], type: 0, offset: 0, bytes: 0, rowBytes: 0 });
  assert.throws(() => modelConfig(moe), /Mixture-of-experts/);
  const badShape = denseGGUF('llama'); badShape.tensors.get('blk.1.attn_v.weight')!.dims = [64, 64];
  assert.throws(() => modelConfig(badShape), /attn_v/);
  const missing = denseGGUF('qwen2', { bias: true }); missing.tensors.delete('blk.1.attn_k.bias');
  assert.throws(() => modelConfig(missing), /attn_k\.bias/);
  const biasMissingButDetected = denseGGUF('llama'); biasMissingButDetected.tensors.delete('blk.1.ffn_norm.weight');
  assert.throws(() => modelConfig(biasMissingButDetected), /ffn_norm/);
});
test('Dense models fall back to embedding_length / head_count when key_length is absent', () => {
  const g = denseGGUF('llama', { fullKv: true }); g.metadata.delete('llama.attention.key_length'); g.metadata.delete('llama.attention.head_count_kv');
  const c = modelConfig(g);
  assert.equal(c.headDim, 16); assert.equal(c.kvHeads, 4); assert.equal(c.rotaryDim, 16);
});
