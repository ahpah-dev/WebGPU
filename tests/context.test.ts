import { test } from 'node:test';
import assert from 'node:assert/strict';
import { attentionMemory, cacheCopies, outputLimit, validateContext } from '../src/core/context.ts';
import { PagedEngine } from '../src/core/engine.ts';
import type { ModelConfig } from '../src/core/gguf.ts';
import { mergeSymbols } from '../src/core/bpe.ts';

test('context respects integer, runtime, and model limits; output uses the remaining space', () => {
  for (const n of [256, 4096, 12000, 32768]) validateContext(n);
  for (const n of [255, 32769, 4096.5, NaN]) assert.throws(() => validateContext(n), /Context must/);
  assert.throws(() => validateContext(32768, 8192), /model declares/);
  assert.equal(outputLimit(32768, 100, 32768), 32668);
  assert.equal(outputLimit(8192, 24000, 32768), 8192);
  assert.equal(outputLimit(8192, 32767, 32768), 1);
  assert.throws(() => outputLimit(8192, 32768, 32768), /prompt has/);
  assert.throws(() => outputLimit(32769, 100, 32768), /maxTokens/);
});
test('Gemma local caches retain the full window for all queries in a batch, including ring wrap', () => {
  const c = { architecture: 'gemma3', layers: 6, period: 6, window: 1024, kvHeads: 4, headDim: 128 } as ModelConfig;
  const memory = attentionMemory(c, 32768, 128);
  assert.deepEqual(memory.capacities, [1151, 1151, 1151, 1151, 1151, 32768]);
  assert.equal(memory.totalBytes, (5 * 1151 + 32768) * 4 * 128 * 8);
  const capacity = 136, window = 129, batch = 8;
  for (const start of [128, 135, 136, 2047, 32760]) {
    const ring = new Array(capacity).fill(-1);
    for (let t = Math.max(0, start - capacity); t < start; t++) ring[t % capacity] = t;
    for (const copy of cacheCopies(start, batch, capacity)) for (let j = 0; j < copy.count; j++) ring[copy.target + j] = start + copy.source + j;
    for (let b = 0; b < batch; b++) for (let t = Math.max(0, start + b + 1 - window); t <= start + b; t++) assert.equal(ring[t % capacity], t);
  }
});
test('Qwen caches only full-attention layers and includes fixed recurrent state', () => {
  const c = { architecture: 'qwen35', layers: 4, kvHeads: 4, headDim: 256, recurrent: { keyHeads: 16, valueHeads: 48, headDim: 128, convKernel: 4, layers: [true, true, true, false] } } as ModelConfig;
  const memory = attentionMemory(c, 32768, 128);
  assert.deepEqual(memory.capacities, [0, 0, 0, 32768]);
  assert.equal(memory.kvBytes, 32768 * 4 * 256 * 8); assert.ok(memory.stateBytes > 0);
});
test('generation continues beyond 2048 output tokens and stops at the exact remaining context boundary', async () => {
  const engine = new PagedEngine(), positions: number[] = [];
  Object.assign(engine, {
    context: 4096, batch: 128, pager: { snapshot: () => ({}) }, config: { architecture: 'qwen35' },
    tokenizer: { chat: () => new Array(1000).fill(0), eos: new Set(), decode: (ids: number[]) => 'a'.repeat(ids.length) },
    forward: async (ids: number[], start: number) => { positions.push(start + ids.length - 1); return new Float32Array([1]); },
  });
  let count = 0;
  for await (const token of engine.generate([{ role: 'user', content: 'x'.repeat(20000) }], { maxTokens: 32768, temperature: 0 })) {
    count++; assert.equal(token.promptTokens, 1000); assert.equal(token.contextLength, 4096);
  }
  assert.equal(count, 3096); assert.equal(Math.max(...positions), 4094);
});
test('heap BPE matches the scan reference with score ties and stale neighboring pairs', () => {
  const score = (a: string, b: string) => (a + b).length <= 6 ? ((a + b).split('').reduce((s, c) => s + c.charCodeAt(0), 0) % 4) - 2 : undefined;
  for (let seed = 0; seed < 200; seed++) {
    let state = seed + 1;
    const input = Array.from({ length: 100 }, () => { state = (Math.imul(state, 1664525) + 1013904223) >>> 0; return 'abc'[state % 3]; }), reference = [...input];
    while (reference.length > 1) {
      let best = -Infinity, at = -1;
      for (let i = 0; i < reference.length - 1; i++) { const value = score(reference[i], reference[i + 1]); if (value !== undefined && value > best) { best = value; at = i; } }
      if (at < 0) break; reference.splice(at, 2, reference[at] + reference[at + 1]);
    }
    assert.deepEqual(mergeSymbols([...input], score), reference);
  }
  const long = mergeSymbols(Array.from('ab'.repeat(100000)), (a, b) => a === 'a' && b === 'b' ? 1 : undefined);
  assert.equal(long.length, 100000); assert.ok(long.every(s => s === 'ab'));
});
