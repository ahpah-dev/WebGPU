import { test } from 'node:test';
import assert from 'node:assert/strict';
import { decodeWeight, halfToFloat } from '../src/core/quant.ts';
import { sample } from '../src/core/engine.ts';

test('IEEE half handles signs, subnormals, infinity, and NaN', () => {
  assert.equal(halfToFloat(0x3c00), 1); assert.equal(halfToFloat(0xc000), -2);
  assert.equal(halfToFloat(1), 2 ** -24); assert.equal(halfToFloat(0x7c00), Infinity);
  assert.ok(Number.isNaN(halfToFloat(0x7e00)));
});
test('Q4_0 and Q8_0 decode signed blocks including the odd-byte next block', () => {
  const q4 = new Uint8Array(36); q4[1] = q4[19] = 0x3c;
  for (let i = 0; i < 16; i++) q4[2 + i] = q4[20 + i] = i | (15 - i) << 4;
  for (let i = 0; i < 64; i++) assert.equal(decodeWeight(q4, 2, i), i % 32 < 16 ? i % 16 - 8 : 7 - i % 16);
  const q8 = new Uint8Array(68); q8[1] = q8[35] = 0x3c;
  for (let i = 0; i < 32; i++) q8[2 + i] = q8[36 + i] = 240 + i;
  for (let i = 0; i < 64; i++) assert.equal(decodeWeight(q8, 8, i), i % 32 - 16);
});
test('K-quants recover independently packed six-bit scales and affine minima', () => {
  for (const type of [12, 13]) {
    const data = new Uint8Array(type === 12 ? 144 : 176); data[1] = data[3] = 0x3c;
    const scales = [1, 17, 35, 63, 49, 21, 38, 54], mins = [62, 34, 18, 2, 57, 43, 29, 15];
    for (let s = 0; s < 8; s++) {
      if (s < 4) { data[4 + s] = scales[s]; data[8 + s] = mins[s]; }
      else { data[8 + s] = (scales[s] & 15) | ((mins[s] & 15) << 4); data[s] |= (scales[s] >>> 4) << 6; data[4 + s] |= (mins[s] >>> 4) << 6; }
    }
    for (let group = 0; group < 4; group++) for (let lane = 0; lane < 32; lane++) data[(type === 12 ? 16 : 48) + group * 32 + lane] = (lane & 15) | ((15 - (lane & 15)) << 4);
    if (type === 13) data.fill(0xa5, 16, 48);
    for (let i = 0; i < 256; i++) { const s = Math.floor(i / 32), q = (s % 2 ? 15 - i % 16 : i % 16) + (type === 13 && ((0xa5 >>> s) & 1) ? 16 : 0); assert.equal(decodeWeight(data, type, i), scales[s] * q - mins[s]); }
  }
});
test('Q6_K reconstructs all four interleaved quarters and signed scales', () => {
  const data = new Uint8Array(420), expected = new Float32Array(512);
  for (let block = 0; block < 2; block++) {
    const base = block * 210; data[base + 209] = 0x3c;
    for (let j = 0; j < 16; j++) data[base + 192 + j] = j - 8;
    for (let i = 0; i < 256; i++) {
      const value = (i * 7 + block) % 64, segment = Math.floor(i / 128), quarter = Math.floor(i % 128 / 32), lane = i % 32;
      data[base + segment * 64 + quarter % 2 * 32 + lane] |= (value & 15) << (Math.floor(quarter / 2) * 4);
      data[base + 128 + segment * 32 + lane] |= (value >>> 4) << (quarter * 2);
      expected[block * 256 + i] = (value - 32) * (Math.floor(i / 16) - 8);
    }
  }
  for (let i = 0; i < 512; i++) assert.equal(decodeWeight(data, 14, i), expected[i]);
});
test('greedy sampling rejects invalid logits and chooses the finite maximum', () => {
  assert.equal(sample(new Float32Array([NaN, -2, 7, 3]), 0, 40, .9), 2);
  assert.throws(() => sample(new Float32Array([NaN, -Infinity]), 0, 40, .9), /non-finite/);
});
