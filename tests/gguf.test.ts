import { test } from 'node:test';
import assert from 'node:assert/strict';
import { peekGGUF, peekGGUFHeader } from '../src/core/gguf.ts';
import { makeSource } from '../src/core/source.ts';

class Writer {
  private parts: number[] = [];
  u32(v: number) { const b = new Uint8Array(4); new DataView(b.buffer).setUint32(0, v, true); this.parts.push(...b); return this; }
  u64(v: number) { const b = new Uint8Array(8); new DataView(b.buffer).setBigUint64(0, BigInt(v), true); this.parts.push(...b); return this; }
  str(value: string) { const b = new TextEncoder().encode(value); this.u64(b.length); this.parts.push(...b); return this; }
  bytes() { return Uint8Array.from(this.parts); }
}
type Entry = [key: string, type: number, value: string | number];
function header(entries: Entry[], tensorCount = 0) {
  const w = new Writer().u32(0x46554747).u32(3).u64(tensorCount).u64(entries.length);
  for (const [key, type, value] of entries) {
    w.str(key).u32(type);
    if (type === 8) w.str(value as string); else w.u32(value as number);
  }
  return w.bytes();
}
test('peekGGUFHeader reports the architecture, name and quantization without loading weights', () => {
  const peek = peekGGUFHeader(header([['general.architecture', 8, 'llama'], ['general.name', 8, 'Llama 3.2 1B Instruct'], ['general.file_type', 4, 15]]));
  assert.deepEqual(peek, { architecture: 'llama', name: 'Llama 3.2 1B Instruct', quantization: 'Q4_K_M' });
});
test('peekGGUFHeader asks for more bytes when the metadata is truncated', () => {
  const full = header([['general.architecture', 8, 'qwen2'], ['general.name', 8, 'Qwen2.5 0.5B'], ['general.file_type', 4, 12]]);
  for (const cut of [4, 10, 20, full.length - 1]) assert.equal(peekGGUFHeader(full.subarray(0, cut)), undefined, `cut ${cut}`);
});
test('peekGGUFHeader rejects files that are not GGUF or have no architecture', () => {
  assert.throws(() => peekGGUFHeader(new Uint8Array(64)), /not a GGUF/);
  const missing = header([['general.name', 8, 'x']]);
  assert.throws(() => peekGGUFHeader(missing), /architecture/);
  const version = new Writer().u32(0x46554747).u32(9).u64(0).u64(0).bytes();
  assert.throws(() => peekGGUFHeader(version), /version/);
});
test('peekGGUF reads a small model header through the byte source', async () => {
  const bytes = header([['general.architecture', 8, 'qwen3'], ['general.name', 8, 'Qwen3 0.6B'], ['general.file_type', 4, 18]]);
  const file = new File([bytes], 'qwen3.gguf');
  const peek = await peekGGUF(makeSource(file));
  assert.equal(peek.architecture, 'qwen3'); assert.equal(peek.quantization, 'Q6_K');
});
