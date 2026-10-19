import type { ByteSource } from './source.ts';

export type Metadata = Map<string, unknown>;
export interface Tensor { name: string; dims: number[]; type: number; offset: number; bytes: number; rowBytes: number }
export interface GGUF { metadata: Metadata; tensors: Map<string, Tensor>; dataOffset: number; parameters: number }
export const QUANTS: Record<number, { name: string; block: number; bytes: number }> = {
  0: { name: 'F32', block: 1, bytes: 4 }, 1: { name: 'F16', block: 1, bytes: 2 },
  2: { name: 'Q4_0', block: 32, bytes: 18 }, 8: { name: 'Q8_0', block: 32, bytes: 34 },
  12: { name: 'Q4_K', block: 256, bytes: 144 }, 13: { name: 'Q5_K', block: 256, bytes: 176 },
  14: { name: 'Q6_K', block: 256, bytes: 210 },
};
class NeedMore extends Error {}
class Reader {
  offset = 0;
  view: DataView;
  decoder = new TextDecoder();
  bytes: Uint8Array;
  constructor(bytes: Uint8Array) { this.bytes = bytes; this.view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength); }
  need(n: number) { if (this.offset + n > this.bytes.length) throw new NeedMore(); }
  u32() { this.need(4); const v = this.view.getUint32(this.offset, true); this.offset += 4; return v; }
  u64() {
    this.need(8); const v = this.view.getBigUint64(this.offset, true); this.offset += 8;
    if (v > BigInt(Number.MAX_SAFE_INTEGER)) throw new Error('GGUF integer exceeds safe file addressing.');
    return Number(v);
  }
  string() { const n = this.u64(); if (n > 16 * 1024 ** 2) throw new Error('Oversized GGUF string.'); this.need(n); const v = this.decoder.decode(this.bytes.subarray(this.offset, this.offset + n)); this.offset += n; return v; }
  value(type: number, depth = 0): unknown {
    if (depth > 2) throw new Error('Invalid nested metadata.');
    if (type === 8) return this.string();
    if (type === 9) {
      const t = this.u32(), n = this.u64();
      if (n > 2_000_000) throw new Error('Oversized GGUF array.');
      return Array.from({ length: n }, () => this.value(t, depth + 1));
    }
    if (type === 10) return this.u64();
    const widths: Record<number, number> = { 0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 11: 8, 12: 8 };
    const w = widths[type]; if (!w) throw new Error(`Unknown GGUF metadata type ${type}.`);
    this.need(w); const p = this.offset; this.offset += w;
    switch (type) {
      case 0: return this.view.getUint8(p); case 1: return this.view.getInt8(p);
      case 2: return this.view.getUint16(p, true); case 3: return this.view.getInt16(p, true);
      case 4: return this.view.getUint32(p, true); case 5: return this.view.getInt32(p, true);
      case 6: return this.view.getFloat32(p, true); case 7: return this.view.getUint8(p) !== 0;
      case 11: return Number(this.view.getBigInt64(p, true)); case 12: return this.view.getFloat64(p, true);
    }
  }
}
export function parseGGUFHeader(bytes: Uint8Array, fileSize: number): GGUF {
  const r = new Reader(bytes);
  if (r.u32() !== 0x46554747) throw new Error('Select a GGUF model file.');
  const version = r.u32(); if (version !== 2 && version !== 3) throw new Error(`GGUF version ${version} is unsupported.`);
  const count = r.u64(), kvCount = r.u64();
  if (count > 100_000 || kvCount > 100_000) throw new Error('Invalid GGUF counts.');
  const metadata: Metadata = new Map();
  for (let i = 0; i < kvCount; i++) { const key = r.string(); metadata.set(key, r.value(r.u32())); }
  const tensors = new Map<string, Tensor>(); let parameters = 0;
  for (let i = 0; i < count; i++) {
    const name = r.string(), nd = r.u32();
    if (nd < 1 || nd > 4 || tensors.has(name)) throw new Error('Invalid tensor directory.');
    const dims = Array.from({ length: nd }, () => r.u64());
    const type = r.u32(), offset = r.u64(), q = QUANTS[type];
    if (!q) throw new Error(`Unsupported tensor quantization ${type} in ${name}. Use Q4_K_M, Q5_K_M, Q6_K or Q8_0.`);
    const n = dims.reduce((a, b) => a * b, 1);
    if (!Number.isSafeInteger(n) || dims.some(d => d < 1) || dims[0] % q.block !== 0) throw new Error(`Invalid shape for ${name}.`);
    const rowBytes = dims[0] / q.block * q.bytes;
    tensors.set(name, { name, dims, type, offset, rowBytes, bytes: n / q.block * q.bytes });
    parameters += n;
  }
  const alignment = Number(metadata.get('general.alignment') ?? 32);
  if (!Number.isInteger(alignment) || alignment < 1 || alignment > 4096 || (alignment & (alignment - 1))) throw new Error('Invalid GGUF alignment.');
  const dataOffset = Math.ceil(r.offset / alignment) * alignment;
  for (const t of tensors.values()) {
    t.offset += dataOffset;
    if (t.offset + t.bytes > fileSize) throw new Error(`Truncated model: ${t.name}.`);
  }
  return { metadata, tensors, dataOffset, parameters };
}
export async function readGGUF(source: ByteSource, signal?: AbortSignal): Promise<GGUF> {
  for (let n = Math.min(source.size, 4 * 1024 ** 2); ; n = Math.min(source.size, n * 2)) {
    try { return parseGGUFHeader(await source.read(0, n, signal), source.size); }
    catch (e) { if (!(e instanceof NeedMore)) throw e; if (n >= source.size || n >= 64 * 1024 ** 2) throw new Error('GGUF header exceeds 64 MiB or is truncated.'); }
  }
}
export interface ModelConfig {
  hidden: number; intermediate: number; layers: number; heads: number; kvHeads: number; headDim: number;
  vocab: number; epsilon: number; theta: number; localTheta: number; window: number; period: number;
  attentionScale: number; ropeScale: number; name: string; parameters: number;
}
export function modelConfig(g: GGUF): ModelConfig {
  if (g.metadata.get('general.architecture') !== 'gemma3') throw new Error('This runtime currently implements Gemma 3 text inference. Other model architectures are rejected explicitly.');
  const num = (key: string, fallback?: number) => {
    const v = Number(g.metadata.get(`gemma3.${key}`) ?? fallback);
    if (!Number.isFinite(v) || v <= 0) throw new Error(`Missing or invalid Gemma 3 metadata: ${key}.`);
    return v;
  };
  const layers = num('block_count'), hidden = num('embedding_length'), heads = num('attention.head_count');
  const headDim = num('attention.key_length');
  const cfg: ModelConfig = {
    hidden, layers, heads, headDim, kvHeads: num('attention.head_count_kv'), intermediate: num('feed_forward_length'),
    vocab: g.tensors.get('token_embd.weight')?.dims[1] ?? 0, epsilon: num('attention.layer_norm_rms_epsilon', 1e-6),
    theta: num('rope.freq_base', 1_000_000), localTheta: num('rope.local.freq_base', 10_000),
    window: num('attention.sliding_window', 1024), period: 6,
    attentionScale: 1 / Math.sqrt(layers === 62 ? hidden / heads : headDim),
    ropeScale: 1 / num('rope.scaling.factor', 1),
    name: String(g.metadata.get('general.name') ?? 'Gemma 3'), parameters: g.parameters,
  };
  if (heads % cfg.kvHeads || headDim % 2 || !cfg.vocab || layers > 128 || hidden > 16384 || cfg.intermediate > 131072 || cfg.vocab > 1_000_000) throw new Error('Unsupported Gemma 3 dimensions.');
  const shape = (name: string, dims: number[]) => {
    const t = g.tensors.get(name);
    if (!t || t.dims.join(',') !== dims.join(',')) throw new Error(`Missing or invalid tensor ${name}.`);
  };
  shape('token_embd.weight', [hidden, cfg.vocab]); shape('output_norm.weight', [hidden]);
  if (g.tensors.has('output.weight')) shape('output.weight', [hidden, cfg.vocab]);
  for (let l = 0; l < layers; l++) {
    const prefix = `blk.${l}.`;
    for (const key of ['attn_norm', 'post_attention_norm', 'ffn_norm', 'post_ffw_norm']) shape(prefix + key + '.weight', [hidden]);
    for (const key of ['attn_q_norm', 'attn_k_norm']) shape(prefix + key + '.weight', [headDim]);
    shape(prefix + 'attn_q.weight', [hidden, heads * headDim]);
    for (const key of ['attn_k', 'attn_v']) shape(prefix + key + '.weight', [hidden, cfg.kvHeads * headDim]);
    shape(prefix + 'attn_output.weight', [heads * headDim, hidden]);
    for (const key of ['ffn_gate', 'ffn_up']) shape(prefix + key + '.weight', [hidden, cfg.intermediate]);
    shape(prefix + 'ffn_down.weight', [cfg.intermediate, hidden]);
  }
  return cfg;
}
