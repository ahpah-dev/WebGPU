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
export type Family = 'gemma3' | 'qwen35' | 'llama';
/** GGUF architectures that share the standard dense decoder: RMSNorm, RoPE, GQA, SwiGLU. */
const DENSE = new Set(['llama', 'qwen2', 'qwen3']);
export interface ModelConfig {
  architecture: string;
  family: Family;
  hidden: number; intermediate: number; layers: number; heads: number; kvHeads: number; headDim: number;
  vocab: number; epsilon: number; theta: number; localTheta: number; window: number; period: number;
  attentionScale: number; ropeScale: number; name: string; parameters: number;
  rotaryDim: number;
  maxContext: number;
  /** RoPE pairing convention. GGUF files for llama/mistral store Q/K permuted for
   * adjacent-pair rotation; Qwen2/3, Gemma and Qwen3.8 use the half-split (NeoX) layout. */
  rope: 'neox' | 'norm';
  /** Non-zero restricts attention to the most recent tokens (Mistral sliding window). */
  slidingWindow: number;
  /** Per-head RMS normalization of Q and K with learned weights (Qwen3 style). */
  qkNorm: boolean;
  /** Gated feed-forward. Some Llama 2 era files store a plain MLP without ffn_gate. */
  ffnGate: boolean;
  activation: 'gelu' | 'silu';
  bias: { q: boolean; k: boolean; v: boolean; output: boolean };
  recurrent?: { keyHeads: number; valueHeads: number; headDim: number; convKernel: number; layers: boolean[] };
}
export function modelConfig(g: GGUF): ModelConfig {
  const raw = g.metadata.get('general.architecture');
  if (typeof raw !== 'string' || !raw) throw new Error('The GGUF file has no architecture metadata.');
  const architecture = raw;
  const family: Family | undefined = architecture === 'gemma3' ? 'gemma3' : architecture === 'qwen35' ? 'qwen35' : DENSE.has(architecture) ? 'llama' : undefined;
  if (!family) throw new Error(`Supported text architectures: Gemma 3, Qwen3.5, Qwen3.8, Llama 3.x and Mistral (llama), Qwen2 and Qwen3. The ${architecture} architecture is not supported.`);
  const num = (key: string, fallback?: number) => {
    const v = Number(g.metadata.get(`${architecture}.${key}`) ?? fallback);
    if (!Number.isFinite(v) || v <= 0) throw new Error(`Missing or invalid ${architecture} metadata: ${key}.`);
    return v;
  };
  const opt = (key: string, fallback: number) => g.metadata.get(`${architecture}.${key}`) === undefined ? fallback : num(key);
  const extra = architecture === 'qwen35' ? Number(g.metadata.get('qwen35.nextn_predict_layers') ?? 0) : 0;
  if (!Number.isInteger(extra) || extra < 0 || extra > 8) throw new Error('Invalid prediction layer count.');
  const layers = num('block_count') - extra, hidden = num('embedding_length'), heads = num('attention.head_count');
  const headDim = family === 'llama' ? opt('attention.key_length', hidden / heads) : num('attention.key_length');
  const cfg: ModelConfig = {
    architecture, family, hidden, layers, heads, headDim,
    kvHeads: family === 'llama' ? opt('attention.head_count_kv', heads) : num('attention.head_count_kv'),
    intermediate: num('feed_forward_length'),
    vocab: g.tensors.get('token_embd.weight')?.dims[1] ?? 0, epsilon: num('attention.layer_norm_rms_epsilon', 1e-6),
    theta: family === 'llama' ? opt('rope.freq_base', 10_000) : num('rope.freq_base', 1_000_000),
    localTheta: family === 'llama' ? opt('rope.freq_base', 10_000) : num('rope.local.freq_base', 10_000),
    window: family === 'llama' ? 0 : num('attention.sliding_window', 1024),
    // 0 means every layer attends over the full context; no sliding window at all.
    period: family === 'gemma3' ? 6 : family === 'qwen35' ? num('full_attention_interval', 4) : 0,
    attentionScale: 1 / Math.sqrt(family === 'gemma3' && layers === 62 ? hidden / heads : headDim),
    ropeScale: 1,
    name: String(g.metadata.get('general.name') ?? architecture), parameters: g.parameters,
    rotaryDim: family === 'llama' ? opt('rope.dimension_count', headDim) : family === 'gemma3' ? headDim : num('rope.dimension_count'),
    maxContext: num('context_length', 32768),
    rope: family === 'llama' && architecture === 'llama' ? 'norm' : 'neox',
    slidingWindow: family === 'llama' ? Math.floor(opt('attention.sliding_window', 0)) : family === 'gemma3' ? num('attention.sliding_window', 1024) : 0,
    qkNorm: true, ffnGate: true, activation: 'silu', bias: { q: false, k: false, v: false, output: false },
  };
  if (family === 'gemma3') cfg.activation = 'gelu';
  // A linear RoPE factor is a plain position divisor. llama3/yarn metadata encodes
  // frequency-band factors that cannot be expressed as one scale, so those models
  // keep unscaled positions: correct up to their original trained context.
  if (String(g.metadata.get(`${architecture}.rope.scaling.type`) ?? 'linear') === 'linear') cfg.ropeScale = 1 / num('rope.scaling.factor', 1);
  if (!Number.isSafeInteger(cfg.maxContext)) throw new Error('Invalid model context length.');
  if ([layers, hidden, heads, headDim, cfg.kvHeads, cfg.intermediate, cfg.rotaryDim].some(n => !Number.isInteger(n) || n < 1) || !Number.isInteger(cfg.period) || cfg.period < 0 || heads % cfg.kvHeads || headDim % 2 || cfg.rotaryDim % 2 || cfg.rotaryDim > headDim || !cfg.vocab || layers > 128 || heads > 256 || headDim > 512 || hidden > 16384 || cfg.intermediate > 131072 || cfg.vocab > 1_000_000) throw new Error('Unsupported model dimensions.');
  if (architecture === 'qwen35') {
    const flags = g.metadata.get('qwen35.attention.recurrent_layers');
    if (flags !== undefined && (!Array.isArray(flags) || flags.length !== layers + extra || flags.some(v => typeof v !== 'boolean'))) throw new Error('Invalid recurrent layer map.');
    cfg.recurrent = { keyHeads: num('ssm.group_count'), valueHeads: num('ssm.time_step_rank'), headDim: num('ssm.state_size'), convKernel: num('ssm.conv_kernel'), layers: Array.from({ length: layers }, (_, i) => Array.isArray(flags) ? flags[i] : (i + 1) % cfg.period !== 0) };
    const r = cfg.recurrent;
    if ([r.keyHeads, r.valueHeads, r.headDim, r.convKernel].some(n => !Number.isInteger(n)) || r.valueHeads % r.keyHeads || r.valueHeads > 128 || r.headDim > 256 || r.convKernel > 16 || num('ssm.inner_size') !== r.valueHeads * r.headDim) throw new Error('Unsupported recurrent dimensions.');
    // MTP weights are stored in this GGUF but do not participate in ordinary decoding.
    cfg.parameters = [...g.tensors.values()].filter(t => !t.name.startsWith('blk.') || Number(t.name.split('.')[1]) < layers).reduce((s, t) => s + t.dims.reduce((a, b) => a * b, 1), 0);
  }
  const shape = (name: string, dims: number[]) => {
    const t = g.tensors.get(name);
    if (!t || t.dims.join(',') !== dims.join(',')) throw new Error(`Missing or invalid tensor ${name}.`);
  };
  if (family === 'llama') {
    // Llama 3.x, Mistral, SmolLM and Qwen2/3 all share one dense decoder. Optional
    // tensors (Q/K normalization, attention bias, gated feed-forward) are detected
    // from the file instead of being assumed from the architecture name.
    if (g.tensors.has('blk.0.ffn_gate_inp.weight') || g.tensors.has('blk.0.ffn_gate_exps.weight')) throw new Error('Mixture-of-experts models are not supported.');
    cfg.ffnGate = g.tensors.has('blk.0.ffn_gate.weight');
    cfg.qkNorm = g.tensors.has('blk.0.attn_q_norm.weight');
    for (const key of ['q', 'k', 'v', 'output'] as const) cfg.bias[key] = g.tensors.has(`blk.0.attn_${key}.bias`);
    const attn = heads * headDim, kv = cfg.kvHeads * headDim;
    shape('token_embd.weight', [hidden, cfg.vocab]); shape('output_norm.weight', [hidden]);
    if (g.tensors.has('output.weight')) shape('output.weight', [hidden, cfg.vocab]);
    for (let l = 0; l < layers; l++) {
      const prefix = `blk.${l}.`;
      shape(prefix + 'attn_norm.weight', [hidden]); shape(prefix + 'ffn_norm.weight', [hidden]);
      shape(prefix + 'attn_q.weight', [hidden, attn]);
      shape(prefix + 'attn_k.weight', [hidden, kv]); shape(prefix + 'attn_v.weight', [hidden, kv]);
      shape(prefix + 'attn_output.weight', [attn, hidden]);
      if (cfg.bias.q) shape(prefix + 'attn_q.bias', [attn]);
      if (cfg.bias.k) shape(prefix + 'attn_k.bias', [kv]);
      if (cfg.bias.v) shape(prefix + 'attn_v.bias', [kv]);
      if (cfg.bias.output) shape(prefix + 'attn_output.bias', [hidden]);
      if (cfg.qkNorm) for (const key of ['attn_q_norm', 'attn_k_norm']) shape(prefix + key + '.weight', [headDim]);
      if (cfg.ffnGate) shape(prefix + 'ffn_gate.weight', [hidden, cfg.intermediate]);
      shape(prefix + 'ffn_up.weight', [hidden, cfg.intermediate]);
      shape(prefix + 'ffn_down.weight', [cfg.intermediate, hidden]);
    }
    return cfg;
  }
  shape('token_embd.weight', [hidden, cfg.vocab]); shape('output_norm.weight', [hidden]);
  if (g.tensors.has('output.weight')) shape('output.weight', [hidden, cfg.vocab]);
  for (let l = 0; l < layers; l++) {
    const prefix = `blk.${l}.`;
    for (const key of architecture === 'gemma3' ? ['attn_norm', 'post_attention_norm', 'ffn_norm', 'post_ffw_norm'] : ['attn_norm', 'post_attention_norm']) shape(prefix + key + '.weight', [hidden]);
    if (cfg.recurrent?.layers[l]) {
      const r = cfg.recurrent, value = r.valueHeads * r.headDim, channels = 2 * r.keyHeads * r.headDim + value;
      shape(prefix + 'attn_qkv.weight', [hidden, channels]); shape(prefix + 'attn_gate.weight', [hidden, value]);
      shape(prefix + 'ssm_conv1d.weight', [r.convKernel, channels]);
      for (const key of ['ssm_alpha', 'ssm_beta']) shape(prefix + key + '.weight', [hidden, r.valueHeads]);
      shape(prefix + 'ssm_a', [r.valueHeads]); shape(prefix + 'ssm_dt.bias', [r.valueHeads]);
      shape(prefix + 'ssm_norm.weight', [r.headDim]); shape(prefix + 'ssm_out.weight', [value, hidden]);
    } else {
      for (const key of ['attn_q_norm', 'attn_k_norm']) shape(prefix + key + '.weight', [headDim]);
      shape(prefix + 'attn_q.weight', [hidden, heads * headDim * (architecture === 'qwen35' ? 2 : 1)]);
      for (const key of ['attn_k', 'attn_v']) shape(prefix + key + '.weight', [hidden, cfg.kvHeads * headDim]);
      shape(prefix + 'attn_output.weight', [heads * headDim, hidden]);
    }
    for (const key of ['ffn_gate', 'ffn_up']) shape(prefix + key + '.weight', [hidden, cfg.intermediate]);
    shape(prefix + 'ffn_down.weight', [cfg.intermediate, hidden]);
  }
  return cfg;
}
export interface GGUFPeek { architecture: string; name: string; quantization: string }
const FILE_TYPES: Record<number, string> = {
  0: 'F32', 1: 'F16', 2: 'Q4_0', 3: 'Q4_1', 7: 'Q8_0', 8: 'Q5_0', 9: 'Q5_1',
  10: 'Q2_K', 11: 'Q3_K_S', 12: 'Q3_K_M', 13: 'Q3_K_L', 14: 'Q4_K_S', 15: 'Q4_K_M',
  16: 'Q5_K_S', 17: 'Q5_K_M', 18: 'Q6_K', 19: 'Q8_K',
};
/** Reads only the leading metadata entries of a GGUF, for listing a library without
 * loading any weights. Returns undefined when more header bytes are needed. */
export function peekGGUFHeader(bytes: Uint8Array): GGUFPeek | undefined {
  try {
    const r = new Reader(bytes);
    if (r.u32() !== 0x46554747) throw new Error('This file is not a GGUF model.');
    const version = r.u32();
    if (version !== 2 && version !== 3) throw new Error(`GGUF version ${version} is unsupported.`);
    r.u64(); const count = r.u64();
    if (count > 100_000) throw new Error('Invalid GGUF counts.');
    const found: Record<string, unknown> = {};
    for (let i = 0; i < Math.min(count, 512); i++) {
      const key = r.string();
      if (key === 'general.architecture' || key === 'general.name' || key === 'general.file_type') found[key] = r.value(r.u32());
      else { const type = r.u32(); if (type === 9) { r.u32(); const n = r.u64(); if (n > 2_000_000) throw new Error('Oversized GGUF array.'); for (let j = 0; j < n; j++) r.value(r.u32(), 1); } else r.value(type); }
      if (found['general.architecture'] !== undefined && found['general.name'] !== undefined && found['general.file_type'] !== undefined) break;
    }
    const architecture = String(found['general.architecture'] ?? '');
    if (!architecture) throw new Error('The GGUF file has no architecture metadata.');
    return { architecture, name: String(found['general.name'] ?? ''), quantization: FILE_TYPES[Number(found['general.file_type'])] ?? '' };
  } catch (e) { if (e instanceof NeedMore) return undefined; throw e; }
}
export async function peekGGUF(source: ByteSource, signal?: AbortSignal): Promise<GGUFPeek> {
  for (let n = Math.min(source.size, 128 * 1024); ; n = Math.min(source.size, n * 2)) {
    const peek = peekGGUFHeader(await source.read(0, n, signal));
    if (peek) return peek;
    if (n >= source.size || n >= 8 * 1024 ** 2) throw new Error('The GGUF header is truncated or invalid.');
  }
}
