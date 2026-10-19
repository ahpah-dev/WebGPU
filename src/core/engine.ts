import { Compute, MiB } from './gpu.ts';
import { makeSource, type ByteSource, type SourceInput } from './source.ts';
import { readGGUF, modelConfig, type GGUF, type ModelConfig, type Tensor } from './gguf.ts';
import { WeightPager, type PagingStats } from './pager.ts';
import { decodeWeight } from './quant.ts';
import { Tokenizer } from './tokenizer.ts';
import { QwenTokenizer } from './qwen-tokenizer.ts';
import { qwenKernels } from './qwen-shaders.ts';
import { matmulShader, normShader, ropeShader, attentionShader, addShader, geluShader } from './shaders.ts';

export interface LoadOptions { gpuBudgetMiB?: number; contextLength?: number; batchSize?: number }
export interface Message { role: 'user' | 'assistant'; content: string }
export interface GenerationOptions { maxTokens?: number; temperature?: number; topK?: number; topP?: number }
export interface Progress { phase: string; detail: string; fraction: number; layer?: number; stats?: PagingStats }
export interface ModelInfo { config: ModelConfig; fileBytes: number; adapter: string; contextLength: number; gpuBudgetMiB: number; stats: PagingStats }
export interface TokenEvent { token: number; text: string; delta: string; tokens: number; seconds: number; stats: PagingStats }

export class PagedEngine {
  private gpu!: Compute;
  private source!: ByteSource;
  private gguf!: GGUF;
  private config!: ModelConfig;
  private tokenizer!: Tokenizer | QwenTokenizer;
  private pager!: WeightPager;
  private work: Record<string, GPUBuffer> = {};
  private norms = new Map<string, GPUBuffer>();
  private kv: ({ k: GPUBuffer; v: GPUBuffer } | undefined)[] = [];
  private recurrent = new Map<number, { conv: GPUBuffer; state: GPUBuffer }>();
  private mm = new Map<number, GPUComputePipeline>();
  private kernels: Record<string, GPUComputePipeline> = {};
  private context = 512;
  private batch = 16;
  private budget = 2048;
  private busy = false;
  private disposed = false;
  private tokensInCache: number[] = [];
  private pendingPages = 0;
  private progress: (p: Progress) => void;
  constructor(progress: (p: Progress) => void = () => {}) { this.progress = progress; }
  async load(input: SourceInput, options: LoadOptions = {}, signal?: AbortSignal): Promise<ModelInfo> {
    this.context = options.contextLength ?? 512; this.batch = options.batchSize ?? 16; this.budget = options.gpuBudgetMiB ?? 2048;
    if (![256, 512, 1024, 2048].includes(this.context)) throw new Error('Context must be 256, 512, 1024 or 2048 tokens.');
    if (![1, 8, 16, 32].includes(this.batch)) throw new Error('Batch size must be 1, 8, 16 or 32.');
    if (!Number.isInteger(this.budget) || this.budget < 512 || this.budget > 16384) throw new Error('GPU budget must be 512–16384 MiB.');
    this.source = makeSource(input);
    this.progress({ phase: 'loading', detail: 'Reading GGUF metadata and tokenizer', fraction: 0.05 });
    this.gguf = await readGGUF(this.source, signal); this.config = modelConfig(this.gguf);
    this.tokenizer = this.config.architecture === 'qwen35' ? new QwenTokenizer(this.gguf.metadata) : new Tokenizer(this.gguf.metadata);
    const c = this.config, r = c.recurrent, recurrentLayers = r?.layers.filter(Boolean).length ?? 0;
    const kvBytes = (c.layers - recurrentLayers) * this.context * c.kvHeads * c.headDim * 8;
    const stateBytes = r ? recurrentLayers * (r.valueHeads * r.headDim ** 2 + (2 * r.keyHeads + r.valueHeads) * r.headDim * (r.convKernel - 1)) * 4 : 0;
    if (kvBytes + stateBytes + 96 * MiB > this.budget * MiB) throw new Error(`This context needs about ${Math.ceil((kvBytes + stateBytes) / MiB)} MiB for attention state. Reduce context length or increase the GPU budget.`);
    this.gpu = new Compute(this.budget);
    try {
      await this.gpu.init(); signal?.throwIfAborted();
      this.progress({ phase: 'loading', detail: 'Compiling quantized WebGPU kernels', fraction: 0.25 });
      const types = new Set([...this.gguf.tensors.values()].filter(t => t.dims.length === 2).map(t => t.type));
      for (const type of types) this.mm.set(type, await this.gpu.pipeline(matmulShader(type), `quantized matmul ${type}`));
      for (const [name, code] of Object.entries({ norm: normShader, rope: ropeShader, attention: attentionShader(this.context), add: addShader, gelu: geluShader })) this.kernels[name] = await this.gpu.pipeline(code, name);
      if (r) for (const [name, code] of Object.entries(qwenKernels)) this.kernels[name] = await this.gpu.pipeline(code, name);
      const sizes: Record<string, number> = { hidden: c.hidden, normed: c.hidden, q: c.heads * c.headDim, k: c.kvHeads * c.headDim, v: c.kvHeads * c.headDim, attention: c.heads * c.headDim, projection: c.hidden, gate: c.intermediate, up: c.intermediate, product: c.intermediate, down: c.hidden };
      if (r) Object.assign(sizes, { qg: 2 * c.heads * c.headDim, attnGate: c.heads * c.headDim, qkv: (2 * r.keyHeads + r.valueHeads) * r.headDim, convolved: (2 * r.keyHeads + r.valueHeads) * r.headDim, z: r.valueHeads * r.headDim, alpha: r.valueHeads, beta: r.valueHeads, control: 2 * r.valueHeads, recOutput: r.valueHeads * r.headDim, recNorm: r.valueHeads * r.headDim });
      for (const [name, n] of Object.entries(sizes)) this.work[name] = this.gpu.buffer(this.batch * n * 4, name);
      this.work.logits = this.gpu.buffer(c.vocab * 4, 'logits');
      for (let layer = 0; layer < c.layers; layer++) {
        if (r?.layers[layer]) this.recurrent.set(layer, { conv: this.gpu.buffer((2 * r.keyHeads + r.valueHeads) * r.headDim * (r.convKernel - 1) * 4, `convolution state ${layer}`), state: this.gpu.buffer(r.valueHeads * r.headDim ** 2 * 4, `recurrent state ${layer}`) });
        else this.kv[layer] = { k: this.gpu.buffer(this.context * c.kvHeads * c.headDim * 4, `K ${layer}`), v: this.gpu.buffer(this.context * c.kvHeads * c.headDim * 4, `V ${layer}`) };
      }
      const norms = [...this.gguf.tensors.values()].filter(t => (t.dims.length === 1 || t.name.endsWith('ssm_conv1d.weight')) && (!t.name.startsWith('blk.') || Number(t.name.split('.')[1]) < c.layers));
      for (let i = 0; i < norms.length; i++) {
        signal?.throwIfAborted(); const t = norms[i];
        const bytes = await this.source.read(t.offset, t.bytes, signal), values = new Float32Array(t.dims.reduce((a, b) => a * b, 1));
        for (let j = 0; j < values.length; j++) values[j] = decodeWeight(bytes, t.type, j);
        const b = this.gpu.buffer(values.byteLength, t.name); this.gpu.write(b, values); this.norms.set(t.name, b);
        if (i % 20 === 0) this.progress({ phase: 'loading', detail: 'Loading normalization vectors', fraction: 0.5 + 0.45 * i / norms.length });
      }
      this.pager = new WeightPager(this.gpu, this.source); await this.gpu.done();
      this.progress({ phase: 'ready', detail: 'Ready. Model weights will page in as you generate.', fraction: 1, stats: this.pager.snapshot() });
      return this.info();
    } catch (e) { this.dispose(); throw e; }
  }
  info(): ModelInfo { return { config: this.config, fileBytes: this.source.size, adapter: this.gpu.adapterName, contextLength: this.context, gpuBudgetMiB: this.budget, stats: this.pager.snapshot() }; }
  private tensor(name: string): Tensor { const t = this.gguf.tensors.get(name); if (!t) throw new Error(`Missing ${name}.`); return t; }
  private norm(src: GPUBuffer, dst: GPUBuffer, name: string, batch: number) {
    this.gpu.run(this.kernels.norm, [src, this.norms.get(name)!, dst], [this.config.hidden, { f: this.config.epsilon }], batch);
  }
  private async linear(name: string, src: GPUBuffer, dst: GPUBuffer, batch: number, signal?: AbortSignal, inputOffset = 0) {
    const t = this.tensor(name), [columns, rows] = t.dims, pageRows = this.pager.rowsPerPage(t), pipeline = this.mm.get(t.type)!;
    for (let row = 0; row < rows; row += pageRows) {
      signal?.throwIfAborted(); const n = Math.min(pageRows, rows - row);
      const weights = await this.pager.page(t, row, n, signal);
      this.gpu.run(pipeline, [weights, src, dst], [columns, n, rows, row, inputOffset, 0, 0, 0], n, batch);
      // writeBuffer and submit share an ordered GPU queue. A subsequent upload cannot
      // overwrite the slot before the previous dispatch consumes it. Fence every eight
      // pages to bound staging/queued work instead of stalling for every matrix page.
      if (++this.pendingPages >= 8) { await this.gpu.done(); this.pendingPages = 0; }
    }
  }
  private async forward(ids: number[], start: number, logits: boolean, signal?: AbortSignal): Promise<Float32Array | undefined> {
    const c = this.config, w = this.work, n = ids.length;
    const embedding = this.tensor('token_embd.weight'), input = new Float32Array(n * c.hidden);
    for (let b = 0; b < n; b++) {
      const row = await this.source.read(embedding.offset + ids[b] * embedding.rowBytes, embedding.rowBytes, signal);
      for (let i = 0; i < c.hidden; i++) input[b * c.hidden + i] = decodeWeight(row, embedding.type, i) * (c.architecture === 'gemma3' ? Math.sqrt(c.hidden) : 1);
    }
    this.gpu.write(w.hidden, input);
    for (let l = 0; l < c.layers; l++) {
      signal?.throwIfAborted(); const prefix = `blk.${l}.`, local = l % c.period !== c.period - 1;
      this.progress({ phase: start + n > 1 && n > 1 ? 'prefill' : 'generating', detail: `Layer ${l + 1} / ${c.layers} · token ${start + 1}${n > 1 ? `–${start + n}` : ''}`, fraction: l / c.layers, layer: l, stats: this.pager.snapshot() });
      this.norm(w.hidden, w.normed, prefix + 'attn_norm.weight', n);
      if (c.architecture === 'qwen35') {
        await this.qwenLayer(l, n, start, signal);
        await this.gpu.done(); this.pendingPages = 0; continue;
      }
      await this.linear(prefix + 'attn_q.weight', w.normed, w.q, n, signal);
      await this.linear(prefix + 'attn_k.weight', w.normed, w.k, n, signal);
      await this.linear(prefix + 'attn_v.weight', w.normed, w.v, n, signal);
      for (const [name, heads] of [['q', c.heads], ['k', c.kvHeads]] as const) this.gpu.run(this.kernels.rope, [w[name], this.norms.get(prefix + `attn_${name}_norm.weight`)!], [heads, c.headDim, start, { f: local ? c.localTheta : c.theta }, { f: local ? 1 : c.ropeScale }, { f: c.epsilon }, 0, 0], heads, n);
      const kvStride = c.kvHeads * c.headDim * 4;
      this.gpu.copy(w.k, this.kv[l]!.k, n * kvStride, 0, start * kvStride); this.gpu.copy(w.v, this.kv[l]!.v, n * kvStride, 0, start * kvStride);
      this.gpu.run(this.kernels.attention, [w.q, this.kv[l]!.k, this.kv[l]!.v, w.attention], [c.heads, c.kvHeads, c.headDim, start, { f: c.attentionScale }, local ? c.window : this.context, 0, 0], c.heads, n);
      await this.linear(prefix + 'attn_output.weight', w.attention, w.projection, n, signal);
      this.norm(w.projection, w.normed, prefix + 'post_attention_norm.weight', n);
      this.gpu.run(this.kernels.add, [w.normed, w.hidden], [n * c.hidden], Math.ceil(n * c.hidden / 256));
      this.norm(w.hidden, w.normed, prefix + 'ffn_norm.weight', n);
      await this.linear(prefix + 'ffn_gate.weight', w.normed, w.gate, n, signal);
      await this.linear(prefix + 'ffn_up.weight', w.normed, w.up, n, signal);
      this.gpu.run(this.kernels.gelu, [w.gate, w.up, w.product], [n * c.intermediate], Math.ceil(n * c.intermediate / 256));
      await this.linear(prefix + 'ffn_down.weight', w.product, w.down, n, signal);
      this.norm(w.down, w.normed, prefix + 'post_ffw_norm.weight', n);
      this.gpu.run(this.kernels.add, [w.normed, w.hidden], [n * c.hidden], Math.ceil(n * c.hidden / 256));
      await this.gpu.done(); this.pendingPages = 0;
    }
    if (!logits) { await this.gpu.done(); return; }
    this.norm(w.hidden, w.normed, 'output_norm.weight', n);
    await this.linear(this.gguf.tensors.has('output.weight') ? 'output.weight' : 'token_embd.weight', w.normed, w.logits, 1, signal, n - 1);
    return new Float32Array(await this.gpu.read(w.logits, c.vocab * 4));
  }
  private async qwenLayer(layer: number, n: number, start: number, signal?: AbortSignal) {
    const c = this.config, r = c.recurrent!, w = this.work, prefix = `blk.${layer}.`;
    if (r.layers[layer]) {
      const state = this.recurrent.get(layer)!, channels = (2 * r.keyHeads + r.valueHeads) * r.headDim;
      await this.linear(prefix + 'attn_qkv.weight', w.normed, w.qkv, n, signal);
      await this.linear(prefix + 'attn_gate.weight', w.normed, w.z, n, signal);
      await this.linear(prefix + 'ssm_alpha.weight', w.normed, w.alpha, n, signal);
      await this.linear(prefix + 'ssm_beta.weight', w.normed, w.beta, n, signal);
      this.gpu.run(this.kernels.conv, [w.qkv, this.norms.get(prefix + 'ssm_conv1d.weight')!, state.conv, w.convolved], [channels, r.convKernel, n], Math.ceil(channels / 256));
      this.gpu.run(this.kernels.l2, [w.convolved], [channels, r.headDim, { f: 1e-6 }], 2 * r.keyHeads, n);
      this.gpu.run(this.kernels.decay, [w.alpha, w.beta, this.norms.get(prefix + 'ssm_a')!, this.norms.get(prefix + 'ssm_dt.bias')!, w.control], [n * r.valueHeads, r.valueHeads], Math.ceil(n * r.valueHeads / 256));
      this.gpu.run(this.kernels.delta, [w.convolved, w.control, state.state, w.recOutput], [r.headDim, r.keyHeads, r.valueHeads, n], r.valueHeads * r.headDim);
      this.gpu.run(this.kernels.normGate, [w.recOutput, this.norms.get(prefix + 'ssm_norm.weight')!, w.z, w.recNorm], [r.headDim, { f: c.epsilon }], n * r.valueHeads);
      await this.linear(prefix + 'ssm_out.weight', w.recNorm, w.projection, n, signal);
    } else {
      await this.linear(prefix + 'attn_q.weight', w.normed, w.qg, n, signal);
      this.gpu.run(this.kernels.splitQuery, [w.qg, w.q, w.attnGate], [n * c.heads * c.headDim, c.headDim], Math.ceil(n * c.heads * c.headDim / 256));
      await this.linear(prefix + 'attn_k.weight', w.normed, w.k, n, signal);
      await this.linear(prefix + 'attn_v.weight', w.normed, w.v, n, signal);
      for (const [name, heads] of [['q', c.heads], ['k', c.kvHeads]] as const) this.gpu.run(this.kernels.rope, [w[name], this.norms.get(prefix + `attn_${name}_norm.weight`)!], [heads, c.headDim, start, { f: c.theta }, { f: c.ropeScale }, { f: c.epsilon }, c.rotaryDim, 0], heads, n);
      const kv = this.kv[layer]!, stride = c.kvHeads * c.headDim * 4;
      this.gpu.copy(w.k, kv.k, n * stride, 0, start * stride); this.gpu.copy(w.v, kv.v, n * stride, 0, start * stride);
      this.gpu.run(this.kernels.attention, [w.q, kv.k, kv.v, w.attention], [c.heads, c.kvHeads, c.headDim, start, { f: c.attentionScale }, this.context, 0, 0], c.heads, n);
      this.gpu.run(this.kernels.sigmoidGate, [w.attnGate, w.attention], [n * c.heads * c.headDim], Math.ceil(n * c.heads * c.headDim / 256));
      await this.linear(prefix + 'attn_output.weight', w.attention, w.projection, n, signal);
    }
    this.gpu.run(this.kernels.add, [w.projection, w.hidden], [n * c.hidden], Math.ceil(n * c.hidden / 256));
    this.norm(w.hidden, w.normed, prefix + 'post_attention_norm.weight', n);
    await this.linear(prefix + 'ffn_gate.weight', w.normed, w.gate, n, signal);
    await this.linear(prefix + 'ffn_up.weight', w.normed, w.up, n, signal);
    this.gpu.run(this.kernels.silu, [w.gate, w.up, w.product], [n * c.intermediate], Math.ceil(n * c.intermediate / 256));
    await this.linear(prefix + 'ffn_down.weight', w.product, w.down, n, signal);
    this.gpu.run(this.kernels.add, [w.down, w.hidden], [n * c.hidden], Math.ceil(n * c.hidden / 256));
  }
  async *generate(messages: Message[], options: GenerationOptions = {}, signal?: AbortSignal): AsyncGenerator<TokenEvent> {
    if (this.busy) throw new Error('A generation is already running.');
    if (this.disposed || !this.pager) throw new Error('Load a model first.');
    if (!Array.isArray(messages) || !messages.length || messages.some(m => !['user', 'assistant'].includes(m.role) || typeof m.content !== 'string' || m.content.length > 16000)) throw new Error('Provide user/assistant text messages (at most 16,000 characters each).');
    const maxTokens = options.maxTokens ?? 128;
    if (!Number.isInteger(maxTokens) || maxTokens < 1 || maxTokens > 2048) throw new Error('maxTokens must be 1–2048.');
    const temperature = options.temperature ?? 0.7, topK = options.topK ?? 40, topP = options.topP ?? 0.9;
    if (!Number.isFinite(temperature) || temperature < 0 || temperature > 2 || !Number.isInteger(topK) || topK < 1 || topK > 200 || !Number.isFinite(topP) || topP <= 0 || topP > 1) throw new Error('Invalid sampling settings.');
    this.busy = true;
    try {
      const prompt = this.tokenizer.chat(messages);
      if (prompt.length >= this.context) throw new Error(`The prompt has ${prompt.length} tokens; the context limit is ${this.context}. Start a new chat or reload with a longer context.`);
      let shared = 0;
      while (shared < prompt.length && shared < this.tokensInCache.length && prompt[shared] === this.tokensInCache[shared]) shared++;
      // Last prompt token must be evaluated to obtain logits even for an exact-prefix retry.
      shared = Math.min(shared, prompt.length - 1);
      // A recurrent state cannot be rewound like a KV cache. Rebuild on every new
      // request so edits, retries, cancellation and multi-turn chat are deterministic.
      if (this.config.recurrent) {
        shared = 0;
        this.gpu.clear([...this.recurrent.values()].flatMap(s => [s.conv, s.state]));
      }
      let values: Float32Array | undefined;
      const started = performance.now();
      this.tokensInCache = this.tokensInCache.slice(0, shared);
      for (let i = shared; i < prompt.length; i += this.batch) {
        const batch = prompt.slice(i, i + this.batch);
        values = await this.forward(batch, i, i + batch.length === prompt.length, signal);
        this.tokensInCache.push(...batch);
      }
      const generated: number[] = []; let text = '';
      for (let i = 0; i < Math.min(maxTokens, this.context - prompt.length); i++) {
        signal?.throwIfAborted(); const token = sample(values!, temperature, topK, topP);
        if (this.tokenizer.eos.has(token)) break;
        generated.push(token); const full = this.tokenizer.decode(generated), delta = full.slice(text.length); text = full;
        yield { token, text, delta, tokens: generated.length, seconds: (performance.now() - started) / 1000, stats: this.pager.snapshot() };
        if (i + 1 < Math.min(maxTokens, this.context - prompt.length)) {
          values = await this.forward([token], this.tokensInCache.length, true, signal); this.tokensInCache.push(token);
        }
      }
    } catch (e) { this.tokensInCache = []; throw e; }
    finally { this.busy = false; }
  }
  reset() { if (this.busy) throw new Error('Stop generation first.'); this.tokensInCache = []; }
  dispose() { this.disposed = true; this.gpu?.dispose(); this.tokensInCache = []; }
}
export function sample(logits: Float32Array, temperature: number, topK: number, topP: number): number {
  let best = -Infinity, bestId = -1;
  for (let i = 0; i < logits.length; i++) if (Number.isFinite(logits[i]) && logits[i] > best) { best = logits[i]; bestId = i; }
  if (bestId < 0) throw new Error('Inference produced non-finite logits. Stop and report this model/runtime combination.');
  if (temperature === 0) return bestId;
  const candidates: { id: number; value: number; probability: number }[] = [];
  for (let i = 0; i < logits.length; i++) {
    if (!Number.isFinite(logits[i])) continue;
    if (candidates.length < topK || logits[i] > candidates[candidates.length - 1].value) {
      candidates.push({ id: i, value: logits[i], probability: Math.exp((logits[i] - best) / temperature) });
      candidates.sort((a, b) => b.value - a.value); if (candidates.length > topK) candidates.pop();
    }
  }
  const total = candidates.reduce((s, c) => s + c.probability, 0);
  let retained = 0, cutoff = 0;
  for (; cutoff < candidates.length; cutoff++) { retained += candidates[cutoff].probability; if (retained / total >= topP) { cutoff++; break; } }
  let draw = Math.random() * retained;
  for (const c of candidates.slice(0, cutoff)) { draw -= c.probability; if (draw <= 0) return c.id; }
  return bestId;
}
