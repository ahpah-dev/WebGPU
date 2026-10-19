// Numerical check of the dense decoder on real WebGPU hardware.
//
// Builds tiny synthetic GGUF files (llama / qwen2 / qwen3) with known float32
// weights, computes the expected greedy tokens with an independent float64 CPU
// reference, then runs the browser runtime in headless Chrome and compares the
// generated token ids. Requires a WebGPU-capable Chrome and a built dist/.
//
//   npm run build && npm run check:dense
//
// Set CHROME_PATH to override browser discovery. Without a browser the script
// reports the CPU reference and skips the GPU comparison.
import { writeFileSync, mkdirSync, existsSync } from 'node:fs';
import { spawn } from 'node:child_process';
import { setTimeout as delay } from 'node:timers/promises';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';

const CHROME = [process.env.CHROME_PATH, '/usr/bin/google-chrome', '/usr/bin/chromium', '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
  'C:/Program Files/Google/Chrome/Application/chrome.exe', 'C:/Program Files (x86)/Google/Chrome/Application/chrome.exe', 'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe']
  .filter(Boolean).find(path => existsSync(path));
const WORK = join(process.env.TEMP || process.env.TMPDIR || '.', 'webgpu-dense-check');
mkdirSync(WORK, { recursive: true });

// ---------------------------------------------------------------- tiny GGUF
function rng(seed) { let state = seed >>> 0; return () => { state = (Math.imul(state, 1664525) + 1013904223) >>> 0; return state / 4294967296; }; }
class Writer {
  parts = [];
  u8(v) { this.parts.push(v & 255); return this; }
  u32(v) { const b = new Uint8Array(4); new DataView(b.buffer).setUint32(0, v >>> 0, true); this.parts.push(...b); return this; }
  i32(v) { const b = new Uint8Array(4); new DataView(b.buffer).setInt32(0, v, true); this.parts.push(...b); return this; }
  u64(v) { const b = new Uint8Array(8); new DataView(b.buffer).setBigUint64(0, BigInt(v), true); this.parts.push(...b); return this; }
  f32(v) { const b = new Uint8Array(4); new DataView(b.buffer).setFloat32(0, v, true); this.parts.push(...b); return this; }
  str(s) { const b = new TextEncoder().encode(s); this.u64(b.length); this.parts.push(...b); return this; }
  bytes() { return Uint8Array.from(this.parts); }
}
function entry(w, key, type, value) {
  w.str(key).u32(type);
  if (type === 8) w.str(value);
  else if (type === 4) w.u32(value);
  else if (type === 6) w.f32(value);
  else if (type === 9) throw new Error('use arrayEntry');
  else throw new Error('bad type');
}
function arrayEntry(w, key, elementType, values) {
  w.str(key).u32(9).u32(elementType).u64(values.length);
  for (const value of values) {
    if (elementType === 8) w.str(value);
    else if (elementType === 5) w.i32(value);
    else throw new Error('bad element type');
  }
}
function align(n, a = 32) { return Math.ceil(n / a) * a; }

const BYTE_ALPHABET = (() => {
  const kept = Array.from({ length: 256 }, (_, b) => b).filter(b => b >= 33 && b <= 126 || b >= 161 && b <= 172 || b >= 174);
  const codes = new Set(kept); let extra = 256;
  return Array.from({ length: 256 }, (_, b) => String.fromCodePoint(codes.has(b) ? b : extra++));
})();

function buildVariant(kind) {
  const hidden = 64, heads = 4, kvHeads = 2, headDim = 16, layers = 2, intermediate = 128;
  const SPECIALS = ['<|im_start|>', '<|im_end|>', '<|eot_id|>'];
  const tokens = [...BYTE_ALPHABET, ...SPECIALS];
  const vocab = tokens.length;
  const bias = kind === 'qwen2', qkNorm = kind === 'qwen3';
  const random = rng(kind === 'llama' ? 7 : kind === 'qwen2' ? 11 : 13);
  const weights = new Map();
  const make = (name, dims, scale) => {
    const n = dims.reduce((a, b) => a * b, 1), values = new Float32Array(n);
    for (let i = 0; i < n; i++) values[i] = (random() * 2 - 1) * scale;
    weights.set(name, { dims, values });
    return name;
  };
  const eins = (name, dims) => { const n = dims.reduce((a, b) => a * b, 1), values = new Float32Array(n); for (let i = 0; i < n; i++) values[i] = 1 + (random() * 2 - 1) * 0.1; weights.set(name, { dims, values }); };
  make('token_embd.weight', [hidden, vocab], 0.6);
  eins('output_norm.weight', [hidden]);
  make('output.weight', [hidden, vocab], 0.35);
  for (let l = 0; l < layers; l++) {
    const p = `blk.${l}.`;
    eins(p + 'attn_norm.weight', [hidden]);
    make(p + 'attn_q.weight', [hidden, heads * headDim], 0.5);
    make(p + 'attn_k.weight', [hidden, kvHeads * headDim], 0.5);
    make(p + 'attn_v.weight', [hidden, kvHeads * headDim], 0.5);
    make(p + 'attn_output.weight', [heads * headDim, hidden], 0.5);
    if (bias) {
      make(p + 'attn_q.bias', [heads * headDim], 0.1);
      make(p + 'attn_k.bias', [kvHeads * headDim], 0.1);
      make(p + 'attn_v.bias', [kvHeads * headDim], 0.1);
    }
    if (qkNorm) { eins(p + 'attn_q_norm.weight', [headDim]); eins(p + 'attn_k_norm.weight', [headDim]); }
    eins(p + 'ffn_norm.weight', [hidden]);
    make(p + 'ffn_gate.weight', [hidden, intermediate], 0.45);
    make(p + 'ffn_up.weight', [hidden, intermediate], 0.45);
    make(p + 'ffn_down.weight', [intermediate, hidden], 0.45);
  }
  const types = tokens.map((_, i) => i < 256 ? 1 : 3);
  const eos = tokens.indexOf(SPECIALS[1]);
  const architecture = kind === 'llama' ? 'llama' : kind;
  const metadata = [
    ['general.architecture', 8, architecture],
    ['general.name', 8, `tiny-${kind}`],
    ['general.file_type', 4, 0],
    [`${architecture}.block_count`, 4, layers],
    [`${architecture}.context_length`, 4, 2048],
    [`${architecture}.embedding_length`, 4, hidden],
    [`${architecture}.feed_forward_length`, 4, intermediate],
    [`${architecture}.attention.head_count`, 4, heads],
    [`${architecture}.attention.head_count_kv`, 4, kvHeads],
    [`${architecture}.attention.key_length`, 4, headDim],
    [`${architecture}.attention.layer_norm_rms_epsilon`, 6, 1e-6],
    [`${architecture}.rope.dimension_count`, 4, headDim],
    [`${architecture}.rope.freq_base`, 6, 10000],
    ['tokenizer.ggml.model', 8, 'gpt2'],
    ['tokenizer.ggml.pre', 8, 'llama-bpe'],
    ['tokenizer.ggml.eos_token_id', 4, eos],
    ['tokenizer.chat_template', 8, '<|im_start|>{{ role }}\n{{ content }}<|im_end|>\n'],
  ];
  const w = new Writer().u32(0x46554747).u32(3).u64(weights.size).u64(metadata.length + 3);
  for (const [key, type, value] of metadata) entry(w, key, type, value);
  arrayEntry(w, 'tokenizer.ggml.tokens', 8, tokens);
  arrayEntry(w, 'tokenizer.ggml.token_type', 5, types);
  arrayEntry(w, 'tokenizer.ggml.merges', 8, []);
  const infos = [...weights.entries()].map(([name, { dims }]) => ({ name, dims }));
  let offset = 0;
  const placed = infos.map(info => { offset = align(offset); const entry = { ...info, offset }; const n = info.dims.reduce((a, b) => a * b, 1); offset += n * 4; return entry; });
  for (const info of placed) { w.str(info.name).u32(info.dims.length); for (const d of info.dims) w.u64(d); w.u32(0).u64(info.offset); }
  const header = w.bytes(), dataStart = align(header.length);
  const file = new Uint8Array(dataStart + align(offset));
  file.set(header, 0);
  for (const info of placed) {
    const { values } = weights.get(info.name), bytes = new Uint8Array(values.buffer.slice(0));
    file.set(bytes, dataStart + info.offset);
  }
  return { file, weights, config: { hidden, heads, kvHeads, headDim, layers, intermediate, vocab, bias, qkNorm, rope: kind === 'llama' ? 'norm' : 'neox', eps: 1e-6, theta: 10000, eos } };
}

// ------------------------------------------------------------ CPU reference
const W = (model, name) => model.weights.get(name).values;
function rms(x, weight, eps) {
  let sum = 0; for (const v of x) sum += v * v;
  const inv = 1 / Math.sqrt(sum / x.length + eps);
  return x.map((v, i) => v * inv * weight[i]);
}
function project(x, weight, outSize, biasValue) {
  const width = x.length, out = new Float64Array(outSize);
  for (let r = 0; r < outSize; r++) { let sum = 0; for (let c = 0; c < width; c++) sum += x[c] * weight[r * width + c]; out[r] = sum + (biasValue ? biasValue[r] : 0); }
  return out;
}
function rotate(x, offset, pos, c) {
  const { headDim, rope, theta } = c, rot = headDim;
  for (let j = 0; j < rot / 2; j++) {
    const angle = pos / Math.pow(theta, (2 * j) / rot);
    const cos = Math.cos(angle), sin = Math.sin(angle);
    const a = offset + (rope === 'neox' ? j : 2 * j), b = offset + (rope === 'neox' ? j + rot / 2 : 2 * j + 1);
    const left = x[a], right = x[b];
    x[a] = left * cos - right * sin;
    x[b] = left * sin + right * cos;
  }
}
function rmsHead(vector, weight, eps) {
  let sum = 0; for (const v of vector) sum += v * v;
  const inv = 1 / Math.sqrt(sum / vector.length + eps);
  return vector.map((v, i) => v * inv * weight[i]);
}
function forward(model, ids) {
  const c = model.config, { hidden, heads, kvHeads, headDim, layers } = c, group = heads / kvHeads;
  const keys = Array.from({ length: layers }, () => []), values = Array.from({ length: layers }, () => []);
  let logits;
  for (let pos = 0; pos < ids.length; pos++) {
    const embedding = W(model, 'token_embd.weight');
    let x = Array.from({ length: hidden }, (_, i) => embedding[ids[pos] * hidden + i]);
    for (let l = 0; l < layers; l++) {
      const p = `blk.${l}.`;
      const normed = rms(x, W(model, p + 'attn_norm.weight'), c.eps);
      let q = project(normed, W(model, p + 'attn_q.weight'), heads * headDim, c.bias ? W(model, p + 'attn_q.bias') : undefined);
      let k = project(normed, W(model, p + 'attn_k.weight'), kvHeads * headDim, c.bias ? W(model, p + 'attn_k.bias') : undefined);
      const v = project(normed, W(model, p + 'attn_v.weight'), kvHeads * headDim, c.bias ? W(model, p + 'attn_v.bias') : undefined);
      if (c.qkNorm) {
        const qn = W(model, p + 'attn_q_norm.weight'), kn = W(model, p + 'attn_k_norm.weight');
        q = Array.from({ length: heads }, (_, h) => rmsHead(Array.from(q.slice(h * headDim, (h + 1) * headDim)), qn, c.eps)).flat();
        k = Array.from({ length: kvHeads }, (_, h) => rmsHead(Array.from(k.slice(h * headDim, (h + 1) * headDim)), kn, c.eps)).flat();
      }
      for (let h = 0; h < heads; h++) rotate(q, h * headDim, pos, c);
      for (let h = 0; h < kvHeads; h++) rotate(k, h * headDim, pos, c);
      keys[l].push(Float64Array.from(k)); values[l].push(Float64Array.from(v));
      const attention = new Float64Array(heads * headDim);
      for (let h = 0; h < heads; h++) {
        const kv = keys[l], vv = values[l], scores = [];
        let max = -Infinity;
        for (let t = 0; t <= pos; t++) {
          let dot = 0; for (let d = 0; d < headDim; d++) dot += q[h * headDim + d] * kv[t][Math.floor(h / group) * headDim + d];
          const score = dot / Math.sqrt(headDim); scores.push(score); if (score > max) max = score;
        }
        let total = 0; for (let t = 0; t < scores.length; t++) { scores[t] = Math.exp(scores[t] - max); total += scores[t]; }
        for (let d = 0; d < headDim; d++) { let sum = 0; for (let t = 0; t < scores.length; t++) sum += (scores[t] / total) * vv[t][Math.floor(h / group) * headDim + d]; attention[h * headDim + d] = sum; }
      }
      const projected = project(attention, W(model, p + 'attn_output.weight'), hidden, undefined);
      x = x.map((value, i) => value + projected[i]);
      const ffnNormed = rms(x, W(model, p + 'ffn_norm.weight'), c.eps);
      const gate = project(ffnNormed, W(model, p + 'ffn_gate.weight'), c.intermediate, undefined);
      const up = project(ffnNormed, W(model, p + 'ffn_up.weight'), c.intermediate, undefined);
      const product = Float64Array.from(gate, (g, i) => (g / (1 + Math.exp(-g))) * up[i]);
      const down = project(product, W(model, p + 'ffn_down.weight'), hidden, undefined);
      x = x.map((value, i) => value + down[i]);
    }
    const final = rms(x, W(model, 'output_norm.weight'), c.eps);
    logits = project(final, W(model, 'output.weight'), c.vocab, undefined);
  }
  const sorted = [...logits].map((value, id) => ({ id, value })).sort((a, b) => b.value - a.value);
  return { token: sorted[0].id, margin: sorted[0].value - sorted[1].value };
}
function chat(idList) {
  const tokens = [...BYTE_ALPHABET, '<|im_start|>', '<|im_end|>', '<|eot_id|>'];
  const start = tokens.indexOf('<|im_start|>'), end = tokens.indexOf('<|im_end|>');
  const bytes = text => [...new TextEncoder().encode(text)];
  const ids = [];
  for (const message of idList) { ids.push(start, ...bytes(`${message.role}\n`), ...bytes(message.content.trim()), end, ...bytes('\n')); }
  ids.push(start, ...bytes('assistant\n'));
  return ids;
}

// ------------------------------------------------------------ Chrome driver
class Chrome {
  constructor(port) { this.port = port; this.id = 0; }
  async start() {
    const args = ['--headless=new', `--remote-debugging-port=${this.port}`, `--user-data-dir=${join(WORK, 'profile')}`,
      '--enable-unsafe-webgpu', '--ignore-gpu-blocklist', '--no-first-run', '--no-default-browser-check', 'about:blank'];
    this.child = spawn(CHROME, args, { stdio: 'ignore' });
    for (let i = 0; i < 60; i++) {
      try { const response = await fetch(`http://127.0.0.1:${this.port}/json/list`); if (response.ok) { this.targets = await response.json(); break; } } catch { /* wait */ }
      await delay(250);
    }
    if (!this.targets) throw new Error('Chrome debugging endpoint did not start');
    this.ws = new WebSocket(this.targets.find(t => t.type === 'page').webSocketDebuggerUrl);
    await new Promise((resolve, reject) => { this.ws.onopen = resolve; this.ws.onerror = reject; });
    this.ws.addEventListener('message', event => {
      const data = JSON.parse(event.data);
      const pending = this.pending?.get(data.id); if (!pending) return;
      this.pending.delete(data.id); data.error ? pending.reject(new Error(JSON.stringify(data.error))) : pending.resolve(data.result);
    });
    this.pending = new Map();
    await this.call('Page.enable');
  }
  call(method, params = {}) {
    return new Promise((resolve, reject) => { const id = ++this.id; this.pending.set(id, { resolve, reject }); this.ws.send(JSON.stringify({ id, method, params })); });
  }
  async open(url) {
    await this.call('Page.navigate', { url });
    for (let i = 0; i < 80; i++) {
      const state = await this.call('Runtime.evaluate', { expression: 'document.readyState', returnByValue: true });
      if (state.result?.value === 'complete') return;
      await delay(100);
    }
  }
  async evaluate(expression) {
    const result = await this.call('Runtime.evaluate', { expression, awaitPromise: true, returnByValue: true });
    if (result.exceptionDetails) throw new Error(result.exceptionDetails.exception?.description || result.exceptionDetails.text);
    return result.result?.value;
  }
  stop() { try { this.ws?.close(); } catch { /* ignore */ } this.child?.kill(); }
}
// ------------------------------------------------------------------- driver
const root = fileURLToPath(new URL('..', import.meta.url));
const variants = ['llama', 'qwen2', 'qwen3'];
const chrome = CHROME ? new Chrome(9460) : undefined;
let failures = 0;
if (!existsSync(join(root, 'dist', 'sdk', 'sdk.js'))) {
  console.error('Build the app first: npm run build');
  process.exit(1);
}
try {
  if (chrome) await chrome.start();
  for (const [index, kind] of variants.entries()) {
    const model = buildVariant(kind);
    const file = join(WORK, `tiny-${kind}.gguf`);
    writeFileSync(file, model.file);
    const prompt = 'hello';
    const expectedIds = chat([{ role: 'user', content: prompt }]);
    const expected = [];
    const ids = [...expectedIds];
    for (let i = 0; i < 4; i++) {
      const { token, margin } = forward(model, ids);
      expected.push(token); ids.push(token);
      if (token === model.config.eos) break;
      process.stdout.write(`  ${kind} step ${i + 1}: CPU token ${token} (top-2 margin ${margin.toFixed(4)})\n`);
    }
    console.log(`${kind}: expected [${expected.join(',')}] from ${expectedIds.length}-token prompt, ${model.file.length}-byte GGUF`);
    if (!chrome) { console.log(`${kind}: SKIP (no Chrome found; set CHROME_PATH)`); continue; }
    const port = 8790 + index;
    const server = spawn(process.execPath, ['server/serve.mjs', '--model', file, '--port', String(port)], { cwd: root, stdio: 'ignore' });
    try {
      await delay(600);
      await chrome.open(`http://127.0.0.1:${port}/`);
      const actual = await chrome.evaluate(`(async () => {
        const sdk = await import('/sdk/sdk.js');
        const info = await (await fetch('/model-info')).json();
        const model = new sdk.WebGPU();
        const loaded = await model.load({ url: new URL('/model.gguf', location.origin).href, size: info.size, name: info.name }, { contextLength: 512, performance: 'lite', gpuBudgetMiB: 512 });
        const tokens = [];
        for await (const event of model.generate([{ role: 'user', content: ${JSON.stringify(prompt)} }], { maxTokens: 4, temperature: 0 })) tokens.push(event.token);
        model.dispose();
        return { tokens, adapter: loaded.adapter, batchSize: loaded.batchSize };
      })()`);
      const pass = Array.isArray(actual?.tokens) && actual.tokens.join(',') === expected.join(',');
      console.log(`${kind}: browser [${actual?.tokens}] ${pass ? 'PASS' : 'FAIL'} (${actual?.adapter}, batch ${actual?.batchSize})`);
      if (!pass) failures++;
    } finally { server.kill(); }
  }
} finally { chrome?.stop(); }
if (!chrome) console.log('GPU comparison skipped; CPU reference tokens printed above.');
process.exitCode = failures ? 1 : 0;
