import { Compute } from './core/gpu.ts';
import { matmulShader, attentionShader, normShader, ropeShader } from './core/shaders.ts';
import { decodeWeight } from './core/quant.ts';
import { QUANTS } from './core/gguf.ts';
import { WebGPU } from './sdk.ts';
const result = document.getElementById('results')!;
const log = (s: string) => { result.textContent += s + '\n'; };
let model: WebGPU | undefined;
function near(actual: number, expected: number, label: string) { if (!Number.isFinite(actual) || Math.abs(actual - expected) > .001 * Math.max(1, Math.abs(expected))) throw new Error(`${label}: got ${actual}, expected ${expected}`); }
document.getElementById('math')!.onclick = async () => {
  result.textContent = 'Running…\n'; const gpu = new Compute(512);
  try {
    await gpu.init(); log(`Adapter: ${gpu.adapterName}`);
    for (const type of [0, 1, 2, 8, 12, 13, 14]) {
      const rows = 5, cols = 256, batches = 3, layout = QUANTS[type];
      const packed = new Uint8Array(Math.ceil(rows * cols / layout.block * layout.bytes / 4) * 4);
      for (let i = 0; i < packed.length; i++) packed[i] = (i * 17 + 93) % 256;
      if (type === 0) { const f = new Float32Array(packed.buffer); for (let i = 0; i < f.length; i++) f[i] = Math.sin(i) * .5; }
      else if (type === 1) { for (let i = 0; i < packed.length; i += 2) { packed[i] = i % 256; packed[i + 1] = 0x30; } }
      else for (let off = 0; off < rows * cols / layout.block * layout.bytes; off += layout.bytes) {
        const scale = type === 14 ? off + 208 : off; packed[scale] = 0; packed[scale + 1] = 0x20;
        if (type === 12 || type === 13) { packed[off + 2] = 0; packed[off + 3] = 0x1c; }
      }
      const x = Float32Array.from({ length: (batches + 1) * cols }, (_, i) => Math.cos(i * .17) * .1);
      const wb = gpu.buffer(packed.length, 'test weights'), xb = gpu.buffer(x.byteLength, 'test input'), yb = gpu.buffer(batches * (rows + 2) * 4, 'test output');
      gpu.write(wb, packed); gpu.write(xb, x);
      const pipeline = await gpu.pipeline(matmulShader(type), `test ${layout.name}`);
      gpu.run(pipeline, [wb, xb, yb], [cols, rows, rows + 2, 2, 1, 0, 0, 0], rows, batches);
      const y = new Float32Array(await gpu.read(yb, batches * (rows + 2) * 4));
      for (let b = 0; b < batches; b++) for (let row = 0; row < rows; row++) {
        let expected = 0; for (let c = 0; c < cols; c++) expected += decodeWeight(packed, type, row * cols + c) * x[(b + 1) * cols + c];
        near(y[b * (rows + 2) + row + 2], expected, `${layout.name} b${b} r${row}`);
      }
      log(`PASS ${layout.name}: packed matrix × batch, output offset, input offset`);
      for (const b of [wb, xb, yb]) gpu.destroy(b);
    }
    const heads = 4, kvHeads = 2, hd = 8, start = 3, n = 2, context = 16;
    const q = Float32Array.from({ length: n * heads * hd }, (_, i) => Math.sin(i)), k = Float32Array.from({ length: context * kvHeads * hd }, (_, i) => Math.cos(i)), v = Float32Array.from({ length: k.length }, (_, i) => Math.sin(i / 3));
    const buffers = [gpu.buffer(q.byteLength, 'q'), gpu.buffer(k.byteLength, 'k'), gpu.buffer(v.byteLength, 'v'), gpu.buffer(q.byteLength, 'output')];
    [q, k, v].forEach((a, i) => gpu.write(buffers[i], a));
    const attention = await gpu.pipeline(attentionShader(context), 'attention reference test');
    for (const window of [2, context]) {
      gpu.run(attention, buffers, [heads, kvHeads, hd, start, { f: .3 }, window, 0, 0], heads, n);
      const y = new Float32Array(await gpu.read(buffers[3], q.byteLength));
      for (let b = 0; b < n; b++) for (let h = 0; h < heads; h++) {
        const end = start + b + 1, begin = Math.max(0, end - window), kh = Math.floor(h / (heads / kvHeads)), scores = [];
        for (let t = begin; t < end; t++) { let dot = 0; for (let d = 0; d < hd; d++) dot += q[(b * heads + h) * hd + d] * k[(t * kvHeads + kh) * hd + d]; scores.push(Math.exp(dot * .3)); }
        const total = scores.reduce((a, b) => a + b, 0);
        for (let d = 0; d < hd; d++) { let expected = 0; for (let t = begin; t < end; t++) expected += scores[t - begin] / total * v[(t * kvHeads + kh) * hd + d]; near(y[(b * heads + h) * hd + d], expected, 'attention'); }
      }
      log(`PASS causal grouped-query attention, window ${window}, batch ${n}`);
    }
    const weights = Float32Array.from({ length: hd }, (_, i) => 1 + i * .1), wb = gpu.buffer(weights.byteLength, 'norm weights'); gpu.write(wb, weights);
    const norm = await gpu.pipeline(normShader, 'RMS norm'); gpu.run(norm, [buffers[0], wb, buffers[3]], [hd, { f: 1e-6 }], n * heads);
    const normal = new Float32Array(await gpu.read(buffers[3], q.byteLength));
    for (let g = 0; g < n * heads; g++) { let sq = 0; for (let d = 0; d < hd; d++) sq += q[g * hd + d] ** 2; for (let d = 0; d < hd; d++) near(normal[g * hd + d], q[g * hd + d] * weights[d] / Math.sqrt(sq / hd + 1e-6), 'RMS norm'); }
    log('PASS RMS normalization');
    const rope = await gpu.pipeline(ropeShader, 'QK norm and RoPE'); gpu.run(rope, [buffers[0], wb], [heads, hd, start, { f: 10000 }, { f: .125 }, { f: 1e-6 }, 0, 0], heads, n);
    const rotated = new Float32Array(await gpu.read(buffers[0], q.byteLength));
    for (let b = 0; b < n; b++) for (let h = 0; h < heads; h++) for (let d = 0; d < hd / 2; d++) {
      const base = (b * heads + h) * hd, angle = (start + b) * .125 / 10000 ** (2 * d / hd), a = normal[base + d], z = normal[base + d + hd / 2];
      near(rotated[base + d], a * Math.cos(angle) - z * Math.sin(angle), 'RoPE first half'); near(rotated[base + d + hd / 2], a * Math.sin(angle) + z * Math.cos(angle), 'RoPE second half');
    }
    log('PASS QK normalization + scaled RoPE');
    // Reuse one weight buffer between queued dispatches without a CPU fence.
    const reusable = gpu.buffer(16, 'reused weights'), vector = gpu.buffer(16, 'ones'), outputs = gpu.buffer(8, 'two outputs');
    const linear = await gpu.pipeline(matmulShader(0), 'queue ordering');
    gpu.write(vector, new Float32Array([1, 1, 1, 1]));
    gpu.write(reusable, new Float32Array([1, 2, 3, 4])); gpu.run(linear, [reusable, vector, outputs], [4, 1, 2, 0, 0, 0, 0, 0], 1);
    gpu.write(reusable, new Float32Array([5, 6, 7, 8])); gpu.run(linear, [reusable, vector, outputs], [4, 1, 2, 1, 0, 0, 0, 0, 0], 1);
    const ordered = new Float32Array(await gpu.read(outputs, 8)); near(ordered[0], 10, 'old page consumed'); near(ordered[1], 26, 'new page consumed');
    log('PASS ordered writes preserve both uses of a recycled weight page'); log('ALL 12 GPU CHECKS PASSED');
  } catch (e) { log(`FAIL: ${e instanceof Error ? e.message : e}`); }
  finally { gpu.dispose(); }
};
document.getElementById('model')!.onclick = async () => {
  model?.dispose(); model = new WebGPU(); result.textContent = 'Loading local model…\n';
  const progress = document.createElement('p'); result.before(progress);
  model.onProgress = p => { progress.textContent = p.detail; };
  try {
    const response = await fetch('/model-info'), descriptor = await response.json(); if (!descriptor) throw new Error('Start the byte server with --ollama-model gemma3:27b.'); descriptor.url = new URL(descriptor.url, location.href).href;
    const info = await model.load(descriptor, { gpuBudgetMiB: 2048, contextLength: 512, batchSize: 32 });
    log(JSON.stringify(info));
    for await (const token of model.generate([{ role: 'user', content: 'Name the capital of France. Answer with one word.' }], { temperature: 0, maxTokens: 1 })) { log(JSON.stringify(token)); }
    log('GENERATION COMPLETE');
  } catch (e) { log(`ERROR: ${e instanceof Error ? e.message : e}`); }
  finally { progress.remove(); }
};
document.getElementById('cancel')!.onclick = () => { model?.dispose(); log('Runtime disposed.'); };
