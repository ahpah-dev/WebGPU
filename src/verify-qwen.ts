import { Compute } from './core/gpu.ts';
import { qwenKernels } from './core/qwen-shaders.ts';
import { ropeShader } from './core/shaders.ts';

export async function verifyQwen(gpu: Compute, log: (s: string) => void) {
  const owned: GPUBuffer[] = [];
  const data = (a: Float32Array<ArrayBuffer>) => { const b = gpu.buffer(a.byteLength, 'Qwen fixture'); gpu.write(b, a); owned.push(b); return b; };
  const empty = (n: number) => data(new Float32Array(n));
  const series = (n: number, scale = 1) => Float32Array.from({ length: n }, (_, i) => Math.sin(i * .31 + .7) * scale);
  const compare = async (buffer: GPUBuffer, expected: ArrayLike<number>, label: string) => {
    const actual = new Float32Array(await gpu.read(buffer, expected.length * 4));
    for (let i = 0; i < actual.length; i++) if (!Number.isFinite(actual[i]) || Math.abs(actual[i] - expected[i]) > .002 * Math.max(1, Math.abs(expected[i]))) throw new Error(`${label} [${i}]: ${actual[i]} != ${expected[i]}`);
  };
  const kernels: Record<string, GPUComputePipeline> = {};
  try {
    for (const [name, source] of Object.entries(qwenKernels)) kernels[name] = await gpu.pipeline(source, `Qwen test ${name}`);
    // Interleaved per-head Q/gate projection, including the next token's stride.
    const hd = 128, kh = 2, vh = 6, batch = 3, channels = (2 * kh + vh) * hd, values = vh * hd;
    const qg = series(2 * batch * values), qb = empty(batch * values), gate = empty(batch * values), attn = series(batch * values, .3), ab = data(attn);
    gpu.run(kernels.splitQuery, [data(qg), qb, gate], [attn.length, hd], Math.ceil(attn.length / 256));
    const query = Float32Array.from(attn, (_, i) => qg[Math.floor(i / hd) * 2 * hd + i % hd]);
    await compare(qb, query, 'query split');
    gpu.run(kernels.sigmoidGate, [gate, ab], [attn.length], Math.ceil(attn.length / 256));
    await compare(ab, Float32Array.from(attn, (v, i) => v / (1 + Math.exp(-qg[Math.floor(i / hd) * 2 * hd + hd + i % hd]))), 'sigmoid attention gate');
    log('PASS Qwen interleaved Q/gate split and sigmoid attention gate');

    const input = series(batch * channels), weights = series(channels * 4, .2), history = series(channels * 3, .1), convExpected = new Float32Array(input.length);
    for (let t = 0; t < batch; t++) for (let c = 0; c < channels; c++) {
      let x = input[t * channels + c] * weights[c * 4 + 3];
      for (let j = 0; j < 3; j++) x += history[c * 3 + j] * weights[c * 4 + j];
      history[c * 3] = history[c * 3 + 1]; history[c * 3 + 1] = history[c * 3 + 2]; history[c * 3 + 2] = input[t * channels + c];
      convExpected[t * channels + c] = x / (1 + Math.exp(-x));
    }
    const cb = empty(input.length), hb = data(series(channels * 3, .1)), wb = data(weights), xb = data(input);
    gpu.run(kernels.conv, [xb, wb, hb, cb], [channels, 4, batch], Math.ceil(channels / 256));
    await compare(cb, convExpected, 'causal convolution'); await compare(hb, history, 'convolution history');
    log('PASS causal convolution + SiLU with carried history');

    for (let t = 0; t < batch; t++) for (let h = 0; h < 2 * kh; h++) {
      const offset = t * channels + h * hd; let ss = 1e-6;
      for (let d = 0; d < hd; d++) ss += convExpected[offset + d] ** 2;
      for (let d = 0; d < hd; d++) convExpected[offset + d] /= Math.sqrt(ss);
    }
    gpu.run(kernels.l2, [cb], [channels, hd, { f: 1e-6 }], 2 * kh, batch);
    await compare(cb, convExpected, 'Q/K L2 normalization');
    const alpha = series(batch * vh, 30), beta = series(batch * vh, 2), a = Float32Array.from(series(vh), x => -Math.exp(x)), dt = series(vh);
    const controls = Float32Array.from({ length: batch * vh * 2 }, (_, i) => { const j = Math.floor(i / 2), z = alpha[j] + dt[j % vh]; return i % 2 ? 1 / (1 + Math.exp(-beta[j])) : Math.exp(a[j % vh] * (Math.max(z, 0) + Math.log1p(Math.exp(-Math.abs(z))))); });
    const control = empty(controls.length);
    gpu.run(kernels.decay, [data(alpha), data(beta), data(a), data(dt), control], [batch * vh, vh], 1);
    await compare(control, controls, 'stable decay and beta');

    const initial = series(vh * hd * hd, .03), state = initial.slice(), expected = new Float32Array(batch * values);
    // Independent scalar recurrent equation: S = decay*S + k*(beta*(v - decay*S*k)).
    for (let t = 0; t < batch; t++) for (let h = 0; h < vh; h++) for (let v = 0; v < hd; v++) {
      const qbase = t * channels + (h % kh) * hd, kbase = qbase + kh * hd, sbase = (h * hd + v) * hd;
      let memory = 0;
      for (let d = 0; d < hd; d++) { state[sbase + d] *= controls[2 * (t * vh + h)]; memory += state[sbase + d] * convExpected[kbase + d]; }
      const correction = (convExpected[t * channels + 2 * kh * hd + h * hd + v] - memory) * controls[2 * (t * vh + h) + 1];
      let output = 0;
      for (let d = 0; d < hd; d++) { state[sbase + d] += convExpected[kbase + d] * correction; output += state[sbase + d] * convExpected[qbase + d] / Math.sqrt(hd); }
      expected[t * values + h * hd + v] = output;
    }
    const sb = data(initial), output = empty(expected.length);
    gpu.run(kernels.delta, [cb, control, sb, output], [hd, kh, vh, batch], vh * hd);
    await compare(output, expected, 'batched recurrent output'); await compare(sb, state, 'recurrent state');
    // Token-at-a-time evaluation must match prefill and carry state across dispatches.
    gpu.write(sb, initial);
    for (let t = 0; t < batch; t++) {
      gpu.run(kernels.delta, [data(convExpected.slice(t * channels, (t + 1) * channels)), data(controls.slice(t * vh * 2, (t + 1) * vh * 2)), sb, output], [hd, kh, vh, 1], vh * hd);
      await compare(output, expected.slice(t * values, (t + 1) * values), 'sequential recurrent output');
    }
    await compare(sb, state, 'sequential final state');
    gpu.clear([sb]); await compare(sb, new Float32Array(initial.length), 'reset recurrent state');
    log('PASS L2 Q/K, decay, tiled V heads and Gated DeltaNet: batch = sequential, carried state and reset');

    const normWeights = Float32Array.from(series(hd), v => v + 1), z = series(expected.length), normalized = new Float32Array(expected.length);
    for (let h = 0; h < batch * vh; h++) {
      let ss = 0; for (let d = 0; d < hd; d++) ss += expected[h * hd + d] ** 2;
      for (let d = 0; d < hd; d++) { const i = h * hd + d; normalized[i] = expected[i] / Math.sqrt(ss / hd + 1e-6) * normWeights[d] * z[i] / (1 + Math.exp(-z[i])); }
    }
    gpu.run(kernels.normGate, [data(expected), data(normWeights), data(z), output], [hd, { f: 1e-6 }], batch * vh);
    await compare(output, normalized, 'gated RMS norm');
    gpu.run(kernels.silu, [data(z), data(expected), output], [expected.length], Math.ceil(expected.length / 256));
    await compare(output, Float32Array.from(expected, (v, i) => v * z[i] / (1 + Math.exp(-z[i]))), 'SwiGLU');
    log('PASS recurrent RMS norm + SiLU gate and SwiGLU feed-forward activation');

    const full = 256, rot = 64, heads = 2, start = 17, x = series(batch * heads * full), rw = Float32Array.from(series(full), v => v + 1), ref = x.slice();
    for (let t = 0; t < batch; t++) for (let h = 0; h < heads; h++) {
      const base = (t * heads + h) * full; let ss = 0;
      for (let d = 0; d < full; d++) ss += x[base + d] ** 2;
      for (let d = 0; d < full; d++) ref[base + d] = x[base + d] * rw[d] / Math.sqrt(ss / full + 1e-6);
      for (let d = 0; d < rot / 2; d++) { const angle = (start + t) / 1e7 ** (2 * d / rot), av = ref[base + d], bv = ref[base + d + rot / 2]; ref[base + d] = av * Math.cos(angle) - bv * Math.sin(angle); ref[base + d + rot / 2] = av * Math.sin(angle) + bv * Math.cos(angle); }
    }
    const rb = data(x), rope = await gpu.pipeline(ropeShader, 'partial RoPE');
    gpu.run(rope, [rb, data(rw)], [heads, full, start, { f: 1e7 }, { f: 1 }, { f: 1e-6 }, rot, 0], heads, batch);
    await compare(rb, ref, 'partial RoPE including normalized unrotated tail');
    log('PASS 64/256 partial RoPE at nonzero positions with unrotated tail');
    log('ALL QWEN GPU CHECKS PASSED');
  } finally { for (const b of owned) gpu.destroy(b); }
}
