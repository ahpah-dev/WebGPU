import { Compute } from './core/gpu.ts';
import { attentionShader } from './core/shaders.ts';
import { cacheCopies } from './core/context.ts';

/** Exercises real attention at the old limit, the new maximum, and ring wrap.
 * The reference uses absolute token positions and a conventional two-pass softmax. */
export async function verifyContext(gpu: Compute, log: (s: string) => void) {
  for (const scenario of [
    { context: 4096, start: 2047, batch: 3, window: 4096, capacity: 4096, hd: 128 },
    { context: 32768, start: 32765, batch: 3, window: 32768, capacity: 32768, hd: 256 },
    { context: 32768, start: 32765, batch: 3, window: 32768, capacity: 32768, hd: 512 },
    { context: 32768, start: 32760, batch: 8, window: 129, capacity: 136, hd: 128 },
  ]) {
    const { context, start, batch, window, capacity, hd } = scenario, heads = 2, kvHeads = 1;
    const key = (t: number, d: number) => Math.sin(t * .013 + d * .37), value = (t: number, d: number) => Math.cos(t * .019 + d * .23);
    const q = Float32Array.from({ length: batch * heads * hd }, (_, i) => Math.sin(i * .7) * 12);
    const k = new Float32Array(capacity * hd), v = new Float32Array(k.length);
    const begin = Math.max(0, start + batch - capacity);
    for (let t = begin; t < start; t++) for (let d = 0; d < hd; d++) { k[(t % capacity) * hd + d] = key(t, d); v[(t % capacity) * hd + d] = value(t, d); }
    const buffers = [gpu.buffer(q.byteLength, 'long queries'), gpu.buffer(k.byteLength, 'long K'), gpu.buffer(v.byteLength, 'long V'), gpu.buffer(q.byteLength, 'long attention output')];
    try {
      [q, k, v].forEach((a, i) => gpu.write(buffers[i], a));
      const newK = Float32Array.from({ length: batch * hd }, (_, i) => key(start + Math.floor(i / hd), i % hd));
      const newV = Float32Array.from({ length: batch * hd }, (_, i) => value(start + Math.floor(i / hd), i % hd));
      for (const copy of cacheCopies(start, batch, capacity)) for (const [data, dst] of [[newK, buffers[1]], [newV, buffers[2]]] as const)
        gpu.write(dst, data.subarray(copy.source * hd, (copy.source + copy.count) * hd), copy.target * hd * 4);
      const pipeline = await gpu.pipeline(attentionShader(context), 'tiled long-context attention');
      gpu.run(pipeline, buffers, [heads, kvHeads, hd, start, { f: .125 }, window, capacity, 0], heads, batch);
      const actual = new Float32Array(await gpu.read(buffers[3], q.byteLength));
      let error = 0;
      for (let b = 0; b < batch; b++) for (let h = 0; h < heads; h++) {
        const end = start + b + 1, first = Math.max(0, end - window), scores = new Float64Array(end - first);
        let maximum = -Infinity;
        for (let t = first; t < end; t++) { let dot = 0; for (let d = 0; d < hd; d++) dot += q[(b * heads + h) * hd + d] * Math.fround(key(t, d)); scores[t - first] = dot * .125; maximum = Math.max(maximum, dot * .125); }
        let total = 0; for (let i = 0; i < scores.length; i++) { scores[i] = Math.exp(scores[i] - maximum); total += scores[i]; }
        for (let d = 0; d < hd; d++) {
          let expected = 0; for (let t = first; t < end; t++) expected += scores[t - first] / total * Math.fround(value(t, d));
          const got = actual[(b * heads + h) * hd + d]; error = Math.max(error, Math.abs(got - expected));
          if (!Number.isFinite(got) || Math.abs(got - expected) > .001) throw new Error(`Long attention mismatch at token ${start + b}, head ${h}, dimension ${d}: ${got} vs ${expected}`);
        }
      }
      log(`PASS tiled attention: context ${context}, positions ${start}–${start + batch - 1}, head dim ${hd}, window ${window}, cache ${capacity}; max error ${error.toExponential(2)}`);
    } finally { for (const b of buffers) gpu.destroy(b); }
  }
}
