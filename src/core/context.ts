import type { ModelConfig } from './gguf.ts';

export const MAX_CONTEXT = 32768;
export const MAX_MESSAGE_CHARS = 1048576;
export function validateContext(context: number, modelLimit = MAX_CONTEXT) {
  if (!Number.isInteger(context) || context < 256 || context > MAX_CONTEXT) throw new Error(`Context must be an integer from 256 to ${MAX_CONTEXT} tokens.`);
  if (context > modelLimit) throw new Error(`This model declares a ${modelLimit.toLocaleString()}-token context. Choose a smaller context.`);
}
export function outputLimit(requested: number, promptTokens: number, context: number) {
  if (!Number.isInteger(requested) || requested < 1 || requested > MAX_CONTEXT) throw new Error(`maxTokens must be 1–${MAX_CONTEXT}.`);
  if (promptTokens >= context) throw new Error(`The prompt has ${promptTokens.toLocaleString()} tokens; the context limit is ${context.toLocaleString()}. Start a new chat or reload with a longer context.`);
  return Math.min(requested, context - promptTokens);
}
/** Sliding attention needs the window plus the rest of the current prompt batch.
 * This preserves old keys until every query in that batch has consumed them. */
export function attentionMemory(c: ModelConfig, context: number, batch: number) {
  const capacities = Array.from({ length: c.layers }, (_, l) => c.recurrent?.layers[l] ? 0 :
    c.architecture === 'gemma3' && l % c.period !== c.period - 1 ? Math.min(context, c.window + batch - 1) : context);
  const kvBytes = capacities.reduce((a, b) => a + b, 0) * c.kvHeads * c.headDim * 8;
  const r = c.recurrent, recurrentLayers = r?.layers.filter(Boolean).length ?? 0;
  const stateBytes = r ? recurrentLayers * (r.valueHeads * r.headDim ** 2 + (2 * r.keyHeads + r.valueHeads) * r.headDim * (r.convKernel - 1)) * 4 : 0;
  return { capacities, kvBytes, stateBytes, totalBytes: kvBytes + stateBytes };
}
/** Consecutive copies into a ring; splitting at wrap avoids copying one token at a time. */
export function cacheCopies(start: number, count: number, capacity: number) {
  const offset = start % capacity, first = Math.min(count, capacity - offset);
  return [{ source: 0, target: offset, count: first }, ...(first < count ? [{ source: first, target: 0, count: count - first }] : [])];
}
