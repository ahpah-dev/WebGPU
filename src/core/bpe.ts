/** The GPT-2 byte ↔ unicode table shared by every byte-level BPE vocabulary. */
export function byteAlphabet(): string[] {
  const kept = Array.from({ length: 256 }, (_, b) => b).filter(b => b >= 33 && b <= 126 || b >= 161 && b <= 172 || b >= 174);
  const codes = new Set(kept); let extra = 256;
  return Array.from({ length: 256 }, (_, b) => String.fromCodePoint(codes.has(b) ? b : extra++));
}
/** Adjacent-pair BPE with a heap, avoiding a full scan after every merge on long input.
 * Higher scores win; ties keep the original left-to-right ordering. */
export function mergeSymbols(symbols: string[], score: (a: string, b: string) => number | undefined): string[] {
  const n = symbols.length;
  if (n < 2) return symbols;
  const next = Int32Array.from({ length: n }, (_, i) => i + 1 < n ? i + 1 : -1), previous = Int32Array.from({ length: n }, (_, i) => i - 1), version = new Uint32Array(n);
  type Pair = { left: number; right: number; lv: number; rv: number; score: number };
  const heap: Pair[] = [];
  const better = (a: Pair, b: Pair) => a.score > b.score || a.score === b.score && a.left < b.left;
  const enqueue = (left: number) => {
    if (left < 0 || next[left] < 0) return;
    const right = next[left], value = score(symbols[left], symbols[right]);
    if (value === undefined || !(value > -Infinity)) return;
    const pair = { left, right, lv: version[left], rv: version[right], score: value };
    let i = heap.length; heap.push(pair);
    while (i > 0) { const parent = (i - 1) >> 1; if (!better(pair, heap[parent])) break; heap[i] = heap[parent]; i = parent; } heap[i] = pair;
  };
  for (let i = 0; i < n - 1; i++) enqueue(i);
  while (heap.length) {
    const pair = heap[0], tail = heap.pop()!;
    if (heap.length) {
      let i = 0;
      while (2 * i + 1 < heap.length) { let child = 2 * i + 1; if (child + 1 < heap.length && better(heap[child + 1], heap[child])) child++; if (!better(heap[child], tail)) break; heap[i] = heap[child]; i = child; } heap[i] = tail;
    }
    const { left, right, lv, rv } = pair;
    if (next[left] !== right || version[left] !== lv || version[right] !== rv) continue;
    symbols[left] += symbols[right]; version[left]++; version[right]++;
    next[left] = next[right]; if (next[right] >= 0) previous[next[right]] = left; next[right] = -1;
    enqueue(previous[left]); enqueue(left);
  }
  const result: string[] = [];
  for (let i = 0; i >= 0; i = next[i]) result.push(symbols[i]);
  return result;
}
