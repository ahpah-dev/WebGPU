import type { GenerationOptions, LoadOptions, Message, ModelInfo, Progress, TokenEvent, PerformanceMode } from './core/engine.ts';
import type { SourceInput } from './core/source.ts';
export type { GenerationOptions, LoadOptions, Message, ModelInfo, Progress, TokenEvent, SourceInput, PerformanceMode };

/** Browser API. Worker owns all WebGPU resources; no inference server is used. */
export class WebGPU {
  private worker: Worker;
  private sequence = 0;
  private busy = false;
  private disposed = false;
  private rejectPending?: (e: Error) => void;
  onProgress?: (progress: Progress) => void;
  constructor() { this.worker = new Worker(new URL('./worker.ts', import.meta.url), { type: 'module' }); }
  private request(action: string, payload: object, token?: (t: TokenEvent) => void): Promise<ModelInfo> {
    if (this.disposed) return Promise.reject(new Error('This runtime was disposed. Create a new WebGPU instance.'));
    if (this.busy) return Promise.reject(new Error('The runtime is busy.'));
    this.busy = true; const id = ++this.sequence;
    return new Promise((resolve, reject) => {
      this.rejectPending = reject;
      const finish = () => { this.busy = false; this.rejectPending = undefined; this.worker.removeEventListener('message', listener); this.worker.removeEventListener('error', error); };
      const error = (e: ErrorEvent) => { finish(); reject(new Error(e.message)); };
      const listener = (event: MessageEvent) => {
        const m = event.data; if (m.id !== id) return;
        if (m.type === 'progress') this.onProgress?.(m.value);
        if (m.type === 'token') token?.(m.value);
        if (m.type === 'done') { finish(); resolve(m.value); }
        if (m.type === 'error') { finish(); const e = new Error(m.error); if (m.aborted) e.name = 'AbortError'; reject(e); }
      };
      this.worker.addEventListener('message', listener); this.worker.addEventListener('error', error);
      this.worker.postMessage({ id, action, ...payload });
    });
  }
  load(source: SourceInput, options: LoadOptions = {}): Promise<ModelInfo> { return this.request('load', { source, options }); }
  async *generate(messages: Message[], options: GenerationOptions = {}): AsyncGenerator<TokenEvent> {
    const queue: TokenEvent[] = []; let ended = false, failure: unknown;
    let wake = () => {};
    const task = this.request('generate', { messages, options }, t => { queue.push(t); wake(); }).catch(e => { failure = e; }).finally(() => { ended = true; wake(); });
    try {
      while (!ended || queue.length) {
        if (queue.length) yield queue.shift()!;
        else await new Promise<void>(resolve => { wake = resolve; });
      }
      if (failure) throw failure;
    } finally { if (!ended) this.stop(); await task; }
  }
  stop() { this.worker.postMessage({ action: 'cancel' }); }
  async reset() { await this.request('reset', {}); }
  dispose() { this.disposed = true; this.worker.terminate(); this.rejectPending?.(new Error('Runtime disposed.')); this.rejectPending = undefined; this.busy = false; }
}
