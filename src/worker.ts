import { PagedEngine, type Message, type LoadOptions, type GenerationOptions } from './core/engine.ts';
import type { SourceInput } from './core/source.ts';
let engine: PagedEngine | undefined;
let current: AbortController | undefined;
let activeId = 0;
self.onmessage = async (event: MessageEvent<{ id: number; action: string; source?: SourceInput; options?: LoadOptions & GenerationOptions; messages?: Message[] }>) => {
  const { id, action } = event.data;
  if (action === 'cancel') { current?.abort(new DOMException('Stopped by user', 'AbortError')); return; }
  if (current) { postMessage({ id, type: 'error', error: 'The runtime is busy.' }); return; }
  activeId = id;
  current = new AbortController();
  try {
    if (action === 'load') {
      engine?.dispose(); engine = new PagedEngine(progress => postMessage({ id: activeId, type: 'progress', value: progress }));
      const info = await engine.load(event.data.source!, event.data.options, current.signal);
      postMessage({ id, type: 'done', value: info });
    } else if (action === 'generate') {
      if (!engine) throw new Error('Load a model first.');
      for await (const token of engine.generate(event.data.messages!, event.data.options, current.signal)) postMessage({ id, type: 'token', value: token });
      postMessage({ id, type: 'done' });
    } else if (action === 'reset') { engine?.reset(); postMessage({ id, type: 'done' }); }
    else throw new Error('Unknown runtime action.');
  } catch (e) { postMessage({ id, type: 'error', error: e instanceof Error ? e.message : String(e), aborted: current.signal.aborted }); }
  finally { current = undefined; }
};
