import type { BridgeHost } from './bridge.ts';
import type { GenerationOptions, Message, TokenEvent } from './core/engine.ts';

const LOCAL_HOSTS = new Set(['127.0.0.1', 'localhost', '[::1]', '::1']);
const RETRY_MS = 2_000;

export type CliBridgeStatus = 'starting' | 'ready' | 'offline' | 'disabled';

export interface CliBridgeHandle {
  dispose(): void;
  status(): CliBridgeStatus;
  /** Pushes the current model state to the relay without waiting for the next poll. */
  announce(): void;
}

interface CliJob { id: string; messages: Message[]; options: GenerationOptions }

interface CliEvent {
  type: 'chunk' | 'done' | 'error';
  delta?: string;
  text?: string;
  tokens?: number;
  promptTokens?: number;
  contextLength?: number;
  seconds?: number;
  message?: string;
}

/**
 * Connects this tab to the loopback relay in `server/cli-bridge.mjs`, so
 * Anthropic- and OpenAI-compatible CLIs can use the loaded model. Only runs
 * when the app is served from a local host; static hosting has no relay.
 */
export function createCliBridge(host: BridgeHost, key: string): CliBridgeHandle {
  if (typeof location === 'undefined' || !LOCAL_HOSTS.has(location.hostname)) {
    return { dispose() { /* Nothing to stop. */ }, status: () => 'disabled', announce() { /* No relay on static hosting. */ } };
  }

  let disposed = false;
  let status: CliBridgeStatus = 'starting';
  let controller = new AbortController();
  let active: AsyncGenerator<TokenEvent> | undefined;

  const headers = { authorization: `Bearer ${key}`, 'content-type': 'application/json' };
  const sleep = (ms: number) => new Promise(resolve => setTimeout(resolve, ms));
  const describe = () => {
    const info = host.info();
    return { model: info?.name ?? null, architecture: info?.architecture ?? null, contextLength: info?.contextLength ?? null };
  };

  async function postEvent(id: string, body: CliEvent): Promise<{ cancelled?: boolean } | undefined> {
    try {
      const response = await fetch(`/bridge/jobs/${id}/events`, { method: 'POST', headers, body: JSON.stringify(body), signal: controller.signal });
      if (!response.ok) return undefined;
      return await response.json() as { cancelled?: boolean };
    } catch {
      return undefined;
    }
  }

  async function run(job: CliJob): Promise<void> {
    const info = host.info();
    if (!info?.ready) { await postEvent(job.id, { type: 'error', message: 'No model is loaded in this tab. Open a model in the WebGPU tab first.' }); return; }
    if (host.busy()) { await postEvent(job.id, { type: 'error', message: 'The runtime is busy with another generation. Wait for it or stop it in the WebGPU tab.' }); return; }

    const generation = host.generate(job.messages, job.options);
    active = generation;
    let last: TokenEvent | undefined;
    try {
      for await (const token of generation) {
        if (disposed) break;
        last = token;
        const reply = await postEvent(job.id, {
          type: 'chunk', delta: token.delta, text: token.text, tokens: token.tokens,
          promptTokens: token.promptTokens, contextLength: token.contextLength, seconds: token.seconds,
        });
        if (reply?.cancelled) break;
      }
      if (!disposed) await postEvent(job.id, { type: 'done', text: last?.text ?? '', tokens: last?.tokens ?? 0 });
    } catch (error) {
      if (!disposed) await postEvent(job.id, { type: 'error', message: error instanceof Error ? error.message : String(error) });
    } finally {
      active = undefined;
    }
  }

  async function loop(): Promise<void> {
    while (!disposed) {
      try {
        const response = await fetch('/bridge/poll', {
          method: 'POST', headers, signal: controller.signal,
          body: JSON.stringify(describe()),
        });
        if (disposed) return;
        if (response.status === 404) { status = 'offline'; return; }
        if (response.status === 204) { status = 'ready'; continue; }
        if (!response.ok) { status = 'offline'; await sleep(RETRY_MS); continue; }
        status = 'ready';
        await run(await response.json() as CliJob);
      } catch {
        if (disposed || controller.signal.aborted) return;
        status = 'offline';
        await sleep(RETRY_MS);
      }
    }
  }
  void loop();

  return {
    dispose(): void {
      if (disposed) return;
      disposed = true;
      controller.abort();
      controller = new AbortController();
      void active?.return(undefined);
      active = undefined;
    },
    status: () => status,
    announce(): void {
      if (disposed) return;
      void fetch('/bridge/hello', { method: 'POST', headers, body: JSON.stringify(describe()), signal: controller.signal }).catch(() => undefined);
    },
  };
}
