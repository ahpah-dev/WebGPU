import { keysMatch } from './api-key.ts';
import type { GenerationOptions, Message, TokenEvent } from './core/engine.ts';

const DEFAULT_CHANNEL = 'webgpu-runtime';
const CLIENT_TTL_MS = 5 * 60 * 1000;

export type BridgeEvent =
  | { type: 'ready'; model: string; architecture: string; contextLength: number }
  | { type: 'chunk'; delta: string; text: string; tokens: number; promptTokens: number; contextLength: number; seconds: number }
  | { type: 'done'; text: string; tokens: number }
  | { type: 'error'; message: string };

export interface BridgeRequest {
  type: 'webgpu.request';
  id: string;
  apiKey: string;
  method: 'ping' | 'chat.completions';
  messages?: Message[];
  maxTokens?: number;
  temperature?: number;
  topK?: number;
  topP?: number;
}

export interface BridgeResponse { type: 'webgpu.response'; id: string; event: BridgeEvent }

export interface BridgeHost {
  info(): { name: string; architecture: string; contextLength: number; ready: boolean } | undefined;
  generate(messages: Message[], options: GenerationOptions): AsyncGenerator<TokenEvent>;
  busy(): boolean;
}

export interface BridgeHandle { dispose(): void; clientCount(): number }

type Responder = (response: BridgeResponse) => void;
interface MessageSource { postMessage(message: unknown, targetOrigin: string): void }

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

function isMessage(value: unknown): value is Message {
  if (!isRecord(value)) return false;
  return (value.role === 'user' || value.role === 'assistant') && typeof value.content === 'string';
}

function isMessages(value: unknown): value is Message[] {
  return Array.isArray(value) && value.length > 0 && value.every(isMessage);
}

function samplingOptions(request: Record<string, unknown>): GenerationOptions | undefined {
  const options: GenerationOptions = {};
  for (const name of ['maxTokens', 'temperature', 'topK', 'topP'] as const) {
    const value = request[name];
    if (value === undefined) continue;
    if (typeof value !== 'number' || !Number.isFinite(value)) return undefined;
    options[name] = value;
  }
  return options;
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

function messageSource(event: MessageEvent): MessageSource | undefined {
  const source = event.source as unknown as MessageSource | null;
  return source && typeof source.postMessage === 'function' ? source : undefined;
}

/**
 * Accept requests over same-origin BroadcastChannel and over window messages
 * from an embedding page. Every request is rejected unless its key matches.
 */
export function createBridge(host: BridgeHost, expectedKey: string, options: { channelName?: string } = {}): BridgeHandle {
  const channelName = options.channelName ?? DEFAULT_CHANNEL;
  const clients = new Map<string, { seen: number; active: number }>();
  let channel: BroadcastChannel | undefined;
  let disposed = false;
  let generating = false;

  const send = (respond: Responder, id: string, event: BridgeEvent): void => {
    if (disposed) return;
    try { respond({ type: 'webgpu.response', id, event }); } catch { /* The peer may have closed. */ }
  };

  const ping = (id: string, respond: Responder): void => {
    let info: ReturnType<BridgeHost['info']>;
    try { info = host.info(); }
    catch (error) {
      send(respond, id, { type: 'error', message: errorMessage(error) });
      return;
    }
    if (!info || !info.ready) {
      send(respond, id, { type: 'error', message: 'No model is loaded yet. Open a model in the WebGPU tab first.' });
      return;
    }
    send(respond, id, { type: 'ready', model: info.name, architecture: info.architecture, contextLength: info.contextLength });
  };

  const chat = async (id: string, request: Record<string, unknown>, respond: Responder): Promise<void> => {
    const messages = request.messages;
    if (!isMessages(messages)) {
      send(respond, id, { type: 'error', message: 'Send a non-empty messages array of { role: \'user\' | \'assistant\', content: string } objects.' });
      return;
    }
    const generation = samplingOptions(request);
    if (!generation) {
      send(respond, id, { type: 'error', message: 'maxTokens, temperature, topK, and topP must be finite numbers when provided.' });
      return;
    }
    if (generating) {
      send(respond, id, { type: 'error', message: 'The runtime is busy with another generation. Wait for it to finish or stop it in the WebGPU tab.' });
      return;
    }
    try {
      if (host.busy()) {
        send(respond, id, { type: 'error', message: 'The runtime is busy with another generation. Wait for it to finish or stop it in the WebGPU tab.' });
        return;
      }
    } catch (error) {
      send(respond, id, { type: 'error', message: errorMessage(error) });
      return;
    }

    generating = true;
    const client = clients.get(id);
    if (client) client.active++;
    let last: TokenEvent | undefined;
    try {
      for await (const token of host.generate(messages, generation)) {
        if (disposed) break;
        last = token;
        send(respond, id, {
          type: 'chunk', delta: token.delta, text: token.text, tokens: token.tokens,
          promptTokens: token.promptTokens, contextLength: token.contextLength, seconds: token.seconds,
        });
      }
      if (!disposed) send(respond, id, { type: 'done', text: last?.text ?? '', tokens: last?.tokens ?? 0 });
    } catch (error) {
      if (!disposed) send(respond, id, { type: 'error', message: errorMessage(error) });
    } finally {
      generating = false;
      const current = clients.get(id);
      if (current) {
        current.active = Math.max(0, current.active - 1);
        current.seen = Date.now();
      }
    }
  };

  function handle(data: unknown, respond: Responder): void {
    if (disposed || !isRecord(data) || data.type !== 'webgpu.request') return;
    const id = data.id;
    if (typeof id !== 'string' || !id) return;

    if (!keysMatch(expectedKey, data.apiKey)) {
      send(respond, id, { type: 'error', message: 'The API key is invalid. Copy the current key from the WebGPU Developer API panel.' });
      return;
    }

    const now = Date.now();
    const client = clients.get(id);
    if (client) client.seen = now;
    else clients.set(id, { seen: now, active: 0 });

    if (data.method === 'ping') { ping(id, respond); return; }
    if (data.method === 'chat.completions') { void chat(id, data, respond); return; }
    send(respond, id, { type: 'error', message: 'Unknown bridge method. Use "ping" or "chat.completions".' });
  }

  const onChannelMessage = (event: MessageEvent): void => {
    handle(event.data, response => { channel?.postMessage(response); });
  };
  const onWindowMessage = (event: MessageEvent): void => {
    const source = messageSource(event);
    if (!source) return;
    handle(event.data, response => { source.postMessage(response, '*'); });
  };

  if (typeof BroadcastChannel !== 'undefined') {
    try {
      channel = new BroadcastChannel(channelName);
      channel.addEventListener('message', onChannelMessage);
    } catch {
      try { channel?.close(); } catch { /* Nothing to close. */ }
      channel = undefined;
    }
  }
  if (typeof window !== 'undefined') {
    try { window.addEventListener('message', onWindowMessage); } catch { /* No window messaging available. */ }
  }

  return {
    dispose(): void {
      if (disposed) return;
      disposed = true;
      clients.clear();
      if (channel) {
        try { channel.removeEventListener('message', onChannelMessage); } catch { /* Already closed. */ }
        try { channel.close(); } catch { /* Already closed. */ }
        channel = undefined;
      }
      if (typeof window !== 'undefined') {
        try { window.removeEventListener('message', onWindowMessage); } catch { /* No listener was attached. */ }
      }
    },
    clientCount(): number {
      const now = Date.now();
      for (const [id, client] of clients) {
        if (client.active === 0 && now - client.seen > CLIENT_TTL_MS) clients.delete(id);
      }
      return clients.size;
    },
  };
}
