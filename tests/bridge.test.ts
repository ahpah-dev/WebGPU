import { test } from 'node:test';
import assert from 'node:assert/strict';
import { randomUUID } from 'node:crypto';
import { API_KEY_ENTRY, generateApiKey, keysMatch, maskApiKey, readApiKey, rotateApiKey } from '../src/api-key.ts';
import type { KeyStore } from '../src/api-key.ts';
import { createBridge } from '../src/bridge.ts';
import type { BridgeEvent, BridgeHost } from '../src/bridge.ts';
import type { GenerationOptions, Message, TokenEvent } from '../src/core/engine.ts';

const KEY_PATTERN = /^wgpu_sk_[0-9a-f]{40}$/;
const KEY_BODY = '0123456789abcdef0123456789abcdef01234567';
const API_KEY = 'wgpu_sk_' + KEY_BODY;

function fakeStore(initial?: string): KeyStore {
  const values = new Map<string, string>();
  if (initial !== undefined) values.set(API_KEY_ENTRY, initial);
  return {
    getItem: key => values.get(key) ?? null,
    setItem: (key, value) => { values.set(key, value); },
  };
}

function tokenEvent(text: string, delta: string, tokens: number, seconds: number): TokenEvent {
  return { token: tokens, text, delta, tokens, promptTokens: 4, contextLength: 64, seconds, stats: {} as TokenEvent['stats'] };
}

class FakeHost implements BridgeHost {
  generated = 0;
  lastMessages: Message[] | undefined;
  lastOptions: GenerationOptions | undefined;
  info() { return { name: 'qwen3.8:27b', architecture: 'qwen35', contextLength: 4096, ready: true }; }
  busy() { return false; }
  async *generate(messages: Message[], options: GenerationOptions): AsyncGenerator<TokenEvent> {
    this.generated++;
    this.lastMessages = messages;
    this.lastOptions = options;
    yield tokenEvent('Hel', 'Hel', 1, 0.25);
    yield tokenEvent('Hello', 'lo', 2, 0.5);
  }
}

class BlockingHost implements BridgeHost {
  started = 0;
  finished = false;
  release: () => void;
  private gate: Promise<void>;
  constructor() {
    let release = () => {};
    this.gate = new Promise<void>(resolve => { release = resolve; });
    this.release = release;
  }
  info() { return { name: 'qwen3.8:27b', architecture: 'qwen35', contextLength: 1024, ready: true }; }
  busy() { return this.started > 0 && !this.finished; }
  async *generate(): AsyncGenerator<TokenEvent> {
    this.started++;
    yield tokenEvent('One', 'One', 1, 0.1);
    await this.gate;
    yield tokenEvent('One two', ' two', 2, 0.2);
    this.finished = true;
  }
}

class FakeWindow {
  private listeners = new Set<(event: MessageEvent) => void>();
  addEventListener(type: string, listener: (event: MessageEvent) => void): void { if (type === 'message') this.listeners.add(listener); }
  removeEventListener(type: string, listener: (event: MessageEvent) => void): void { if (type === 'message') this.listeners.delete(listener); }
  dispatch(event: MessageEvent): void { for (const listener of [...this.listeners]) listener(event); }
}

type Terminal = (event: BridgeEvent) => boolean;
const TERMINAL: Terminal = event => event.type === 'done' || event.type === 'error' || event.type === 'ready';

/** Send one wire request on a client channel and collect its response events. */
function send(
  channel: BroadcastChannel,
  request: Record<string, unknown>,
  options: { onEvent?: (event: BridgeEvent) => void; terminal?: Terminal } = {},
): Promise<BridgeEvent[]> {
  const terminal = options.terminal ?? TERMINAL;
  return new Promise((resolve, reject) => {
    const events: BridgeEvent[] = [];
    const timer = setTimeout(() => { cleanup(); reject(new Error(`Timed out waiting for bridge response to ${String(request.id)}.`)); }, 4000);
    function cleanup() { clearTimeout(timer); channel.removeEventListener('message', listener); }
    function listener(event: MessageEvent): void {
      const data = event.data as { type?: string; id?: string; event?: BridgeEvent } | null;
      if (!data || data.type !== 'webgpu.response' || data.id !== request.id || !data.event) return;
      events.push(data.event);
      options.onEvent?.(data.event);
      if (terminal(data.event)) { cleanup(); resolve(events); }
    }
    channel.addEventListener('message', listener);
    channel.postMessage(request);
  });
}

function channelName(label: string): string {
  return `webgpu-runtime-test-${label}-${randomUUID()}`;
}

test('generateApiKey uses the injected random source and always emits the documented format', () => {
  const filled = generateApiKey(bytes => { bytes.fill(0xab); return bytes; });
  assert.equal(filled, 'wgpu_sk_' + 'ab'.repeat(20));
  assert.match(filled, KEY_PATTERN);

  const ascending = generateApiKey(bytes => { for (let i = 0; i < bytes.length; i++) bytes[i] = i; return bytes; });
  assert.equal(ascending, 'wgpu_sk_' + Array.from({ length: 20 }, (_, i) => i.toString(16).padStart(2, '0')).join(''));

  assert.match(generateApiKey(), KEY_PATTERN);
});

test('maskApiKey reveals only four leading and four trailing secret characters', () => {
  const masked = maskApiKey(API_KEY);
  assert.equal(masked, 'wgpu_sk_0123…4567');
  assert.equal(masked.includes(KEY_BODY), false);
  assert.equal(maskApiKey('short'), '…');
  assert.equal(maskApiKey(''), '…');
});

test('keysMatch accepts identical strings and rejects mismatches and non-strings', () => {
  assert.equal(keysMatch(API_KEY, API_KEY), true);
  assert.equal(keysMatch(API_KEY, API_KEY.slice(0, -1)), false);
  assert.equal(keysMatch(API_KEY, API_KEY + '0'), false);
  assert.equal(keysMatch(API_KEY, 'x' + API_KEY.slice(1)), false);
  assert.equal(keysMatch(API_KEY, API_KEY.slice(0, 20) + 'x' + API_KEY.slice(21)), false);
  assert.equal(keysMatch(API_KEY, API_KEY.toUpperCase()), false);
  for (const value of [undefined, null, 0, 42, {}, [], new String(API_KEY)]) assert.equal(keysMatch(API_KEY, value), false);
});

test('readApiKey persists, stays stable, rotates, and repairs malformed stored values', () => {
  const store = fakeStore();
  const first = readApiKey(store);
  assert.match(first, KEY_PATTERN);
  assert.equal(store.getItem(API_KEY_ENTRY), first);
  assert.equal(readApiKey(store), first);

  const rotated = rotateApiKey(store);
  assert.match(rotated, KEY_PATTERN);
  assert.equal(store.getItem(API_KEY_ENTRY), rotated);
  assert.equal(readApiKey(store), rotated);

  store.setItem(API_KEY_ENTRY, 'malformed');
  const repaired = readApiKey(store);
  assert.match(repaired, KEY_PATTERN);
  assert.notEqual(repaired, 'malformed');
  assert.equal(store.getItem(API_KEY_ENTRY), repaired);
  assert.equal(readApiKey(store), repaired);
});

test('readApiKey degrades to a stable in-memory key when storage is denied', () => {
  const denied: KeyStore = {
    getItem() { throw new Error('storage denied'); },
    setItem() { throw new Error('storage denied'); },
  };
  const first = readApiKey(denied);
  assert.match(first, KEY_PATTERN);
  assert.equal(readApiKey(denied), first);
  const rotated = rotateApiKey(denied);
  assert.match(rotated, KEY_PATTERN);
  assert.equal(readApiKey(denied), rotated);
});

test('bridge authenticates over BroadcastChannel, streams chunks, reports ping info, and counts clients', async t => {
  const name = channelName('end-to-end'), host = new FakeHost();
  const bridge = createBridge(host, API_KEY, { channelName: name });
  const channel = new BroadcastChannel(name);
  t.after(() => { bridge.dispose(); channel.close(); });

  const denied = await send(channel, {
    type: 'webgpu.request', id: 'denied-client', apiKey: 'wgpu_sk_' + 'ff'.repeat(20), method: 'chat.completions',
    messages: [{ role: 'user', content: 'Hello' }],
  });
  assert.equal(denied.length, 1);
  assert.equal(denied[0]?.type, 'error');
  if (denied[0]?.type === 'error') assert.match(denied[0].message, /api key is invalid/i);
  assert.equal(JSON.stringify(denied).includes(API_KEY), false);
  assert.equal(host.generated, 0);
  assert.equal(bridge.clientCount(), 0);

  const ping = await send(channel, { type: 'webgpu.request', id: 'tab-1', apiKey: API_KEY, method: 'ping' });
  assert.deepEqual(ping, [{ type: 'ready', model: 'qwen3.8:27b', architecture: 'qwen35', contextLength: 4096 }]);
  assert.equal(bridge.clientCount(), 1);

  const messages: Message[] = [{ role: 'user', content: 'Hi' }];
  const events = await send(channel, {
    type: 'webgpu.request', id: 'tab-1', apiKey: API_KEY, method: 'chat.completions', messages,
    maxTokens: 32, temperature: 0.2, topK: 10, topP: 0.8,
  });
  assert.deepEqual(events, [
    { type: 'chunk', delta: 'Hel', text: 'Hel', tokens: 1, promptTokens: 4, contextLength: 64, seconds: 0.25 },
    { type: 'chunk', delta: 'lo', text: 'Hello', tokens: 2, promptTokens: 4, contextLength: 64, seconds: 0.5 },
    { type: 'done', text: 'Hello', tokens: 2 },
  ]);
  assert.deepEqual(host.lastMessages, messages);
  assert.deepEqual(host.lastOptions, { maxTokens: 32, temperature: 0.2, topK: 10, topP: 0.8 });
  assert.equal(host.generated, 1);
  assert.equal(bridge.clientCount(), 1);

  bridge.dispose();
  assert.equal(bridge.clientCount(), 0);
});

test('bridge rejects a concurrent chat while the host is busy without starting a second generation', async t => {
  const name = channelName('busy'), host = new BlockingHost();
  const bridge = createBridge(host, API_KEY, { channelName: name });
  const channel = new BroadcastChannel(name);
  let signalChunk = () => {};
  const chunkSeen = new Promise<void>(resolve => { signalChunk = resolve; });
  const first = send(channel, {
    type: 'webgpu.request', id: 'streaming-client', apiKey: API_KEY, method: 'chat.completions',
    messages: [{ role: 'user', content: 'First' }],
  }, { onEvent: event => { if (event.type === 'chunk') signalChunk(); } });
  t.after(async () => {
    host.release();
    await first.catch(() => {});
    bridge.dispose();
    channel.close();
  });

  await chunkSeen;
  assert.equal(host.busy(), true);

  const second = await send(channel, {
    type: 'webgpu.request', id: 'second-client', apiKey: API_KEY, method: 'chat.completions',
    messages: [{ role: 'user', content: 'Second' }],
  });
  assert.equal(second.length, 1);
  assert.equal(second[0]?.type, 'error');
  if (second[0]?.type === 'error') assert.match(second[0].message, /busy/i);
  assert.equal(host.started, 1);

  host.release();
  const firstEvents = await first;
  assert.deepEqual(firstEvents.map(event => event.type), ['chunk', 'chunk', 'done']);
  assert.equal(host.started, 1);
  assert.equal(bridge.clientCount(), 2);
});

test('bridge replies to an embedding window through source.postMessage with a wildcard origin', async t => {
  const fakeWindow = new FakeWindow();
  const previous = (globalThis as { window?: unknown }).window;
  (globalThis as { window?: unknown }).window = fakeWindow;
  const host = new FakeHost();
  const bridge = createBridge(host, API_KEY, { channelName: channelName('window') });
  t.after(() => {
    bridge.dispose();
    if (previous === undefined) delete (globalThis as { window?: unknown }).window;
    else (globalThis as { window?: unknown }).window = previous;
  });

  const replies: Array<{ message: unknown; targetOrigin: string }> = [];
  const source = { postMessage(message: unknown, targetOrigin: string) { replies.push({ message, targetOrigin }); } };
  fakeWindow.dispatch({ source, data: { type: 'webgpu.request', id: 'frame-1', apiKey: API_KEY, method: 'ping' } } as unknown as MessageEvent);

  assert.equal(replies.length, 1);
  assert.equal(replies[0]?.targetOrigin, '*');
  assert.deepEqual(replies[0]?.message, {
    type: 'webgpu.response', id: 'frame-1',
    event: { type: 'ready', model: 'qwen3.8:27b', architecture: 'qwen35', contextLength: 4096 },
  });
  assert.equal(bridge.clientCount(), 1);
});
