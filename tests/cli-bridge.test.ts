import { test } from 'node:test';
import assert from 'node:assert/strict';
import { createServer } from 'node:http';
import type { AddressInfo } from 'node:net';
import { createCliBridge } from '../server/cli-bridge.mjs';

const settle = () => new Promise(resolve => setTimeout(resolve, 80));

interface Harness { base: string; close(): Promise<void> }

async function harness(): Promise<Harness> {
  const bridge = createCliBridge();
  const server = createServer((req, res) => {
    const url = new URL(req.url ?? '/', 'http://127.0.0.1');
    if (!bridge.handle(req, res, url)) { res.writeHead(404); res.end(); }
  });
  await new Promise<void>(resolve => server.listen(0, '127.0.0.1', resolve));
  const { port } = server.address() as AddressInfo;
  return {
    base: `http://127.0.0.1:${port}`,
    async close() {
      bridge.dispose();
      await new Promise<void>(resolve => server.close(() => resolve()));
    },
  };
}

interface Attachment { promise: Promise<Response | undefined>; cancel(): Promise<void> }

function attach(base: string, key: string, model: string | null = 'Qwen3.5-0.8B'): Attachment {
  const controller = new AbortController();
  const promise = fetch(`${base}/bridge/poll`, {
    method: 'POST',
    signal: controller.signal,
    headers: { authorization: `Bearer ${key}`, 'content-type': 'application/json' },
    body: JSON.stringify({ model, architecture: model ? 'qwen35' : null, contextLength: model ? 4096 : null }),
  }).catch(() => undefined);
  return {
    promise,
    async cancel() { controller.abort(); await promise; },
  };
}

async function event(base: string, key: string, id: string, body: unknown) {
  const response = await fetch(`${base}/bridge/jobs/${id}/events`, {
    method: 'POST',
    headers: { authorization: `Bearer ${key}`, 'content-type': 'application/json' },
    body: JSON.stringify(body),
  });
  assert.equal(response.status, 200);
  return await response.json() as { ok: boolean; cancelled: boolean };
}

/** Starts a completion and waits for the tab to receive the job, without awaiting the reply. */
async function start(base: string, key: string, path: string, payload: unknown, headers: Record<string, string> = {}) {
  const pending = attach(base, key);
  await settle();
  const response = fetch(`${base}${path}`, {
    method: 'POST',
    headers: { authorization: `Bearer ${key}`, 'content-type': 'application/json', ...headers },
    body: JSON.stringify(payload),
  });
  const polled = await pending.promise;
  assert.ok(polled, 'the tab poll was aborted');
  assert.equal(polled.status, 200);
  const job = await polled.json() as { id: string; messages: { role: string; content: string }[]; options: Record<string, unknown> };
  return { response, job };
}

test('OpenAI-compatible completions stream through the tab', async () => {
  const server = await harness();
  try {
    const { response: pending, job } = await start(server.base, 'k1', '/v1/chat/completions', {
      model: 'local', stream: true, temperature: 0.2, max_tokens: 16,
      messages: [{ role: 'system', content: 'Be terse.' }, { role: 'user', content: 'Capital of France?' }],
    });
    assert.deepEqual(job.messages, [{ role: 'user', content: 'Be terse.\n\nCapital of France?' }]);
    assert.equal(job.options.maxTokens, 16);
    assert.equal(job.options.temperature, 0.2);

    await event(server.base, 'k1', job.id, { type: 'chunk', delta: 'Pari', text: 'Pari', tokens: 1, promptTokens: 5 });
    await event(server.base, 'k1', job.id, { type: 'chunk', delta: 's', text: 'Paris', tokens: 2 });
    await event(server.base, 'k1', job.id, { type: 'done', text: 'Paris', tokens: 2 });
    const response = await pending;
    assert.equal(response.status, 200);
    assert.match(response.headers.get('content-type') ?? '', /text\/event-stream/);
    const body = await response.text();
    assert.match(body, /"content":"Pari"/);
    assert.match(body, /"content":"s"/);
    assert.match(body, /"finish_reason":"stop"/);
    assert.match(body, /"prompt_tokens":5/);
    assert.match(body, /data: \[DONE\]/);
  } finally { await server.close(); }
});

test('Anthropic message blocks flatten and stream as text deltas', async () => {
  const server = await harness();
  try {
    const { response: pending, job } = await start(server.base, 'k2', '/v1/messages', {
      model: 'local', max_tokens: 32, stream: true,
      system: [{ type: 'text', text: 'You are a build tool.' }],
      messages: [{ role: 'user', content: [{ type: 'text', text: 'Fix the test.' }, { type: 'tool_result', tool_use_id: 't1', content: [{ type: 'text', text: 'exit code 1' }] }] }],
    }, { 'x-api-key': 'k2' });
    assert.equal(job.messages[0].content, 'You are a build tool.\n\nFix the test.\n[tool result] exit code 1');

    await event(server.base, 'k2', job.id, { type: 'chunk', delta: 'Done', text: 'Done', tokens: 1, promptTokens: 9 });
    await event(server.base, 'k2', job.id, { type: 'done', text: 'Done', tokens: 1 });
    const response = await pending;
    assert.equal(response.status, 200);
    const body = await response.text();
    assert.match(body, /event: message_start/);
    assert.match(body, /event: content_block_delta/);
    assert.match(body, /"type":"text_delta","text":"Done"/);
    assert.match(body, /event: message_stop/);
    assert.match(body, /"stop_reason":"end_turn"/);
    assert.match(body, /"input_tokens":9/);
  } finally { await server.close(); }
});

test('non-streaming Anthropic and OpenAI replies aggregate tokens', async () => {
  const server = await harness();
  try {
    const openai = await start(server.base, 'k3', '/v1/chat/completions', { messages: [{ role: 'user', content: 'Hi' }], max_tokens: 8 });
    await event(server.base, 'k3', openai.job.id, { type: 'chunk', delta: 'Hello', text: 'Hello', tokens: 1, promptTokens: 2 });
    await event(server.base, 'k3', openai.job.id, { type: 'done', text: 'Hello', tokens: 1 });
    const openaiResponse = await openai.response;
    assert.equal(openaiResponse.status, 200);
    const reply = await openaiResponse.json() as { choices: { message: { content: string } }[]; usage: { completion_tokens: number } };
    assert.equal(reply.choices[0].message.content, 'Hello');
    assert.equal(reply.usage.completion_tokens, 1);

    const anthropic = await start(server.base, 'k4', '/v1/messages', { max_tokens: 8, messages: [{ role: 'user', content: 'Hi' }] }, { 'x-api-key': 'k4' });
    await event(server.base, 'k4', anthropic.job.id, { type: 'chunk', delta: 'Hello', text: 'Hello', tokens: 1, promptTokens: 2 });
    await event(server.base, 'k4', anthropic.job.id, { type: 'done', text: 'Hello', tokens: 1 });
    const messageResponse = await anthropic.response;
    assert.equal(messageResponse.status, 200);
    const message = await messageResponse.json() as { type: string; content: { type: string; text: string }[]; usage: { output_tokens: number } };
    assert.equal(message.type, 'message');
    assert.deepEqual(message.content, [{ type: 'text', text: 'Hello' }]);
    assert.equal(message.usage.output_tokens, 1);
  } finally { await server.close(); }
});

test('requests without a connected tab fail with a useful error', async () => {
  const server = await harness();
  try {
    const missing = await fetch(`${server.base}/v1/chat/completions`, {
      method: 'POST', headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ messages: [{ role: 'user', content: 'Hi' }] }),
    });
    assert.equal(missing.status, 401);

    const idle = attach(server.base, 'k5', null);
    await settle();
    const noModel = await fetch(`${server.base}/v1/messages`, {
      method: 'POST', headers: { 'x-api-key': 'k5', 'content-type': 'application/json' },
      body: JSON.stringify({ max_tokens: 8, messages: [{ role: 'user', content: 'Hi' }] }),
    });
    assert.equal(noModel.status, 503);
    assert.match(await noModel.text(), /No model is loaded/);
    await idle.cancel();
  } finally { await server.close(); }
});

test('hello announces a model before the next poll', async () => {
  const server = await harness();
  try {
    const hello = await fetch(`${server.base}/bridge/hello`, {
      method: 'POST', headers: { authorization: 'Bearer k7', 'content-type': 'application/json' },
      body: JSON.stringify({ model: 'Gemma 3 4B', architecture: 'gemma3', contextLength: 8192 }),
    });
    assert.equal(hello.status, 204);
    const models = await fetch(`${server.base}/v1/models`, { headers: { authorization: 'Bearer k7' } });
    const list = await models.json() as { data: { id: string }[] };
    assert.deepEqual(list.data.map(model => model.id), ['gemma-3-4b']);
  } finally { await server.close(); }
});

test('model list, token estimate, and malformed requests', async () => {
  const server = await harness();
  try {
    const pending = attach(server.base, 'k6');
    await settle();
    const models = await fetch(`${server.base}/v1/models`, { headers: { authorization: 'Bearer k6' } });
    assert.equal(models.status, 200);
    const list = await models.json() as { data: { id: string }[] };
    assert.deepEqual(list.data.map(model => model.id), ['qwen3.5-0.8b']);

    const count = await fetch(`${server.base}/v1/messages/count_tokens`, {
      method: 'POST', headers: { 'x-api-key': 'k6', 'content-type': 'application/json' },
      body: JSON.stringify({ messages: [{ role: 'user', content: '12345678' }] }),
    });
    assert.equal(count.status, 200);
    assert.deepEqual(await count.json(), { input_tokens: 2 });

    const bad = await fetch(`${server.base}/v1/messages`, {
      method: 'POST', headers: { 'x-api-key': 'k6', 'content-type': 'application/json' },
      body: JSON.stringify({ messages: [] }),
    });
    assert.equal(bad.status, 400);
    assert.equal((await bad.json() as { type: string }).type, 'error');
    await pending.cancel();
  } finally { await server.close(); }
});
