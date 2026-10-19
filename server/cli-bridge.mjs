// Loopback relay for coding CLIs. Node never evaluates the model: requests are
// parked until the browser tab picks them up, and the tab posts token events
// back here. This module only translates HTTP/SSE shapes in both directions.
import { randomUUID } from 'node:crypto';

const TAB_TTL_MS = 30_000;
const POLL_WAIT_MS = 20_000;
const QUEUE_LIMIT = 4;
const IDLE_TIMEOUT_MS = 300_000;
const SWEEP_MS = 5_000;
const KEEPALIVE_MS = 15_000;
const MAX_BODY = 4 * 1024 * 1024;
const trace = process.env.WEBGPU_BRIDGE_LOG === '1' ? (...parts) => console.error('[relay]', ...parts) : () => undefined;

const isRecord = value => typeof value === 'object' && value !== null && !Array.isArray(value);

function text(value) {
  return typeof value === 'string' && value ? value : undefined;
}

function finite(value) {
  return typeof value === 'number' && Number.isFinite(value) ? value : undefined;
}

function requestKey(req) {
  const header = req.headers.authorization;
  if (typeof header === 'string' && /^Bearer\s+/i.test(header)) return header.slice(7).trim();
  const key = req.headers['x-api-key'];
  return typeof key === 'string' ? key.trim() : '';
}

function readJson(req) {
  return new Promise((resolve, reject) => {
    let size = 0, settled = false;
    const chunks = [];
    const fail = error => { if (settled) return; settled = true; reject(error); };
    req.on('data', chunk => {
      size += chunk.length;
      if (size > MAX_BODY) { fail(Object.assign(new Error('The request body is too large.'), { status: 413 })); req.destroy(); return; }
      chunks.push(chunk);
    });
    req.on('end', () => {
      if (settled) return;
      settled = true;
      if (!chunks.length) return resolve({});
      try { resolve(JSON.parse(Buffer.concat(chunks).toString('utf8'))); }
      catch { reject(Object.assign(new Error('Send a JSON request body.'), { status: 400 })); }
    });
    req.on('error', error => fail(Object.assign(error, { status: 400 })));
  });
}

function sendJson(res, status, value) {
  if (res.writableEnded) return;
  if (res.headersSent) { res.end(JSON.stringify(value)); return; }
  res.writeHead(status, { 'Content-Type': 'application/json; charset=utf-8', 'Cache-Control': 'no-store' });
  res.end(JSON.stringify(value));
}

function sendNoContent(res) {
  if (res.writableEnded) return;
  res.writeHead(204, { 'Cache-Control': 'no-store' });
  res.end();
}

/** Flattens Anthropic content blocks and OpenAI content parts into plain text. */
function contentText(content) {
  if (typeof content === 'string') return content;
  if (isRecord(content)) return contentText([content]);
  if (!Array.isArray(content)) return '';
  const parts = [];
  for (const part of content) {
    if (typeof part === 'string') { parts.push(part); continue; }
    if (!isRecord(part)) continue;
    switch (part.type) {
      case 'text': if (typeof part.text === 'string') parts.push(part.text); break;
      case 'tool_result': parts.push(`[tool result] ${contentText(part.content)}`); break;
      case 'tool_use': parts.push(`[tool call] ${String(part.name ?? 'tool')}(${JSON.stringify(part.input ?? {})})`); break;
      case 'image': case 'document': parts.push(`[${part.type} omitted]`); break;
      default: break;
    }
  }
  return parts.join('\n');
}

function mergeAdjacent(messages) {
  const merged = [];
  for (const message of messages) {
    const previous = merged[merged.length - 1];
    if (previous && previous.role === message.role) previous.content += `\n\n${message.content}`;
    else merged.push({ ...message });
  }
  return merged;
}

function withSystem(messages, system) {
  if (!system) return messages;
  const index = messages.findIndex(message => message.role === 'user');
  if (index < 0) return [{ role: 'user', content: system }, ...messages];
  messages[index].content = `${system}\n\n${messages[index].content}`;
  return messages;
}

function generationOptions(maxTokens, temperature, topK, topP, contextLength) {
  const options = {};
  if (maxTokens !== undefined) {
    const ceiling = contextLength ? Math.max(1, Math.min(contextLength - 1, 32768)) : 4096;
    options.maxTokens = Math.max(1, Math.min(Math.floor(maxTokens), ceiling));
  }
  if (temperature !== undefined) options.temperature = Math.max(0, Math.min(temperature, 2));
  if (topK !== undefined) options.topK = Math.max(1, Math.floor(topK));
  if (topP !== undefined) options.topP = Math.max(0.01, Math.min(topP, 1));
  return options;
}

function replyCeiling(messages, contextLength) {
  if (!contextLength) return 4096;
  const characters = messages.reduce((sum, message) => sum + message.content.length, 0);
  const estimate = Math.ceil(characters / 3) + messages.length * 4;
  return Math.max(32, contextLength - estimate - 8);
}

function anthropicRequest(body, contextLength) {
  const messages = [], system = [];
  if (!Array.isArray(body.messages) || body.messages.length === 0) throw Object.assign(new Error('Send a non-empty messages array.'), { status: 400 });
  for (const message of body.messages) {
    if (!isRecord(message) || typeof message.role !== 'string') throw Object.assign(new Error('Each message needs a role.'), { status: 400 });
    const content = contentText(message.content).trim();
    if (!content) continue;
    if (message.role === 'system' || message.role === 'developer') { system.push(content); continue; }
    if (message.role !== 'user' && message.role !== 'assistant') { messages.push({ role: 'user', content }); continue; }
    messages.push({ role: message.role, content });
  }
  const prepared = mergeAdjacent(withSystem(messages, [contentText(body.system).trim(), ...system].filter(Boolean).join('\n\n')));
  if (!prepared.length) throw Object.assign(new Error('Send at least one non-empty message.'), { status: 400 });
  const options = generationOptions(finite(body.max_tokens) ?? 1024, finite(body.temperature), finite(body.top_k), finite(body.top_p), contextLength);
  options.maxTokens = Math.min(options.maxTokens ?? 1024, replyCeiling(prepared, contextLength));
  return { messages: prepared, options, stream: body.stream === true };
}

function openaiRequest(body, contextLength) {
  const messages = [], system = [];
  if (!Array.isArray(body.messages) || body.messages.length === 0) throw Object.assign(new Error('Send a non-empty messages array.'), { status: 400 });
  for (const message of body.messages) {
    if (!isRecord(message) || typeof message.role !== 'string') throw Object.assign(new Error('Each message needs a role.'), { status: 400 });
    const content = contentText(message.content).trim();
    if (!content) continue;
    if (message.role === 'system' || message.role === 'developer') { system.push(content); continue; }
    if (message.role === 'tool') { messages.push({ role: 'user', content: `[tool result] ${content}` }); continue; }
    if (message.role !== 'user' && message.role !== 'assistant') throw Object.assign(new Error('Unsupported message role.'), { status: 400 });
    messages.push({ role: message.role, content });
  }
  const prepared = mergeAdjacent(withSystem(messages, system.join('\n\n')));
  if (!prepared.length) throw Object.assign(new Error('Send at least one non-empty message.'), { status: 400 });
  const options = generationOptions(finite(body.max_completion_tokens) ?? finite(body.max_tokens) ?? 1024, finite(body.temperature), finite(body.top_k), finite(body.top_p), contextLength);
  options.maxTokens = Math.min(options.maxTokens ?? 1024, replyCeiling(prepared, contextLength));
  return { messages: prepared, options, stream: body.stream === true };
}

function modelId(name) {
  return String(name).toLowerCase().replace(/[^a-z0-9._-]+/g, '-').replace(/^-+|-+$/g, '') || 'webgpu-local';
}

/**
 * Creates the relay. `handle(req, res, url)` returns true when it owns the route.
 */
export function createCliBridge({ now = () => Date.now() } = {}) {
  const tabs = new Map();
  const parkers = new Map();
  const jobs = new Map();
  const queue = [];
  let active;

  const errorType = kind => (kind === 'anthropic' ? 'invalid_request_error' : 'invalid_request_error');
  const sendError = (kind, res, status, message) => {
    if (kind === 'anthropic') return sendJson(res, status, { type: 'error', error: { type: errorType(kind), message } });
    return sendJson(res, status, { error: { message, type: errorType(kind), param: null, code: null } });
  };

  const tabFor = key => {
    const tab = key ? tabs.get(key) : undefined;
    if (!tab || now() - tab.seen > TAB_TTL_MS) return undefined;
    return tab;
  };

  const requireTab = (kind, res, key) => {
    const tab = tabFor(key);
    if (!tab) { sendError(kind, res, 401, 'The API key is not connected. Run `npm start`, open the app, and check the key on the Developer API page.'); return undefined; }
    if (!tab.model) { sendError(kind, res, 503, 'No model is loaded in the WebGPU tab. Open a model there first.'); return undefined; }
    return tab;
  };

  const sseHeaders = res => {
    res.writeHead(200, {
      'Content-Type': 'text/event-stream; charset=utf-8',
      'Cache-Control': 'no-cache, no-transform',
      Connection: 'keep-alive',
      'X-Accel-Buffering': 'no',
    });
    res.write(': connected\n\n');
  };

  const event = (res, name, data) => {
    if (res.writableEnded) return;
    if (name) res.write(`event: ${name}\n`);
    res.write(`data: ${JSON.stringify(data)}\n\n`);
  };

  const chunkFrame = (job, delta) => {
    if (job.kind === 'anthropic') {
      if (delta) event(job.res, 'content_block_delta', { type: 'content_block_delta', index: 0, delta: { type: 'text_delta', text: delta } });
      return;
    }
    event(job.res, undefined, {
      id: `chatcmpl-${job.id}`,
      object: 'chat.completion.chunk',
      created: Math.floor(job.created / 1000),
      model: modelId(job.model),
      choices: [{ index: 0, delta: delta ? { content: delta } : { role: 'assistant' }, finish_reason: null }],
    });
  };

  const begin = job => {
    if (!job.stream) return;
    job.lastKeepalive = now();
    if (job.kind === 'anthropic') {
      event(job.res, 'message_start', {
        type: 'message_start',
        message: { id: `msg_${job.id}`, type: 'message', role: 'assistant', model: modelId(job.model), content: [], stop_reason: null, stop_sequence: null, usage: { input_tokens: 0, output_tokens: 0 } },
      });
      event(job.res, 'content_block_start', { type: 'content_block_start', index: 0, content_block: { type: 'text', text: '' } });
    } else {
      chunkFrame(job, '');
    }
  };

  const cleanup = job => {
    if (job.finished) return;
    job.finished = true;
    jobs.delete(job.id);
    const index = queue.indexOf(job.id);
    if (index >= 0) queue.splice(index, 1);
    if (active === job) active = undefined;
    dispatch();
  };

  const complete = job => {
    if (job.finished) return;
    if (!job.cancelled && !job.res.writableEnded) {
      if (job.stream) {
        if (job.kind === 'anthropic') {
          event(job.res, 'content_block_stop', { type: 'content_block_stop', index: 0 });
          event(job.res, 'message_delta', { type: 'message_delta', delta: { stop_reason: 'end_turn', stop_sequence: null }, usage: { input_tokens: job.promptTokens || 0, output_tokens: job.tokens || 0 } });
          event(job.res, 'message_stop', { type: 'message_stop' });
        } else {
          event(job.res, undefined, {
            id: `chatcmpl-${job.id}`, object: 'chat.completion.chunk', created: Math.floor(job.created / 1000), model: modelId(job.model),
            choices: [{ index: 0, delta: {}, finish_reason: 'stop' }],
            usage: { prompt_tokens: job.promptTokens || 0, completion_tokens: job.tokens || 0, total_tokens: (job.promptTokens || 0) + (job.tokens || 0) },
          });
          job.res.write('data: [DONE]\n\n');
        }
        job.res.end();
      } else if (job.kind === 'anthropic') {
        sendJson(job.res, 200, {
          id: `msg_${job.id}`, type: 'message', role: 'assistant', model: modelId(job.model),
          content: job.text ? [{ type: 'text', text: job.text }] : [],
          stop_reason: 'end_turn', stop_sequence: null,
          usage: { input_tokens: job.promptTokens || 0, output_tokens: job.tokens || 0 },
        });
      } else {
        sendJson(job.res, 200, {
          id: `chatcmpl-${job.id}`, object: 'chat.completion', created: Math.floor(job.created / 1000), model: modelId(job.model),
          choices: [{ index: 0, message: { role: 'assistant', content: job.text }, finish_reason: 'stop' }],
          usage: { prompt_tokens: job.promptTokens || 0, completion_tokens: job.tokens || 0, total_tokens: (job.promptTokens || 0) + (job.tokens || 0) },
        });
      }
    }
    cleanup(job);
  };

  const fail = (job, message) => {
    if (job.finished) return;
    trace('fail', job.id, message);
    if (!job.cancelled && !job.res.writableEnded) {
      if (job.stream) {
        if (job.kind === 'anthropic') event(job.res, 'error', { type: 'error', error: { type: 'api_error', message } });
        else event(job.res, undefined, { error: { message, type: 'api_error', param: null, code: null } });
        job.res.end();
      } else {
        sendError(job.kind, job.res, 502, message);
      }
    }
    cleanup(job);
  };

  function park(key, res) {
    const previous = parkers.get(key);
    if (previous) { clearTimeout(previous.timer); sendNoContent(previous.res); }
    const entry = { res, timer: setTimeout(() => {
      if (parkers.get(key) !== entry) return;
      parkers.delete(key);
      sendNoContent(res);
    }, POLL_WAIT_MS) };
    parkers.set(key, entry);
    res.on('close', () => {
      if (parkers.get(key) !== entry) return;
      clearTimeout(entry.timer);
      parkers.delete(key);
    });
  }

  function dispatch() {
    if (active || !queue.length) return;
    for (const id of [...queue]) {
      if (active) return;
      const job = jobs.get(id);
      if (!job) { queue.splice(queue.indexOf(id), 1); continue; }
      const parker = parkers.get(job.key);
      if (!parker) continue;
      const tab = tabFor(job.key);
      if (!tab) { queue.splice(queue.indexOf(id), 1); fail(job, 'The WebGPU tab is no longer connected.'); continue; }
      clearTimeout(parker.timer);
      parkers.delete(job.key);
      queue.splice(queue.indexOf(id), 1);
      active = job;
      job.started = now();
      job.lastEvent = now();
      tab.seen = job.lastEvent;
      if (!parker.res.writableEnded) sendJson(parker.res, 200, { id: job.id, messages: job.messages, options: job.options });
      return;
    }
  }

  const tabRecord = body => ({
    model: text(body.model), architecture: text(body.architecture), contextLength: finite(body.contextLength), seen: now(),
  });

  async function poll(req, res) {
    const key = requestKey(req);
    if (!key) return sendJson(res, 401, { error: 'Send the bearer key from the Developer API page.' });
    let body;
    try { body = await readJson(req); }
    catch (error) { return sendJson(res, error.status || 400, { error: error.message }); }
    tabs.set(key, tabRecord(body));
    park(key, res);
    dispatch();
  }

  async function hello(req, res) {
    const key = requestKey(req);
    if (!key) return sendJson(res, 401, { error: 'Send the bearer key from the Developer API page.' });
    let body;
    try { body = await readJson(req); }
    catch (error) { return sendJson(res, error.status || 400, { error: error.message }); }
    tabs.set(key, tabRecord(body));
    sendNoContent(res);
    dispatch();
  }

  async function createJob(kind, key, res, body) {
    const tab = requireTab(kind, res, key);
    if (!tab) return true;
    const request = kind === 'anthropic' ? anthropicRequest(body, tab.contextLength) : openaiRequest(body, tab.contextLength);
    if (queue.length >= QUEUE_LIMIT) { sendError(kind, res, 429, 'The runtime queue is full. Wait for the running generations to finish.'); return true; }
    const job = {
      id: randomUUID(), key, kind, res, stream: request.stream, messages: request.messages, options: request.options,
      model: tab.model, created: now(), lastEvent: now(), lastKeepalive: now(),
      text: '', tokens: 0, promptTokens: 0, cancelled: false, finished: false,
    };
    jobs.set(job.id, job);
    queue.push(job.id);
    trace('queued', job.id, kind, `${request.messages.length} messages`, `${request.messages.reduce((sum, message) => sum + message.content.length, 0)} chars`, job.stream ? 'stream' : 'json');
    res.on('close', () => {
      if (job.finished || job.cancelled) return;
      job.cancelled = true;
      job.closedAt = now();
      if (active !== job) { const index = queue.indexOf(job.id); if (index >= 0) queue.splice(index, 1); cleanup(job); }
    });
    if (request.stream) sseHeaders(res);
    begin(job);
    dispatch();
    return true;
  }

  async function jobEvent(req, res, id) {
    const job = jobs.get(id);
    if (!job) return sendJson(res, 410, { error: { message: 'This job is no longer active.' } });
    if (requestKey(req) !== job.key) return sendJson(res, 403, { error: { message: 'The job belongs to another key.' } });
    let body;
    try { body = await readJson(req); }
    catch (error) { return sendJson(res, error.status || 400, { error: error.message }); }
    if (job.finished) return sendJson(res, 410, { error: { message: 'This job already finished.' } });
    job.lastEvent = now();
    const tab = tabs.get(job.key);
    if (tab) tab.seen = job.lastEvent;

    if (body.type === 'chunk') {
      const delta = typeof body.delta === 'string' ? body.delta : '';
      if (typeof body.text === 'string') job.text = body.text;
      else job.text += delta;
      if (finite(body.tokens) !== undefined) job.tokens = Math.max(0, Math.floor(body.tokens));
      if (finite(body.promptTokens) !== undefined) job.promptTokens = Math.max(0, Math.floor(body.promptTokens));
      if (!job.cancelled && job.stream) chunkFrame(job, delta);
      return sendJson(res, 200, { ok: true, cancelled: job.cancelled });
    }
    if (body.type === 'done') {
      if (typeof body.text === 'string') job.text = body.text;
      if (finite(body.tokens) !== undefined) job.tokens = Math.max(0, Math.floor(body.tokens));
      sendJson(res, 200, { ok: true, cancelled: job.cancelled });
      complete(job);
      return true;
    }
    if (body.type === 'error') {
      sendJson(res, 200, { ok: true, cancelled: false });
      fail(job, text(body.message) || 'The browser runtime reported an error.');
      return true;
    }
    return sendJson(res, 400, { error: { message: 'Send type "chunk", "done", or "error".' } });
  }

  function models(res, key) {
    const tab = tabFor(key);
    if (!tab) return sendJson(res, 401, { error: { message: 'The API key is not connected. Open the app locally and check its key.', type: 'invalid_request_error' } });
    const data = tab.model ? [{ id: modelId(tab.model), object: 'model', created: Math.floor(now() / 1000), owned_by: 'webgpu-local' }] : [];
    sendJson(res, 200, { object: 'list', data });
  }

  async function countTokens(req, res) {
    const key = requestKey(req);
    const tab = requireTab('anthropic', res, key);
    if (!tab) return;
    let body;
    try { body = await readJson(req); }
    catch (error) { return sendJson(res, error.status || 400, { error: { message: error.message } }); }
    try {
      const request = anthropicRequest(body, tab.contextLength);
      const length = request.messages.reduce((sum, message) => sum + message.content.length, 0);
      sendJson(res, 200, { input_tokens: Math.ceil(length / 4) });
    } catch (error) {
      sendError('anthropic', res, error.status || 400, error.message);
    }
  }

  const sweep = setInterval(() => {
    const stamp = now();
    for (const job of [...jobs.values()]) {
      if (job.finished) continue;
      if (job.cancelled && stamp - (job.closedAt || stamp) > 30_000) { cleanup(job); continue; }
      if (active !== job) {
        const tab = tabs.get(job.key);
        if (!tab || stamp - tab.seen > TAB_TTL_MS) { fail(job, 'The WebGPU tab that owns this model is no longer connected.'); continue; }
      }
      if (stamp - job.lastEvent > IDLE_TIMEOUT_MS) { fail(job, 'The WebGPU tab stopped sending tokens.'); continue; }
      if (job.stream && !job.cancelled && stamp - job.lastKeepalive > KEEPALIVE_MS) {
        job.lastKeepalive = stamp;
        if (job.kind === 'anthropic') event(job.res, 'ping', { type: 'ping' });
        else if (!job.res.writableEnded) job.res.write(': keepalive\n\n');
      }
    }
    for (const [key, tab] of tabs) if (stamp - tab.seen > TAB_TTL_MS * 4) tabs.delete(key);
  }, SWEEP_MS);
  sweep.unref?.();

  function handle(req, res, url) {
    const path = url.pathname;
    if (path === '/health' && (req.method === 'GET' || req.method === 'HEAD')) {
      sendJson(res, 200, { ok: true, relay: 'webgpu', models: [...tabs.keys()].some(key => Boolean(tabFor(key)?.model)) });
      return true;
    }
    if (path === '/v1/models' && req.method === 'GET') { models(res, requestKey(req)); return true; }
    if (path === '/v1/messages/count_tokens' && req.method === 'POST') { void countTokens(req, res); return true; }
    if (path === '/v1/messages' && req.method === 'POST') { void (async () => {
      try {
        const body = await readJson(req);
        trace('POST /v1/messages', JSON.stringify({ roles: body.messages?.map(message => message?.role), blocks: body.messages?.map(message => Array.isArray(message?.content) ? message.content.map(part => part?.type) : 'text'), system: typeof body.system, stream: body.stream, max_tokens: body.max_tokens, tools: Array.isArray(body.tools) ? body.tools.length : 0 }));
        await createJob('anthropic', requestKey(req), res, body);
      } catch (error) { sendError('anthropic', res, error.status || 400, error.message); }
    })(); return true; }
    if (path === '/v1/chat/completions' && req.method === 'POST') { void (async () => {
      try {
        const body = await readJson(req);
        trace('POST /v1/chat/completions', JSON.stringify({ roles: body.messages?.map(message => message?.role), stream: body.stream, max_tokens: body.max_tokens, tools: Array.isArray(body.tools) ? body.tools.length : 0 }));
        await createJob('openai', requestKey(req), res, body);
      } catch (error) { sendError('openai', res, error.status || 400, error.message); }
    })(); return true; }
    if (path === '/bridge/poll' && req.method === 'POST') { void poll(req, res); return true; }
    if (path === '/bridge/hello' && req.method === 'POST') { void hello(req, res); return true; }
    const match = /^\/bridge\/jobs\/([0-9a-f-]{36})\/events$/.exec(path);
    if (match && req.method === 'POST') { void jobEvent(req, res, match[1]); return true; }
    return false;
  }

  return {
    handle,
    /** Test/diagnostics hook: how many browser tabs are attached. */
    tabCount() { return [...tabs.keys()].filter(key => Boolean(tabFor(key))).length; },
    dispose() { clearInterval(sweep); for (const parker of parkers.values()) { clearTimeout(parker.timer); sendNoContent(parker.res); } parkers.clear(); for (const job of [...jobs.values()]) { job.finished = true; job.res.destroy?.(); } jobs.clear(); queue.length = 0; active = undefined; },
  };
}
