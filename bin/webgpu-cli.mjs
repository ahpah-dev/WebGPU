#!/usr/bin/env node
// Prints or launches a coding CLI against the WebGPU loopback relay.
// The relay is part of `npm start`; this helper does not start servers or
// evaluate models.
import { spawn } from 'node:child_process';

const args = process.argv.slice(2);
const option = name => {
  const index = args.indexOf(name);
  if (index < 0) return undefined;
  const value = args[index + 1];
  if (!value || value.startsWith('--')) return undefined;
  args.splice(index, 2);
  return value;
};

const key = option('--key') || process.env.WEBGPU_API_KEY || '';
const base = (option('--url') || 'http://127.0.0.1:8787').replace(/\/+$/, '');
const separator = args.indexOf('--');
const command = separator >= 0 ? args[separator + 1] : undefined;
const commandArgs = separator >= 0 ? args.slice(separator + 2) : [];
const windows = process.platform === 'win32';

const say = line => process.stdout.write(`${line}\n`);
const fail = message => { process.stderr.write(`${message}\n`); process.exit(1); };

async function reachable() {
  try {
    const response = await fetch(`${base}/health`, { signal: AbortSignal.timeout(2500) });
    return response.ok;
  } catch {
    return false;
  }
}

if (!(await reachable())) {
  say('The WebGPU relay is not running.');
  say('');
  say('Start the app from the repository, then open it and load a model:');
  say('  npm start');
  say(`  ${base}`);
  say('');
  say('Open the Developer API page to copy the key, then run this command again.');
  process.exit(1);
}

if (!key) {
  say('Missing the API key from the WebGPU Developer API page.');
  say('');
  say('  npm run cli -- --key wgpu_sk_…');
  say('  npm run cli -- --key wgpu_sk_… -- claude');
  say('');
  say('The same key is shown in the app; it changes if you rotate it.');
  process.exit(1);
}

let model;
try {
  const response = await fetch(`${base}/v1/models`, { headers: { authorization: `Bearer ${key}` }, signal: AbortSignal.timeout(4000) });
  if (response.status === 401) fail('That key is not connected. Copy the current key from the Developer API page.');
  if (!response.ok) fail(`The relay answered ${response.status}.`);
  const list = await response.json();
  model = list.data?.[0]?.id;
} catch (error) {
  fail(`Could not read /v1/models: ${error.message}`);
}

const set = (name, value) => windows ? `$env:${name} = '${value}'` : `export ${name}='${value}'`;
say(`Relay   ${base}`);
say(`Model   ${model ?? 'none yet'}`);
if (!model) say('        Load a model in the WebGPU tab before sending a request.');
say('');
say('Claude Code (Anthropic Messages API):');
say(`  ${set('ANTHROPIC_BASE_URL', base)}`);
say(`  ${set('ANTHROPIC_AUTH_TOKEN', key)}`);
say('  claude');
say('');
say('Any OpenAI-compatible CLI:');
say(`  ${set('OPENAI_BASE_URL', `${base}/v1`)}`);
say(`  ${set('OPENAI_API_KEY', key)}`);
say('');
say(`One step: npm run cli -- --key ${key} -- claude`);
say('The relay queues one request at a time. Tool calling is not implemented, so a CLI');
say('that depends on tool use will answer in text but cannot edit files by itself.');

if (!command) {
  process.exitCode = 0;
} else {
  const env = {
    ...process.env,
    ANTHROPIC_BASE_URL: base,
    ANTHROPIC_AUTH_TOKEN: key,
    OPENAI_BASE_URL: `${base}/v1`,
    OPENAI_API_KEY: key,
  };
  const child = spawn(command, commandArgs, { stdio: 'inherit', env, shell: windows });
  child.on('error', error => fail(`Could not start ${command}: ${error.message}`));
  child.on('exit', (code, signal) => process.exit(signal ? 1 : code ?? 0));
}
