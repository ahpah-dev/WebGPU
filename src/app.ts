import './style.css';
import { WebGPU, type Message, type ModelInfo, type Progress, type SourceInput } from './sdk.ts';
import { createOllamaPicker } from './ollama-picker.ts';
import { createLibraryPicker } from './local-library.ts';
import { maskApiKey, readApiKey, rotateApiKey } from './api-key.ts';
import { createBridge } from './bridge.ts';
import { attentionMemory, suggestBudget } from './core/context.ts';

const icons: Record<string, string> = {
  chat: '<path d="M21 11a8 8 0 0 1-8 8H5l-3 3V11a9 9 0 0 1 18-2"/><path d="M7 10h8M7 14h5"/>',
  chip: '<rect x="6" y="6" width="12" height="12" rx="3"/><path d="M9 1v5m6-5v5M9 18v5m6-5v5M1 9h5m-5 6h5m12-6h5m-5 6h5"/><rect x="9" y="9" width="6" height="6" rx="1"/>',
  box: '<path d="m12 3 9 5v9l-9 5-9-5V8Z"/><path d="m3 8 9 5 9-5M12 13v9M7 5.8l9 5"/>',
  code: '<path d="m8 6-6 6 6 6m8-12 6 6-6 6M14 3l-4 18"/>',
  plus: '<path d="M12 5v14M5 12h14"/>',
  arrow: '<path d="M12 19V5m-6 6 6-6 6 6"/>',
  upload: '<path d="M4 15v5h16v-5M12 16V3m-5 5 5-5 5 5"/>',
  link: '<path d="M14 3h7v7M21 3 10 14M10 3H4v17h17v-6"/>',
  shield: '<path d="m12 2 8 4v6c0 5-8 10-8 10S4 17 4 12V6Z"/><path d="m8 12 3 3 5-6"/>',
  clock: '<circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2"/>',
  stop: '<rect x="6" y="6" width="12" height="12" rx="2"/>',
  copy: '<rect x="8" y="8" width="13" height="13" rx="2"/><path d="M16 8V3H3v13h5"/>',
  spark: '<path d="m12 2 3 7 7 3-7 3-3 7-3-7-7-3 7-3Z"/>',
  disk: '<rect x="3" y="5" width="18" height="14" rx="3"/><path d="M3 13h18M7 16h1m3 0h1"/>',
};
const icon = (name: string, cls = '') => `<svg class="icon ${cls}" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${icons[name] ?? icons.box}</svg>`;
/** Mark: the same crisp W as the site icon, drawn as square-joined strokes. */
const brandmark = '<svg viewBox="0 0 28 24" aria-hidden="true"><path d="M4 5l5 14 5-11 5 11 5-14" fill="none" stroke="currentColor" stroke-width="2.6" stroke-linecap="square" stroke-linejoin="miter"/></svg>';
/** Hero illustration: a chip package, pin rows and a pixel die, all on one grid. */
function pixelDie(): string {
  const pins: string[] = [];
  for (let i = 0; i < 8; i++) pins.push(`<rect x="${44 + i * 20}" y="12" width="7" height="12"/>`, `<rect x="${44 + i * 20}" y="166" width="7" height="12"/>`);
  for (let i = 0; i < 7; i++) pins.push(`<rect x="12" y="${34 + i * 18}" width="12" height="7"/>`, `<rect x="216" y="${34 + i * 18}" width="12" height="7"/>`);
  const cells: string[] = [], stripe: string[] = [];
  for (let row = 0; row < 6; row++) for (let column = 0; column < 8; column++) {
    const x = 64 + column * 14, y = 56 + row * 14;
    cells.push(`<rect x="${x}" y="${y}" width="11" height="11" fill="#f2f2f2"/>`);
    if (column + row === 5) stripe.push(`<rect x="${x}" y="${y}" width="11" height="11" fill="#0d0d0d"/>`);
  }
  return `<svg class="pixel-art" viewBox="0 0 240 190" shape-rendering="crispEdges" aria-hidden="true">
    <g fill="#0d0d0d">${pins.join('')}</g>
    <rect x="34" y="26" width="172" height="138" fill="#ffffff" stroke="#0d0d0d" stroke-width="2"/>
    <rect x="56" y="48" width="128" height="94" fill="#fbfbfb" stroke="#dcdcdc" stroke-width="2"/>
    <g>${cells.join('')}</g>
    <g>${stripe.join('')}</g>
    <g fill="#0d0d0d"><rect x="34" y="26" width="6" height="6"/><rect x="200" y="26" width="6" height="6"/><rect x="34" y="158" width="6" height="6"/><rect x="200" y="158" width="6" height="6"/></g>
  </svg>`;
}
const $ = <T extends HTMLElement = HTMLElement>(id: string) => document.getElementById(id) as T;
const gib = (n: number) => `${(n / 1024 ** 3).toFixed(2)} GiB`;
let runtime = new WebGPU(), info: ModelInfo | undefined, busy = false, source: SourceInput | undefined;
let messages: Message[] = [];
let active = 'playground';
let sharedSource: SourceInput | undefined;

$('app').innerHTML = `
<aside class="sidebar">
  <a class="brand" aria-label="WebGPU home" href="#playground"><span class="brandmark">${brandmark}</span><span>WebGPU<span class="brand-dot">.</span></span></a>
  <span class="edition">BROWSER INFERENCE / 0.7</span>
  <button id="new-chat" aria-label="New conversation" class="new-chat">${icon('plus')}<span>New conversation</span><span class="key">↵</span></button>
  <div class="nav-label">WORKSPACE</div>
  <nav aria-label="Main navigation">
    <button data-page="playground" aria-label="Playground" class="nav-item active">${icon('chat')}<span>Playground</span><span class="nav-dot"></span></button>
    <button data-page="runtime" aria-label="Runtime" class="nav-item">${icon('chip')}<span>Runtime</span></button>
    <button data-page="models" aria-label="Model library" class="nav-item">${icon('box')}<span>Model library</span></button>
    <button data-page="api" aria-label="Developer API" class="nav-item">${icon('code')}<span>Developer API</span></button>
  </nav>
  <div class="sidebar-note"><span class="small-chip">${icon('shield')}</span><h3>Local by design.</h3><p>Your model. Your hardware.<br>Your conversations stay here.</p><span class="line-decoration"></span></div>
  <div class="sidebar-bottom"><span class="status-dot" id="support-dot"></span><div><strong id="support-title">Checking WebGPU</strong><small id="adapter-name">Browser compute runtime</small></div></div>
</aside>
<div class="shell">
  <header><div class="breadcrumb">Workspace <span>/</span> <strong id="page-label">Playground</strong></div><div class="header-right"><button class="key-chip" data-page="api" id="key-chip" aria-label="API access and keys">${icon('code')}<span>API access</span><b id="key-chip-state">standby</b></button><span class="local-badge"><span></span> Runs on your device</span><a href="https://github.com/ahpah-dev/WebGPU" target="_blank" rel="noreferrer" class="github-link">GitHub ${icon('link')}</a></div></header>
  <main>
    <section class="page active" id="page-playground">
      <div class="page-heading"><div><div class="eyebrow"><span></span> A LITTLE LESS HARDWARE. A LOT MORE POSSIBILITY.</div><h1>Big models.<br><span>Small footprint.</span></h1><p class="intro">Local AI, right in your browser. Stream the weights.<br>Keep the intelligence on your machine.</p></div><div class="hero-art" aria-hidden="true">${pixelDie()}<div class="art-tag"><i></i> Browser native · paged compute</div></div></div>
      <div class="stats-row"><div class="stat"><span class="stat-icon">${icon('box')}</span><div><label>Model parameters</label><strong id="stat-params">— <small>no model loaded</small></strong></div></div><div class="stat"><span class="stat-icon">${icon('chip')}</span><div><label>GPU allocation budget</label><strong id="stat-budget">2.0 <small>GiB</small></strong></div><span class="stat-tag">YOU SET THE LIMIT</span></div><div class="stat"><span class="stat-icon">${icon('shield')}</span><div><label>Inference location</label><strong>In your browser <span class="tiny-dot"></span></strong></div></div></div>
      <div class="workspace-grid">
        <div class="chat-panel panel"><div class="panel-top"><div class="panel-title">${icon('chat')} Conversation <span class="subtle-badge" id="chat-model">No model</span></div><button class="text-button" id="export-chat">Export ${icon('link')}</button></div>
          <div id="conversation" class="conversation" aria-live="polite"><div class="empty-chat" id="empty-chat"><div class="empty-symbol">${icon('spark')}</div><h2>Any model you already have.</h2><p>Open the folder that holds your GGUF files, or pick one file.<br>Weights stay on your disk. Inference stays in this tab.</p><div class="model-open-actions"><button class="primary" data-open="library">${icon('box')} Open local models</button><button class="secondary" data-open="file">${icon('upload')} Choose a GGUF file</button></div><p class="empty-note">Llama 3.x · Qwen2 / Qwen3 · SmolLM · Mistral · Gemma 3 · Qwen3.8<span>No server. No upload. No API bills.</span></p><div class="starter-prompts"><button data-prompt="Explain gravity in one sentence.">Explain a concept <span>↗</span></button><button data-prompt="Write a short Python function that checks whether a number is prime.">Write some code <span>↗</span></button></div></div></div>
          <div class="generation-progress" id="generation-progress" hidden><div><span id="progress-text">Preparing…</span><span id="progress-percent">0%</span></div><div class="progress-track"><i id="progress-bar"></i></div></div>
          <form id="chat-form" class="composer"><textarea id="prompt" rows="2" placeholder="Ask something. Keep it local." aria-label="Message" maxlength="1048576"></textarea><div class="composer-bottom"><span>${icon('shield')} Private · no cloud inference</span><button class="send-button" id="send" type="submit" aria-label="Send message" disabled>${icon('arrow')}</button><button class="send-button stop" id="stop" type="button" aria-label="Stop generation" hidden>${icon('stop')}</button></div></form><div class="chat-foot"><span>Enter to send · Shift + Enter for a new line</span><span id="token-stats">Ready when you are</span></div>
        </div>
        <aside class="settings-panel panel"><div class="panel-top"><div class="panel-title">${icon('chip')} Run configuration</div></div><div class="settings-body">
          <label class="field-label">LOCAL MODEL <span>GGUF · NO SERVER</span></label><button data-open="library" class="secondary full ollama-browse">${icon('box')} Open local models</button><button data-open="file" class="model-picker">${icon('box')}<span id="model-filename">Select a model file<small>Llama · Qwen · SmolLM · Gemma</small></span>${icon('plus')}</button><button id="open-ollama" class="link-button">Use the Ollama library instead →</button><button id="shared-model" class="shared-model" hidden>Use locally served model →</button>
          <div class="field-divider"></div><label class="field-label" for="performance">PERFORMANCE</label><select id="performance"><option value="lite">Lite · 512 MiB preset</option><option value="low-memory" selected>Memory saver · 2 GiB preset</option><option value="fast">Faster · 6 GiB preset</option></select><p id="performance-help" class="field-help">Memory saver keeps GPU use low. Faster keeps more weights resident and reads larger chunks. Apply & reload after changing modes.</p><label class="field-label" for="budget">GPU MEMORY BUDGET <span id="budget-display">2 GiB</span></label><input id="budget" type="range" min="512" max="8192" step="256" value="2048"><div class="range-labels"><span>512 MiB</span><span>8 GiB</span></div><p class="field-help">Weights that don't fit stream from disk. Small models get a smaller budget automatically so the whole model stays resident.</p>
          <label class="field-label" for="context">CONTEXT LENGTH</label><select id="context"><option value="256">256 tokens · minimum memory</option><option value="512">512 tokens</option><option value="1024">1,024 tokens</option><option value="2048">2,048 tokens</option><option value="4096" selected>4,096 tokens · balanced</option><option value="8192">8,192 tokens</option><option value="16384">16,384 tokens</option><option value="32768">32,768 tokens</option></select><p id="context-help" class="field-help">Context includes your prompt, chat history and reply. Longer contexts need more GPU memory. Apply & reload to change context.</p>
          <div class="field-pair"><div><label class="field-label" for="temperature">TEMPERATURE</label><input id="temperature" type="number" min="0" max="2" step="0.1" value="0.7"></div><div><label class="field-label" for="max-tokens">MAX OUTPUT</label><select id="max-tokens"><option value="16">16 tokens</option><option value="32">32 tokens</option><option value="64">64 tokens</option><option value="128">128 tokens</option><option value="256">256 tokens</option><option value="512">512 tokens</option><option value="1024">1,024 tokens</option><option value="2048" selected>2,048 tokens</option><option value="4096">4,096 tokens</option><option value="8192">8,192 tokens</option><option value="16384">16,384 tokens</option><option value="32768">32,768 tokens</option></select></div></div><p id="output-help" class="field-help">Reply length is capped by the space left in the loaded context. You can change output length without reloading.</p>
          <button id="reload" class="secondary full" disabled>Apply & reload model</button><div class="memory-note">${icon('disk')}<p><strong>Big models, streamed in pieces.</strong>A 27B model can exceed your GPU's memory. Paging makes it fit by trading speed for memory.</p></div>
        </div></aside>
      </div>
      <div class="bottom-note"><span>BUILT FOR YOUR MACHINE</span><p>Real WebGPU kernels. Bounded weight pages. No inference server.</p><button class="text-button" data-page="runtime">See how it works →</button></div>
    </section>
    <section class="page" id="page-runtime"><div class="section-heading"><span class="eyebrow">UNDER THE HOOD</span><h1>The memory is limited.<br><span>The model doesn't have to fit.</span></h1><p class="intro">Only a small part of the model needs to be on the GPU at once.</p></div><div class="pipeline-panel panel"><div class="pipeline-node">${icon('disk')}<b>Local GGUF</b><span id="runtime-file">Weights on your disk</span></div><span class="pipeline-arrow">→</span><div class="pipeline-node">${icon('box')}<b id="runtime-pages">16 MiB pages</b><span>Bounded file reads</span></div><span class="pipeline-arrow">→</span><div class="pipeline-node highlighted">${icon('chip')}<b>WebGPU kernels</b><span>Quantized computation</span></div><span class="pipeline-arrow">→</span><div class="pipeline-node">${icon('chat')}<b>Live tokens</b><span>Generated in a worker</span></div></div>
      <div class="telemetry-grid"><div class="panel metric"><label>Tracked GPU allocations</label><strong id="gpu-allocated">—</strong><p>Weights, KV/recurrent state, and work buffers</p></div><div class="panel metric"><label>Peak tracked allocation</label><strong id="gpu-peak">—</strong><p>Browser and driver overhead are additional</p></div><div class="panel metric"><label>Weight bytes read</label><strong id="bytes-read">—</strong><p>Cumulative; repeated paging increases this</p></div><div class="panel metric"><label>Resident cache hits</label><strong id="cache-hits">—</strong><p>Weight pages reused without disk reads</p></div></div>
      <div class="panel layer-panel"><div class="panel-top"><div class="panel-title">Transformer layers</div><span class="subtle-badge" id="layer-count">Load a model to inspect</span></div><div id="layers" class="layers"></div><p class="field-help"><span id="runtime-batch">Prompt batches: up to 32 tokens.</span> Each weight page is reused across the batch. A fixed resident cache keeps frequently traversed weight pages on the GPU. Other pages reuse one streaming buffer.</p></div><div class="honest-note"><b>What this mechanism does</b><p>It reduces the amount of model weight memory allocated at once. It does not reduce the work required to run 27 billion parameters. Speed depends heavily on disk throughput, GPU performance, context length, and the chosen memory budget.</p></div>
    </section>
    <section class="page" id="page-models"><div class="section-heading"><span class="eyebrow">BRING YOUR OWN WEIGHTS</span><h1>A local file.<br><span>A whole lot of intelligence.</span></h1><p class="intro">Llama 3.x, Qwen2/3, SmolLM, Mistral, Gemma 3 and Qwen3.8. Text inference, entirely in your browser.</p></div><div class="panel ollama-library-banner"><div><h2>Your own model folder.</h2><p>Choose the folder that holds your GGUF files or your Ollama library. The browser remembers it, so nothing has to be installed or served.</p></div><button class="primary" data-open="library">Choose a folder →</button></div><div class="model-guide-grid"><div class="panel model-card"><span class="model-badge">RECOMMENDED · 1–3B</span><div class="model-card-symbol">L<span>3.2</span></div><h2>Llama 3.2</h2><p>Small Llama models. A whole chat model in about a gigabyte, fast on a laptop and fully resident in GPU memory.</p><div class="model-facts"><span>Q4_K_M file <b>0.8–2 GB</b></span><span>Architecture <b>llama</b></span><span>Execution <b>Browser WebGPU</b></span></div><button class="primary full" data-open="library">${icon('upload')} Open from a folder</button><p class="field-help"><a href="https://ollama.com/library/llama3.2:1b" target="_blank" rel="noreferrer">Get Llama 3.2 weights ↗</a></p></div><div class="panel model-card"><span class="model-badge">SMALL · 0.5–14B</span><div class="model-card-symbol">Q<span>2.5</span></div><h2>Qwen2.5 / Qwen3</h2><p>Strong small models from 0.5B to 14B. Q/K normalization and attention bias are detected from the file.</p><div class="model-facts"><span>Q4_K_M file <b>0.4–9 GB</b></span><span>Architectures <b>qwen2 · qwen3</b></span><span>Execution <b>Browser WebGPU</b></span></div><button class="primary full" data-open="library">${icon('upload')} Open from a folder</button><p class="field-help"><a href="https://ollama.com/library/qwen2.5:0.5b" target="_blank" rel="noreferrer">Get Qwen2.5 weights ↗</a></p></div><div class="panel model-card"><span class="model-badge">TINY · 135M–1.7B</span><div class="model-card-symbol">S<span>2</span></div><h2>SmolLM2</h2><p>About 400 MB for the 360M model. The lowest memory footprint this runtime can run, and a good first check.</p><div class="model-facts"><span>Q4_K_M file <b>0.1–1 GB</b></span><span>Architecture <b>llama</b></span><span>Execution <b>Browser WebGPU</b></span></div><button class="primary full" data-open="library">${icon('upload')} Open from a folder</button><p class="field-help"><a href="https://ollama.com/library/smollm2:360m" target="_blank" rel="noreferrer">Get SmolLM2 weights ↗</a></p></div><div class="panel model-card"><span class="model-badge">LARGE · 27B PAGED</span><div class="model-card-symbol">G<span>3</span></div><h2>Gemma 3 / Qwen3.8 27B</h2><p>27B weights streamed through a bounded GPU budget instead of being loaded whole. Slow, but they fit where they otherwise would not.</p><div class="model-facts"><span>Q4_K_M file <b>~16.6–16.8 GB</b></span><span>Architectures <b>gemma3 · qwen35</b></span><span>Execution <b>Browser WebGPU</b></span></div><button class="primary full" data-open="library">${icon('upload')} Open from a folder</button></div><div class="panel guide-card"><h2>Get your model ready</h2><ol><li><strong>Download any supported GGUF</strong><p>From Ollama, Hugging Face or an existing file on disk. You need disk space, and the model licence applies.</p><code>ollama pull llama3.2:1b</code><p>Also <code class="inline-code">qwen2.5:0.5b</code>, <code class="inline-code">smollm2:360m</code> and <code class="inline-code">gemma3:27b</code>.</p></li><li><strong>Point the browser at the folder</strong><p>Pick <code class="inline-code">~/.ollama/models</code> or any folder of GGUFs with <b>Open local models</b>. The folder handle is remembered between visits.</p></li><li><strong>Start small</strong><p>A 0.5–1B model stays fully resident in a small GPU budget. Large models page weights from disk and need patience.</p></li></ol></div></div></section>
    <section class="page" id="page-api"><div class="section-heading"><span class="eyebrow">MADE TO BUILD WITH</span><h1>One key.<br><span>Your own inference endpoint.</span></h1><p class="intro">Let another page, tab or an embedding site drive the model that is loaded in this tab. The key is the only thing you copy.</p></div><div class="access-grid"><div class="panel access-card"><div class="panel-top"><div class="panel-title">${icon('shield')} Access key</div><span class="subtle-badge" id="key-status">Active</span></div><div class="access-body"><label class="field-label" for="api-key-value">BEARER KEY <span id="key-visibility">HIDDEN</span></label><div class="key-row"><code class="key-value" id="api-key-value">…</code></div><div class="key-actions"><button class="secondary" id="copy-key">${icon('copy')} Copy key</button><button class="secondary" id="reveal-key">Reveal</button><button class="text-button" id="rotate-key">Generate new key</button></div><p class="field-help">Stored in this browser only. The key authorises the bridge in this tab. Rotating it invalidates the previous key immediately.</p><div class="key-meta"><span>Channel <b>webgpu-runtime</b></span><span>Origin <b id="api-origin">…</b></span><span><b id="bridge-clients">0</b> clients</span><span>Model <b id="bridge-model">none</b></span></div></div></div><div class="panel access-card"><div class="panel-top"><div class="panel-title">${icon('code')} Stream a completion</div><button class="text-button" id="copy-client">${icon('copy')} Copy</button></div><div class="access-body"><p class="access-intro">Paste this into any page on the same origin. Tokens stream here as they are generated.</p><pre class="access-code"><code id="api-client"></code></pre></div></div></div><div class="api-grid"><div class="panel api-card"><code>method: 'chat.completions'</code><p>Send <b>messages</b> with optional <b>maxTokens</b>, <b>temperature</b>, <b>topK</b> and <b>topP</b>. The reply streams as <b>chunk</b> events and ends with <b>done</b>.</p></div><div class="panel api-card"><code>method: 'ping'</code><p>Checks the key and reports the loaded model, architecture and context length without generating anything.</p></div><div class="panel api-card"><code>event.type: 'error'</code><p>Returned for a wrong key, no loaded model, a busy runtime or invalid messages. One generation runs at a time.</p></div></div><div class="panel embed-panel"><div class="panel-top"><div class="panel-title">${icon('link')} From an embedding page</div><button class="text-button" id="copy-embed">${icon('copy')} Copy</button></div><p class="access-intro">Embed this app in an iframe and the parent page can post the same request with the same key. Events arrive through <b>postMessage</b>.</p><pre class="access-code"><code id="api-embed"></code></pre></div><p class="api-note">The bridge lives in this page, not in a server: requests are answered only while this tab is open with a model loaded. Inference, tokenisation and the key stay on this device. A browser cannot listen on an HTTP port, so other tools reach the runtime through this channel rather than a base URL.</p><div class="panel api-panel"><div class="panel-top"><div class="panel-title">${icon('box')} Browser SDK</div><button class="text-button" id="copy-api">${icon('copy')} Copy</button></div><pre><code id="api-example"></code></pre></div></section>
    <div id="notice" class="notice" role="status" hidden></div>
  </main><footer><span>WebGPU <b> / </b> Browser-native local AI</span><span>Open source. No API keys. No inference fees.</span></footer>
</div><input id="model-file" type="file" accept=".gguf,application/octet-stream" hidden>
`;
const sdkExample = `import { WebGPU } from './sdk/sdk.js';

// A File from <input type="file"> — never uploaded.
const model = new WebGPU();

await model.load(file, {
  contextLength: 4096,
  performance: 'lite', // 512 MiB preset; small models stay fully resident.
});

for await (const chunk of model.generate(
  [{ role: 'user', content: 'Explain gravity in one sentence.' }],
  { maxTokens: 64, temperature: 0.7 },
)) {
  output.textContent += chunk.delta;
}

model.dispose();`;
function clientExample(key: string) {
  return `// Any page on this origin can drive the model loaded in this tab.
const channel = new BroadcastChannel('webgpu-runtime');
const id = crypto.randomUUID();
let text = '';

channel.onmessage = ({ data }) => {
  if (data?.id !== id) return;
  const { event } = data;
  if (event.type === 'chunk') text += event.delta;
  if (event.type === 'done') { console.log(text); channel.close(); }
  if (event.type === 'error') { console.error(event.message); channel.close(); }
};

channel.postMessage({
  type: 'webgpu.request',
  id,
  apiKey: '${key}',
  method: 'chat.completions',
  messages: [{ role: 'user', content: 'Name the capital of France.' }],
  maxTokens: 64,
  temperature: 0.7,
});`;
}
function embedExample(key: string) {
  return `// Parent page: embed this app in an <iframe>, then post the same request.
const frame = document.querySelector('iframe');
const id = crypto.randomUUID();
let text = '';

window.addEventListener('message', ({ data }) => {
  if (data?.id !== id) return;
  const { event } = data;
  if (event.type === 'chunk') text += event.delta;
  if (event.type === 'done') console.log(text);
  if (event.type === 'error') console.error(event.message);
});

frame.contentWindow.postMessage({
  type: 'webgpu.request',
  id,
  apiKey: '${key}',
  method: 'chat.completions',
  messages: [{ role: 'user', content: 'Name the capital of France.' }],
}, '*');`;
}
type Preset = 'lite' | 'low-memory' | 'fast';
const BATCHES: Record<Preset, number> = { lite: 8, 'low-memory': 32, fast: 128 };
const PRESET_BUDGET: Record<Preset, number> = { lite: 512, 'low-memory': 2048, fast: 6144 };
const PRESET_HELP: Record<Preset, string> = {
  lite: 'Lite uses a 512 MiB GPU budget, 16 MiB pages and batches of 8 prompt tokens. Small models stay fully resident; large models page more and run slower.',
  'low-memory': 'Memory saver uses a 2 GiB GPU preset and 16 MiB pages. Apply & reload to activate.',
  fast: 'Faster uses a 6 GiB GPU budget, up to 64 MiB pages and batches of 128 prompt tokens. Adjust the budget to fit your GPU. The first prompt still reads the model; later tokens can reuse more weights.',
};
let apiKey = readApiKey();
let revealedKey = false;
let budgetTouched = false;
const bridgeHost = () => ({
  info: () => info ? { name: info.config.name, architecture: info.config.architecture, contextLength: info.contextLength, ready: true } : undefined,
  generate: (messages: Message[], options: Parameters<WebGPU['generate']>[1]) => runtime.generate(messages, options),
  busy: () => busy,
});
let bridge = createBridge(bridgeHost(), apiKey);
function restartBridge() { bridge.dispose(); bridge = createBridge(bridgeHost(), apiKey); }
function notify(message: string, error = false) { $('notice').textContent = message; $('notice').hidden = false; $('notice').classList.toggle('error', error); }
function clearNotice() { $('notice').hidden = true; }
async function copyText(text: string, message: string) {
  try { await navigator.clipboard.writeText(text); notify(message); return; } catch { /* Fall back to a selection copy. */ }
  const area = document.createElement('textarea'); area.value = text; area.setAttribute('readonly', ''); area.style.cssText = 'position:fixed;top:-1000px;opacity:0';
  document.body.append(area); area.select();
  try { document.execCommand('copy'); notify(message); } catch { notify('Copying is blocked in this browser. Select the text and copy it manually.', true); } finally { area.remove(); }
}
function updateApiPanel() {
  $('api-key-value').textContent = revealedKey ? apiKey : maskApiKey(apiKey);
  $('key-visibility').textContent = revealedKey ? 'VISIBLE' : 'HIDDEN';
  $('reveal-key').textContent = revealedKey ? 'Hide' : 'Reveal';
  $('key-status').textContent = info ? 'Model loaded' : 'Waiting for a model';
  $('api-origin').textContent = location.origin;
  $('api-client').textContent = clientExample(apiKey);
  $('api-embed').textContent = embedExample(apiKey);
  $('api-example').textContent = sdkExample;
  $('bridge-clients').textContent = String(bridge.clientCount());
  $('bridge-model').textContent = info ? `${info.config.name} · ${info.config.architecture}` : 'none';
  $('key-chip-state').textContent = info ? 'live' : 'standby';
  $('key-chip-state').dataset.state = info ? 'live' : 'standby';
}
function switchPage(page: string) {
  if (!['playground', 'runtime', 'models', 'api'].includes(page)) return;
  active = page; document.querySelectorAll('.page').forEach(p => p.classList.toggle('active', p.id === `page-${page}`));
  document.querySelectorAll('.nav-item').forEach(p => p.classList.toggle('active', (p as HTMLElement).dataset.page === page));
  $('page-label').textContent = ({ playground: 'Playground', runtime: 'Runtime', models: 'Model library', api: 'Developer API' })[page]!;
  if (page === 'api') updateApiPanel();
}
document.querySelectorAll<HTMLElement>('[data-page]').forEach(el => el.addEventListener('click', () => switchPage(el.dataset.page!)));
document.querySelector('.brand')!.addEventListener('click', () => switchPage('playground'));
function setBusy(value: boolean) {
  busy = value; $<HTMLButtonElement>('send').disabled = value || !info;
  $('send').hidden = value; $('stop').hidden = !value;
  document.querySelectorAll<HTMLButtonElement>('[data-open], #shared-model, #new-chat, #open-ollama').forEach(el => { el.disabled = value; });
  for (const id of ['performance', 'budget', 'context', 'max-tokens']) $<HTMLInputElement>(id).disabled = value;
  $<HTMLButtonElement>('reload').disabled = value || !source;
  $('generation-progress').hidden = !value;
}
function updateStats(stats: NonNullable<Progress['stats']>) {
  $('gpu-allocated').textContent = gib(stats.gpuBytes); $('gpu-peak').textContent = gib(stats.peakGpuBytes);
  $('bytes-read').textContent = gib(stats.readBytes); $('cache-hits').textContent = stats.cacheHits.toLocaleString();
}
function onProgress(p: Progress) {
  $('progress-text').textContent = p.detail; $('progress-percent').textContent = `${Math.round(p.fraction * 100)}%`; $('progress-bar').style.width = `${p.fraction * 100}%`;
  if (p.stats) updateStats(p.stats);
  if (p.layer !== undefined) $('layers').querySelectorAll<HTMLElement>('span').forEach((el, i) => el.classList.toggle('current', i === p.layer));
}
runtime.onProgress = onProgress;
function sizeLabel(mib: number) { return mib >= 1024 ? `${(mib / 1024).toFixed(1)} GiB` : `${Math.round(mib)} MiB`; }
async function loadModel(selected: SourceInput, preserveConversation = false) {
  if (busy) return; clearNotice(); source = selected; info = undefined; switchPage('playground'); setBusy(true);
  runtime.dispose(); runtime = new WebGPU(); runtime.onProgress = onProgress;
  try {
    const mode = $<HTMLSelectElement>('performance').value as Preset;
    // Small models get a budget that holds the whole file, so weights are read once.
    if (!budgetTouched && Number.isSafeInteger(selected.size) && selected.size > 0) {
      const suggested = suggestBudget(selected.size, mode);
      $<HTMLInputElement>('budget').value = String(suggested);
      $('budget-display').textContent = sizeLabel(suggested);
      if (!info) $('stat-budget').textContent = `${(suggested / 1024).toFixed(1)} GiB`;
    }
    const budget = Number($<HTMLInputElement>('budget').value);
    const loaded = await runtime.load(selected, { gpuBudgetMiB: budget, contextLength: Number($<HTMLSelectElement>('context').value), batchSize: BATCHES[mode], performance: mode });
    info = loaded;
    if (!preserveConversation || !messages.length) {
      messages = []; $('conversation').replaceChildren();
      const ready = document.createElement('div'); ready.className = 'ready-message'; ready.innerHTML = `${icon('spark')}<h2>Model ready.</h2><p>Weights stay on your disk. Messages stay in this tab.</p>`; $('conversation').append(ready);
    }
    $('stat-params').textContent = `${(loaded.config.parameters / 1e9).toFixed(1)}B`;
    $('chat-model').textContent = loaded.config.name;
    $('model-filename').textContent = selected instanceof File ? selected.name : selected.name || 'Local GGUF';
    $('stat-budget').textContent = `${(loaded.gpuBudgetMiB / 1024).toFixed(1)} GiB`;
    const recurrent = loaded.config.recurrent, recurrentCount = recurrent?.layers.filter(Boolean).length ?? 0;
    $('runtime-pages').textContent = `${loaded.stats.pageBytes / 1024 ** 2} MiB pages`; $('runtime-batch').textContent = `Prompt batches: up to ${loaded.batchSize} tokens.`;
    $('runtime-file').textContent = gib(loaded.fileBytes);
    $('layer-count').textContent = `${loaded.config.layers} layers${recurrent ? ` · ${recurrentCount} recurrent / ${loaded.config.layers - recurrentCount} full attention` : ` · ${loaded.config.qkNorm ? 'Q/K norm' : 'GQA'} · ${loaded.config.ffnGate ? 'SwiGLU' : 'MLP'}`}`;
    $('layers').replaceChildren(...Array.from({ length: loaded.config.layers }, (_, i) => { const span = document.createElement('span'); span.textContent = String(i + 1).padStart(2, '0'); span.title = `Layer ${i + 1}: ${loaded.config.recurrent?.layers[i] ? 'recurrent attention' : 'attention'}`; return span; }));
    updateStats(loaded.stats); $('adapter-name').textContent = loaded.adapter; updateContextHelp(); updateApiPanel();
    const resident = loaded.fileBytes + 256 * 1024 ** 2 <= loaded.gpuBudgetMiB * 1024 ** 2;
    notify(`${loaded.config.name} (${loaded.config.architecture}) is ready. ${resident ? 'The whole model fits in the GPU budget, so the weights can stay resident after the first pass.' : `${gib(loaded.fileBytes)} of weights will be paged within a ${(loaded.gpuBudgetMiB / 1024).toFixed(1)} GiB GPU budget.`}`);
    switchPage('playground'); $<HTMLTextAreaElement>('prompt').focus();
  } catch (e) { notify(e instanceof Error ? e.message : String(e), true); }
  finally { setBusy(false); updateContextHelp(); }
}
function updateContextHelp() {
  const context = Number($<HTMLSelectElement>('context').value), batch = BATCHES[$<HTMLSelectElement>('performance').value as Preset] ?? 32;
  const pending = info && context !== info.contextLength ? ` Loaded context: ${info.contextLength.toLocaleString()} tokens. Apply & reload to use ${context.toLocaleString()}.` : '';
  const memory = info ? attentionMemory(info.config, context, batch).totalBytes : 0;
  $('context-help').textContent = `Context includes your prompt, chat history and reply.${info ? ` This model needs about ${gib(memory)} for attention state, plus working memory and weight cache.` : ' Longer contexts need more GPU memory.'}${pending || ' Apply & reload to change context.'}`;
  const output = Number($<HTMLSelectElement>('max-tokens').value), loadedContext = info?.contextLength ?? context;
  $('output-help').textContent = `Up to ${output.toLocaleString()} reply tokens, limited by the space left in the ${loadedContext.toLocaleString()}-token ${info ? 'loaded ' : ''}context. Output length changes without reloading.`;
}
for (const id of ['context', 'max-tokens']) $(id).addEventListener('change', updateContextHelp);
updateContextHelp();
const pick = () => $<HTMLInputElement>('model-file').click();
const openOllama = createOllamaPicker(loadModel, pick, () => busy);
const openLibrary = createLibraryPicker(loadModel, pick, () => busy);
$('open-ollama').addEventListener('click', openOllama);
if (new URLSearchParams(location.search).get('models') === 'ollama') openOllama();
for (const el of document.querySelectorAll<HTMLElement>('[data-open]')) el.addEventListener('click', () => { if (busy) return; if (el.dataset.open === 'library') openLibrary(); else pick(); });
$<HTMLInputElement>('model-file').addEventListener('change', e => { const file = (e.target as HTMLInputElement).files?.[0]; if (file) void loadModel(file); });
$('shared-model').addEventListener('click', () => { if (sharedSource) void loadModel(sharedSource); });
$('reload').addEventListener('click', () => { if (source) void loadModel(source, true); });
$<HTMLSelectElement>('performance').addEventListener('change', () => {
  const mode = $<HTMLSelectElement>('performance').value as Preset, budget = PRESET_BUDGET[mode];
  budgetTouched = false;
  $<HTMLInputElement>('budget').value = String(budget); $('budget-display').textContent = sizeLabel(budget);
  if (!info) $('stat-budget').textContent = `${(budget / 1024).toFixed(1)} GiB`;
  $('reload').textContent = mode === 'fast' ? 'Apply Faster & reload model' : mode === 'lite' ? 'Apply Lite & reload model' : 'Apply & reload model';
  $('performance-help').textContent = `${PRESET_HELP[mode]} Apply & reload to activate.`;
  updateContextHelp();
});
$<HTMLInputElement>('budget').addEventListener('input', () => {
  budgetTouched = true;
  const mib = Number($<HTMLInputElement>('budget').value);
  $('budget-display').textContent = sizeLabel(mib);
  if (!info) $('stat-budget').textContent = `${(mib / 1024).toFixed(1)} GiB`;
});
$('stop').addEventListener('click', () => { runtime.stop(); $('progress-text').textContent = 'Stopping at the next page boundary…'; });
$<HTMLButtonElement>('new-chat').addEventListener('click', async () => { if (busy) return; messages = []; if (info) await runtime.reset(); $('conversation').replaceChildren(); clearNotice(); $('token-stats').textContent = 'New conversation'; switchPage('playground'); });
document.querySelectorAll<HTMLButtonElement>('[data-prompt]').forEach(el => el.addEventListener('click', () => { $<HTMLTextAreaElement>('prompt').value = el.dataset.prompt!; $<HTMLTextAreaElement>('prompt').focus(); }));
function appendMessage(role: 'user' | 'assistant', text: string) {
  const article = document.createElement('article'); article.className = `message ${role}`;
  const label = document.createElement('div'); label.className = 'message-label'; label.textContent = role === 'user' ? 'YOU' : 'WEBGPU';
  const content = document.createElement('div'); content.className = 'message-content'; content.textContent = text;
  article.append(label, content); $('conversation').append(article); $('conversation').scrollTop = $('conversation').scrollHeight; return content;
}
$('chat-form').addEventListener('submit', async e => {
  e.preventDefault(); const prompt = $<HTMLTextAreaElement>('prompt').value.trim(); if (!prompt || !info || busy) return;
  clearNotice(); $('conversation').querySelector('.ready-message')?.remove();
  const history: Message[] = [...messages, { role: 'user', content: prompt }];
  appendMessage('user', prompt); const answer = appendMessage('assistant', 'Loading weight pages…');
  $<HTMLTextAreaElement>('prompt').value = ''; setBusy(true); let text = '', lastChunk: { tokens: number; promptTokens: number; contextLength: number } | undefined;
  try {
    for await (const chunk of runtime.generate(history, { maxTokens: Number($<HTMLSelectElement>('max-tokens').value), temperature: Number($<HTMLInputElement>('temperature').value) })) {
      text = chunk.text; lastChunk = chunk; answer.textContent = text || '…'; updateStats(chunk.stats);
      $('token-stats').textContent = `${chunk.tokens.toLocaleString()} reply ${chunk.tokens === 1 ? 'token' : 'tokens'} · ${(chunk.promptTokens + chunk.tokens).toLocaleString()} / ${chunk.contextLength.toLocaleString()} context · ${chunk.seconds.toFixed(1)}s`;
      $('conversation').scrollTop = $('conversation').scrollHeight;
    }
    messages = [...history, { role: 'assistant', content: text }];
    if (!text) answer.textContent = '(The model ended its turn.)';
    if (lastChunk && lastChunk.promptTokens + lastChunk.tokens >= lastChunk.contextLength) notify('Context is full. Start a new conversation or apply a larger context.');
    else if (lastChunk && lastChunk.tokens >= Number($<HTMLSelectElement>('max-tokens').value)) notify('Reply reached the selected output limit. Increase Max output for longer replies.');
  } catch (e) {
    if (e instanceof Error && e.name === 'AbortError') { answer.textContent = text || '(Stopped.)'; if (text) messages = [...history, { role: 'assistant', content: text }]; }
    else { answer.textContent = text || 'Generation could not complete.'; notify(e instanceof Error ? e.message : String(e), true); }
  } finally { setBusy(false); }
});
$<HTMLTextAreaElement>('prompt').addEventListener('keydown', e => { if (e.key === 'Enter' && !e.shiftKey && !e.isComposing) { e.preventDefault(); $<HTMLFormElement>('chat-form').requestSubmit(); } });
$('export-chat').addEventListener('click', () => { if (!messages.length) return notify('Start a conversation before exporting.'); const url = URL.createObjectURL(new Blob([JSON.stringify({ model: info?.config.name, messages }, null, 2)], { type: 'application/json' })); const a = document.createElement('a'); a.href = url; a.download = 'webgpu-conversation.json'; a.click(); setTimeout(() => URL.revokeObjectURL(url), 1000); });
$('copy-api').addEventListener('click', () => void copyText(sdkExample, 'Browser SDK example copied.'));
$('copy-client').addEventListener('click', () => void copyText(clientExample(apiKey), 'Client snippet copied.'));
$('copy-embed').addEventListener('click', () => void copyText(embedExample(apiKey), 'Embedding snippet copied.'));
$('copy-key').addEventListener('click', () => void copyText(apiKey, 'API key copied.'));
$('reveal-key').addEventListener('click', () => { revealedKey = !revealedKey; updateApiPanel(); });
$('rotate-key').addEventListener('click', () => { apiKey = rotateApiKey(); restartBridge(); revealedKey = true; updateApiPanel(); notify('New API key generated. The previous key stopped working.'); });
updateApiPanel();
setInterval(() => { if (active === 'api') updateApiPanel(); }, 2500);
window.addEventListener('beforeunload', () => { runtime.dispose(); bridge.dispose(); });
void (async () => {
  const adapter = await navigator.gpu?.requestAdapter().catch(() => null);
  $('support-title').textContent = adapter ? 'WebGPU available' : 'WebGPU unavailable'; $('support-dot').classList.toggle('offline', !adapter);
  if (!adapter) notify('WebGPU is unavailable in this browser. Enable hardware acceleration and use a browser with WebGPU support.', true);
  try { if (!['127.0.0.1', 'localhost', '[::1]'].includes(location.hostname)) return; const response = await fetch('/model-info'); if (!response.ok) return; const data = await response.json(); if (data?.url && data?.size && data.kind !== 'ollama') { sharedSource = { ...data, url: new URL(data.url, location.href).href }; $('shared-model').hidden = false; $('shared-model').textContent = `Use ${data.name} →`; } } catch { /* Static hosting: use the folder picker or the file picker. */ }
})();
