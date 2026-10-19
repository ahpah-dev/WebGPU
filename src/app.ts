import './style.css';
import { WebGPU, type Message, type ModelInfo, type Progress, type SourceInput } from './sdk.ts';
import { createOllamaPicker } from './ollama-picker.ts';
import { attentionMemory } from './core/context.ts';

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
const $ = <T extends HTMLElement = HTMLElement>(id: string) => document.getElementById(id) as T;
const gib = (n: number) => `${(n / 1024 ** 3).toFixed(2)} GiB`;
let runtime = new WebGPU(), info: ModelInfo | undefined, busy = false, source: SourceInput | undefined;
let messages: Message[] = [];
let active = 'playground';
let sharedSource: SourceInput | undefined;

$('app').innerHTML = `
<aside class="sidebar">
  <a class="brand" aria-label="WebGPU home" href="#playground"><span class="brandmark">W</span><span>WebGPU<span class="brand-dot">.</span></span></a>
  <span class="edition">BROWSER INFERENCE / 0.5</span>
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
  <header><div class="breadcrumb">Workspace <span>/</span> <strong id="page-label">Playground</strong></div><div class="header-right"><span class="local-badge"><span></span> Runs on your device</span><a href="https://github.com/ahpah-dev/WebGPU" target="_blank" rel="noreferrer" class="github-link">GitHub ${icon('link')}</a></div></header>
  <main>
    <section class="page active" id="page-playground">
      <div class="page-heading"><div><div class="eyebrow"><span></span> A LITTLE LESS HARDWARE. A LOT MORE POSSIBILITY.</div><h1>Big models.<br><span>Small footprint.</span></h1><p class="intro">Local AI, right in your browser. Stream the weights.<br>Keep the intelligence on your machine.</p></div><div class="hero-art" aria-hidden="true"><div class="orbit orbit-one"></div><div class="orbit orbit-two"></div><span class="art-node n1"></span><span class="art-node n2"></span><span class="art-node n3"></span><div class="hero-chip">${icon('chip')}<b>WebGPU</b><span>PAGED COMPUTE</span></div><div class="art-tag"><i></i> Browser native</div></div></div>
      <div class="stats-row"><div class="stat"><span class="stat-icon">${icon('box')}</span><div><label>Model parameters</label><strong id="stat-params">— <small>no model loaded</small></strong></div></div><div class="stat"><span class="stat-icon">${icon('chip')}</span><div><label>GPU allocation budget</label><strong id="stat-budget">2.0 <small>GiB</small></strong></div><span class="stat-tag">YOU SET THE LIMIT</span></div><div class="stat"><span class="stat-icon">${icon('shield')}</span><div><label>Inference location</label><strong>In your browser <span class="tiny-dot"></span></strong></div></div></div>
      <div class="workspace-grid">
        <div class="chat-panel panel"><div class="panel-top"><div class="panel-title">${icon('chat')} Conversation <span class="subtle-badge" id="chat-model">No model</span></div><button class="text-button" id="export-chat">Export ${icon('link')}</button></div>
          <div id="conversation" class="conversation" aria-live="polite"><div class="empty-chat" id="empty-chat"><div class="empty-symbol">${icon('spark')}</div><h2>A bigger model. A smaller footprint.</h2><p>Open Qwen3.8 27B or Gemma 3 GGUF weights.<br>The model stays on your disk. Inference stays in this tab.</p><div class="model-open-actions"><button class="primary" id="open-ollama-empty">${icon('box')} Open from Ollama</button><button class="secondary" id="choose-empty">${icon('upload')} Choose GGUF file</button></div><div class="starter-prompts"><button data-prompt="Explain gravity in one sentence.">Explain a concept <span>↗</span></button><button data-prompt="Write a short Python function that checks whether a number is prime.">Write some code <span>↗</span></button></div></div></div>
          <div class="generation-progress" id="generation-progress" hidden><div><span id="progress-text">Preparing…</span><span id="progress-percent">0%</span></div><div class="progress-track"><i id="progress-bar"></i></div></div>
          <form id="chat-form" class="composer"><textarea id="prompt" rows="2" placeholder="Ask something. Keep it local." aria-label="Message" maxlength="1048576"></textarea><div class="composer-bottom"><span>${icon('shield')} Private · no cloud inference</span><button class="send-button" id="send" type="submit" aria-label="Send message" disabled>${icon('arrow')}</button><button class="send-button stop" id="stop" type="button" aria-label="Stop generation" hidden>${icon('stop')}</button></div></form><div class="chat-foot"><span>Enter to send · Shift + Enter for a new line</span><span id="token-stats">Ready when you are</span></div>
        </div>
        <aside class="settings-panel panel"><div class="panel-top"><div class="panel-title">${icon('chip')} Run configuration</div></div><div class="settings-body">
          <label class="field-label">LOCAL MODEL <span>OLLAMA / GGUF</span></label><button id="open-ollama" class="secondary full ollama-browse">${icon('box')} Open from Ollama</button><button id="choose-model" class="model-picker">${icon('box')}<span id="model-filename">Select a model file<small>Qwen3.8 · Gemma 3 · GGUF</small></span>${icon('plus')}</button><button id="shared-model" class="shared-model" hidden>Use locally served model →</button>
          <div class="field-divider"></div><label class="field-label" for="performance">PERFORMANCE</label><select id="performance"><option value="low-memory" selected>Memory saver · 2 GiB preset</option><option value="fast">Faster · 6 GiB preset</option></select><p id="performance-help" class="field-help">Memory saver keeps GPU use low. Faster keeps more weights resident and reads larger chunks. Apply & reload after changing modes.</p><label class="field-label" for="budget">GPU MEMORY BUDGET <span id="budget-display">2 GiB</span></label><input id="budget" type="range" min="1024" max="8192" step="1024" value="2048"><div class="range-labels"><span>1 GiB</span><span>8 GiB</span></div><p class="field-help">Weights that don't fit stream from disk. More memory can reduce repeated reads.</p>
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
    <section class="page" id="page-models"><div class="section-heading"><span class="eyebrow">BRING YOUR OWN WEIGHTS</span><h1>A local file.<br><span>A whole lot of intelligence.</span></h1><p class="intro">Qwen3.8 27B and Gemma 3. Text inference, entirely in your browser.</p></div><div class="panel ollama-library-banner"><div><h2>Your Ollama models, ready to open.</h2><p>Browse your downloaded library by name. No hunting for weight files.</p></div><button class="primary" id="open-ollama-library">Browse Ollama models →</button></div><div class="model-guide-grid"><div class="panel model-card"><span class="model-badge">NEW · 27B HYBRID</span><div class="model-card-symbol">Q<span>3.8</span></div><h2>Qwen3.8 27B</h2><p>64 text layers with recurrent and full attention. A bigger model, streamed within your GPU budget.</p><div class="model-facts"><span>Q4_K_M file <b>~16.8 GB</b></span><span>Chat mode <b>Non-thinking · text</b></span><span>Execution <b>Browser WebGPU</b></span></div><button class="primary full" id="choose-qwen">${icon('upload')} Open Qwen GGUF</button><p class="field-help"><a href="https://ollama.com/library/qwen3.8:27b" target="_blank" rel="noreferrer">Get Qwen3.8 weights ↗</a></p></div><div class="panel model-card"><span class="model-badge">27B TARGET</span><div class="model-card-symbol">G<span>3</span></div><h2>Gemma 3 27B</h2><p>All 62 transformer layers. Quantized weights. Streamed through a configurable GPU memory budget.</p><div class="model-facts"><span>Q4_K_M file <b>~16.6 GB</b></span><span>Model architecture <b>gemma3</b></span><span>Execution <b>Browser WebGPU</b></span></div><button class="primary full" id="choose-library">${icon('upload')} Open local GGUF</button></div><div class="panel guide-card"><h2>Get your model ready</h2><ol><li><strong>Download your model weights</strong><p>Use an existing GGUF, or download with Ollama. You need enough disk space for the model and its license applies.</p><code>ollama pull qwen3.8:27b</code><p>Gemma 3 is also supported: <code>ollama pull gemma3:27b</code></p></li><li><strong>Open the GGUF in this browser</strong><p>Select your file. It is read locally and never uploaded. For Ollama's stored weights, the included byte server can locate them:</p><code>npm start</code><p>Click <strong>Open from Ollama</strong> and choose a downloaded model by name.</p></li><li><strong>Set a memory budget and chat</strong><p>Start with 2 GiB and a 512-token context. Lower budgets mean more paging. Large prompts and long answers can take time.</p></li></ol><p class="field-help">The optional Node server only serves files. It does no inference, and Ollama does not need to be running. Use a single GGUF with Q4_K, Q5_K, Q6_K, Q8_0, Q4_0, F16 or F32 tensors. Qwen uses non-thinking chat. Image projectors, split files and other architectures are not implemented.</p></div></div></section>
    <section class="page" id="page-api"><div class="section-heading"><span class="eyebrow">MADE TO BUILD WITH</span><h1>Your browser.<br><span>Your inference API.</span></h1><p class="intro">A streaming JavaScript API backed by a dedicated Web Worker.</p></div><div class="panel code-panel"><div class="panel-top"><div class="panel-title">${icon('code')} Browser JavaScript</div><button id="copy-api" class="text-button">${icon('copy')} Copy</button></div><pre><code id="api-example"></code></pre></div><div class="api-grid"><div class="panel api-card"><code>await model.load(file, options)</code><p>Inspect the GGUF, compile kernels, allocate bounded attention state, and prepare the weight pager.</p></div><div class="panel api-card"><code>model.generate(messages, options)</code><p>Async iterator of decoded text, token counts, elapsed time, and actual allocation counters.</p></div><div class="panel api-card"><code>model.stop() · model.dispose()</code><p>Cancel generation at the next page or layer boundary. Dispose terminates the worker and releases GPU resources.</p></div></div><p class="api-note">This is a browser SDK, not a server REST endpoint. All model computation stays inside the browser. Run <code>npm run build</code> to produce the standalone SDK under <code>dist/sdk/</code>.</p></section>
    <div id="notice" class="notice" role="status" hidden></div>
  </main><footer><span>WebGPU <b> / </b> Browser-native local AI</span><span>Open source. No API keys. No inference fees.</span></footer>
</div><input id="model-file" type="file" accept=".gguf,application/octet-stream" hidden>
`;
const example = `import { WebGPU } from './sdk/sdk.js';

// A File from <input type="file"> — never uploaded.
const model = new WebGPU();
model.onProgress = ({ detail }) => console.log(detail);

await model.load(file, {
  gpuBudgetMiB: 6144,
  contextLength: 512,
  performance: 'fast', // 6 GiB GPU preset; larger pages and prompt batches.
});

for await (const chunk of model.generate(
  [{ role: 'user', content: 'Explain gravity in one sentence.' }],
  { maxTokens: 64, temperature: 0.7 },
)) {
  output.textContent += chunk.delta;
}

model.dispose();`;
$('api-example').textContent = example;
function notify(message: string, error = false) { $('notice').textContent = message; $('notice').hidden = false; $('notice').classList.toggle('error', error); }
function clearNotice() { $('notice').hidden = true; }
function switchPage(page: string) {
  if (!['playground', 'runtime', 'models', 'api'].includes(page)) return;
  active = page; document.querySelectorAll('.page').forEach(p => p.classList.toggle('active', p.id === `page-${page}`));
  document.querySelectorAll('.nav-item').forEach(p => p.classList.toggle('active', (p as HTMLElement).dataset.page === page));
  $('page-label').textContent = ({ playground: 'Playground', runtime: 'Runtime', models: 'Model library', api: 'Developer API' })[page]!;
}
document.querySelectorAll<HTMLElement>('[data-page]').forEach(el => el.addEventListener('click', () => switchPage(el.dataset.page!)));
document.querySelector('.brand')!.addEventListener('click', () => switchPage('playground'));
function setBusy(value: boolean) {
  busy = value; $<HTMLButtonElement>('send').disabled = value || !info;
  $('send').hidden = value; $('stop').hidden = !value;
  for (const id of ['choose-model', 'choose-library', 'choose-qwen', 'choose-empty', 'shared-model', 'new-chat', 'open-ollama', 'open-ollama-empty', 'open-ollama-library']) { const el = $<HTMLButtonElement>(id); if (el) el.disabled = value; }
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
async function loadModel(selected: SourceInput, preserveConversation = false) {
  if (busy) return; clearNotice(); source = selected; info = undefined; switchPage('playground'); setBusy(true);
  runtime.dispose(); runtime = new WebGPU(); runtime.onProgress = onProgress;
  try {
    const mode = $<HTMLSelectElement>('performance').value as 'low-memory' | 'fast';
    const loaded = await runtime.load(selected, { gpuBudgetMiB: Number($<HTMLInputElement>('budget').value), contextLength: Number($<HTMLSelectElement>('context').value), batchSize: mode === 'fast' ? 128 : 32, performance: mode });
    info = loaded;
    if (!preserveConversation || !messages.length) {
      messages = []; $('conversation').replaceChildren();
      const ready = document.createElement('div'); ready.className = 'ready-message'; ready.innerHTML = `${icon('spark')}<h2>Ready for a bigger conversation.</h2><p>Model loaded. Weight pages will stream from disk as you generate.</p>`; $('conversation').append(ready);
    }
    $('stat-params').textContent = `${(loaded.config.parameters / 1e9).toFixed(1)}B`;
    $('chat-model').textContent = loaded.config.name;
    $('model-filename').textContent = selected instanceof File ? selected.name : selected.name || 'Local GGUF';
    $('stat-budget').textContent = `${(loaded.gpuBudgetMiB / 1024).toFixed(1)} GiB`;
    const recurrentCount = loaded.config.recurrent?.layers.filter(Boolean).length ?? 0;
    $('runtime-pages').textContent = `${loaded.stats.pageBytes / 1024 ** 2} MiB pages`; $('runtime-batch').textContent = `Prompt batches: up to ${loaded.batchSize} tokens.`;
    $('runtime-file').textContent = gib(loaded.fileBytes); $('layer-count').textContent = `${loaded.config.layers} layers${loaded.config.recurrent ? ` · ${recurrentCount} recurrent / ${loaded.config.layers - recurrentCount} full attention` : ''}`;
    $('layers').replaceChildren(...Array.from({ length: loaded.config.layers }, (_, i) => { const span = document.createElement('span'); span.textContent = String(i + 1).padStart(2, '0'); span.title = `Layer ${i + 1}: ${loaded.config.recurrent?.layers[i] ? 'recurrent attention' : 'attention'}`; return span; }));
    updateStats(loaded.stats); $('adapter-name').textContent = loaded.adapter; updateContextHelp();
    notify(`${loaded.config.name} is ready. ${gib(loaded.fileBytes)} of weights will be paged within a ${loaded.gpuBudgetMiB / 1024} GiB GPU allocation budget.`);
    switchPage('playground'); $<HTMLTextAreaElement>('prompt').focus();
  } catch (e) { notify(e instanceof Error ? e.message : String(e), true); }
  finally { setBusy(false); updateContextHelp(); }
}
function updateContextHelp() {
  const context = Number($<HTMLSelectElement>('context').value), batch = $<HTMLSelectElement>('performance').value === 'fast' ? 128 : 32;
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
for (const id of ['open-ollama', 'open-ollama-empty', 'open-ollama-library']) $(id).addEventListener('click', openOllama);
if (new URLSearchParams(location.search).get('models') === 'ollama') openOllama();
for (const id of ['choose-empty', 'choose-model', 'choose-library', 'choose-qwen']) $(id).addEventListener('click', pick);
$<HTMLInputElement>('model-file').addEventListener('change', e => { const file = (e.target as HTMLInputElement).files?.[0]; if (file) void loadModel(file); });
$('shared-model').addEventListener('click', () => { if (sharedSource) void loadModel(sharedSource); });
$('reload').addEventListener('click', () => { if (source) void loadModel(source, true); });
$<HTMLSelectElement>('performance').addEventListener('change', () => {
  const fast = $<HTMLSelectElement>('performance').value === 'fast', budget = fast ? 6144 : 2048;
  $<HTMLInputElement>('budget').value = String(budget); $('budget-display').textContent = `${budget / 1024} GiB`;
  if (!info) $('stat-budget').textContent = `${(budget / 1024).toFixed(1)} GiB`;
  $('reload').textContent = fast ? 'Apply Faster & reload model' : 'Apply & reload model';
  $('performance-help').textContent = fast ? 'Faster uses a 6 GiB GPU budget, up to 64 MiB pages and batches of 128 prompt tokens. Adjust the budget to fit your GPU. The first prompt still reads the model; later tokens can reuse more weights. Apply & reload to activate.' : 'Memory saver uses a 2 GiB GPU preset and 16 MiB pages. Apply & reload to activate.';
  updateContextHelp();
});
$<HTMLInputElement>('budget').addEventListener('input', () => { const n = Number($<HTMLInputElement>('budget').value) / 1024; $('budget-display').textContent = `${n} GiB`; if (!info) $('stat-budget').textContent = `${n.toFixed(1)} GiB`; });
$('stop').addEventListener('click', () => { runtime.stop(); $('progress-text').textContent = 'Stopping at the next page boundary…'; });
$('new-chat').addEventListener('click', async () => { if (busy) return; messages = []; if (info) await runtime.reset(); $('conversation').replaceChildren(); clearNotice(); $('token-stats').textContent = 'New conversation'; switchPage('playground'); });
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
$('copy-api').addEventListener('click', async () => { try { await navigator.clipboard.writeText(example); notify('Browser API example copied.'); } catch { notify('Clipboard unavailable. Select and copy the example.', true); } });
window.addEventListener('beforeunload', () => runtime.dispose());
void (async () => {
  const adapter = await navigator.gpu?.requestAdapter().catch(() => null);
  $('support-title').textContent = adapter ? 'WebGPU available' : 'WebGPU unavailable'; $('support-dot').classList.toggle('offline', !adapter);
  if (!adapter) notify('WebGPU is unavailable in this browser. Enable hardware acceleration and use a browser with WebGPU support.', true);
  try { if (!['127.0.0.1', 'localhost', '[::1]'].includes(location.hostname)) return; const response = await fetch('/model-info'); if (!response.ok) return; const data = await response.json(); if (data?.url && data?.size && data.kind !== 'ollama') { sharedSource = { ...data, url: new URL(data.url, location.href).href }; $('shared-model').hidden = false; $('shared-model').textContent = `Use ${data.name} →`; } } catch { /* Static hosting: use the file picker. */ }
})();
