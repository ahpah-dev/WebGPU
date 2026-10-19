import type { SourceInput } from './core/source.ts';

interface LocalModel { name: string; size: number; architecture: string; quantization: string; parameterSize: string; supported: boolean; reason: string; url: string }
export function createOllamaPicker(load: (source: SourceInput) => Promise<void>, chooseFile: () => void, isBusy: () => boolean) {
  const dialog = document.createElement('dialog'); dialog.className = 'ollama-dialog'; dialog.setAttribute('aria-labelledby', 'ollama-title');
  dialog.innerHTML = `<div class="ollama-heading"><div><span class="eyebrow">ALREADY ON YOUR MACHINE</span><h2 id="ollama-title">Open from Ollama</h2></div><button class="text-button" id="ollama-close" aria-label="Close Ollama library">✕</button></div>
    <p class="ollama-intro">Choose a downloaded model. Its weights stay local and inference runs in your browser.</p>
    <div id="ollama-local"><div class="ollama-toolbar"><input id="ollama-search" type="search" placeholder="Search your models…" aria-label="Search Ollama models"><button class="secondary" id="ollama-refresh">Refresh</button></div><p id="ollama-status" role="status"></p><div id="ollama-list" class="ollama-list"></div></div>
    <div id="ollama-hosted" hidden><p>Browse your installed models in the local WebGPU app. Start it with:</p><pre><code>npm start</code></pre><a class="primary ollama-local-link" href="http://127.0.0.1:8787/?models=ollama">Open local Ollama library →</a><p class="field-help">Ollama can stay closed. The local app reads your existing model files.</p></div>
    <div class="ollama-footer"><span>No copying or converting files.</span><button class="text-button" id="ollama-file">Choose a GGUF file instead</button></div>`;
  document.body.append(dialog);
  const $ = <T extends HTMLElement = HTMLElement>(selector: string) => dialog.querySelector<T>(selector)!;
  const local = ['127.0.0.1', 'localhost', '[::1]'].includes(location.hostname);
  let models: LocalModel[] = [], request: AbortController | undefined;
  const render = () => {
    const query = $<HTMLInputElement>('#ollama-search').value.trim().toLowerCase();
    const shown = models.filter(m => `${m.name} ${m.architecture} ${m.quantization}`.toLowerCase().includes(query));
    $('#ollama-list').replaceChildren();
    if (!models.length) return;
    $('#ollama-status').textContent = shown.length ? `${shown.length} model${shown.length === 1 ? '' : 's'} · select one to open` : 'No models match your search.';
    for (const model of shown) {
      const button = document.createElement('button'); button.className = 'ollama-model'; button.disabled = !model.supported;
      button.setAttribute('aria-label', `${model.supported ? 'Open' : 'Unavailable'} ${model.name}`);
      const details = document.createElement('span'), name = document.createElement('strong'), meta = document.createElement('span'), hint = document.createElement('small'), arrow = document.createElement('span');
      name.textContent = model.name;
      meta.textContent = [(model.size / 1024 ** 3).toFixed(2) + ' GiB', model.parameterSize, model.quantization].filter(Boolean).join(' · ');
      hint.textContent = model.supported ? model.architecture ? 'Ready to open in your browser' : 'Compatibility checked when opening' : model.reason;
      details.append(name, meta, hint); arrow.textContent = model.supported ? '↗' : 'Unavailable'; arrow.className = 'ollama-model-action'; button.append(details, arrow);
      button.onclick = () => { if (isBusy()) return; dialog.close(); void load({ url: new URL(model.url, location.href).href, size: model.size, name: model.name }); };
      $('#ollama-list').append(button);
    }
  };
  const refresh = async () => {
    request?.abort(); const current = new AbortController(); request = current;
    models = []; $('#ollama-list').replaceChildren(); $('#ollama-status').textContent = 'Finding your downloaded Ollama models…'; $<HTMLButtonElement>('#ollama-refresh').disabled = true;
    try {
      const response = await fetch('/api/ollama/models', { signal: current.signal, cache: 'no-store' });
      if (!response.ok) throw new Error(response.status === 404 ? 'Restart the local WebGPU app with npm start to enable the Ollama library.' : 'Could not read your Ollama library. Check access to your model directory, then refresh.');
      const result = await response.json();
      if (!Array.isArray(result.models)) throw new Error('The local app returned an invalid model list.');
      models = result.models.filter((m: LocalModel) => typeof m.name === 'string' && typeof m.supported === 'boolean' && Number.isSafeInteger(m.size) && m.size >= 0 && /^\/ollama\/models\/[a-f0-9]{64}\/weights\.gguf$/.test(m.url));
      $('#ollama-status').textContent = models.length ? '' : 'No downloaded models found. Download a model in Ollama, then refresh. If you use a custom directory, start WebGPU with --ollama-dir or OLLAMA_MODELS.';
      render();
    } catch (e) { if (!current.signal.aborted) $('#ollama-status').textContent = e instanceof Error ? e.message : String(e); }
    finally { if (request === current) $<HTMLButtonElement>('#ollama-refresh').disabled = false; }
  };
  $('#ollama-close').onclick = () => dialog.close();
  $('#ollama-refresh').onclick = () => { void refresh(); };
  $('#ollama-search').oninput = render;
  $('#ollama-file').onclick = () => { dialog.close(); chooseFile(); };
  dialog.addEventListener('close', () => request?.abort());
  return () => {
    if (isBusy()) return;
    $('#ollama-local').hidden = !local; $('#ollama-hosted').hidden = local;
    dialog.showModal();
    if (local) { const search = $<HTMLInputElement>('#ollama-search'); search.value = ''; search.focus(); void refresh(); }
  };
}
