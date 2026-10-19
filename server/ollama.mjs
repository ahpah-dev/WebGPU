import { createHash } from 'node:crypto';
import { open, readFile, readdir, realpath, stat } from 'node:fs/promises';
import { homedir } from 'node:os';
import { join, resolve, sep } from 'node:path';

const architectures = new Set(['gemma3', 'qwen35']);
const quants = new Set(['F32', 'F16', 'Q4_0', 'Q8_0', 'Q4_K_S', 'Q4_K_M', 'Q5_K_S', 'Q5_K_M', 'Q6_K']);
const digestPattern = /^sha256:[a-f0-9]{64}$/;
export async function defaultOllamaDirectory() {
  if (process.env.OLLAMA_MODELS) return resolve(process.env.OLLAMA_MODELS);
  const personal = join(homedir(), '.ollama', 'models');
  if (process.platform !== 'linux' || await stat(personal).then(() => true, () => false)) return personal;
  return '/usr/share/ollama/.ollama/models';
}

/** Reads Ollama manifests and weight bytes only. Never invokes an inference API. */
export class OllamaCatalog {
  #root;
  #models = new Map();
  constructor(root) { this.#root = resolve(root); }
  async #json(path, boundary) {
    const actual = await realpath(path);
    if (!actual.startsWith(boundary + sep)) throw new Error('Model file is outside the Ollama directory.');
    const info = await stat(actual);
    if (!info.isFile() || info.size > 1024 * 1024) throw new Error('Invalid Ollama metadata file.');
    return JSON.parse(await readFile(actual, 'utf8'));
  }
  async refresh() {
    const entries = [];
    let root;
    try { root = await realpath(this.#root); }
    catch (e) { if (e.code === 'ENOENT') { this.#models.clear(); return []; } throw e; }
    const manifests = join(root, 'manifests'), blobs = join(root, 'blobs');
    const walk = async (parts = []) => {
      if (parts.length > 8 || entries.length >= 2000) return;
      let files;
      try { files = await readdir(join(manifests, ...parts), { withFileTypes: true }); }
      catch (e) { if (e.code === 'ENOENT') return; throw e; }
      for (const entry of files) {
        if (entries.length >= 2000) break;
        if (entry.isDirectory()) await walk([...parts, entry.name]);
        else if (entry.isFile() && parts.length >= 3) entries.push([...parts, entry.name]);
      }
    };
    await walk();
    const models = new Map();
    for (const parts of entries) {
      try {
        const manifest = await this.#json(join(manifests, ...parts), root);
        const layer = manifest.layers?.find(l => l.mediaType === 'application/vnd.ollama.image.model');
        if (!layer || !digestPattern.test(layer.digest)) continue; // Cloud entries have no local model bytes.
        const [registry, namespace, ...tail] = parts, tag = tail.pop(), base = tail.join('/');
        const name = `${registry === 'registry.ollama.ai' ? '' : registry + '/'}${namespace === 'library' ? '' : namespace + '/'}${base}:${tag}`;
        const id = createHash('sha256').update(parts.join('/') + '\0' + layer.digest).digest('hex');
        let config = {};
        if (digestPattern.test(manifest.config?.digest)) {
          try { config = await this.#json(join(blobs, manifest.config.digest.replace(':', '-')), root); } catch { /* Inspect compatibility when opening if config is absent. */ }
        }
        const architecture = String(config.model_family || ''), quantization = String(config.file_type || '');
        let reason = '', path, size = Number(layer.size) || 0;
        try {
          path = await realpath(join(blobs, layer.digest.replace(':', '-')));
          if (!path.startsWith(blobs + sep)) throw new Error('Invalid model path.');
          const info = await stat(path); size = info.size;
          if (!info.isFile() || !Number.isSafeInteger(layer.size) || layer.size !== size) throw new Error('Incomplete model.');
          const file = await open(path, 'r');
          try { const magic = Buffer.alloc(4); await file.read(magic, 0, 4, 0); if (magic.toString() !== 'GGUF') reason = 'This model is not stored as GGUF.'; }
          finally { await file.close(); }
        } catch { reason = 'Local weights are missing or incomplete.'; }
        if (!reason && architecture && !architectures.has(architecture)) reason = `The ${architecture} architecture is not supported by this browser runtime yet.`;
        if (!reason && quantization && !quants.has(quantization)) reason = `${quantization} quantization is not supported yet.`;
        models.set(id, { id, name, architecture, quantization, parameterSize: String(config.model_type || ''), size, supported: !reason, reason, url: `/ollama/models/${id}/weights.gguf`, path });
      } catch { /* A malformed or partially written manifest must not hide other models. */ }
    }
    this.#models = models;
    return [...models.values()].map(({ path, ...model }) => model).sort((a, b) => Number(b.supported) - Number(a.supported) || a.name.localeCompare(b.name));
  }
  resolve(id) { const model = this.#models.get(id); return model?.supported ? model : undefined; }
}
