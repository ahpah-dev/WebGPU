// Static byte server only. No model evaluation, tokenizer, GPU, or inference API runs in Node.
import { createServer } from 'node:http';
import { open, readFile, stat } from 'node:fs/promises';
import { createReadStream } from 'node:fs';
import { pipeline } from 'node:stream/promises';
import { dirname, extname, join, resolve, sep } from 'node:path';
import { fileURLToPath } from 'node:url';
import { homedir } from 'node:os';
const root = resolve(dirname(fileURLToPath(import.meta.url)), '../dist');
const args = process.argv.slice(2);
const option = name => { const i = args.indexOf(name); return i < 0 ? undefined : args[i + 1]; };
let modelPath = option('--model'), modelName;
const ollamaName = option('--ollama-model');
if (ollamaName) {
  if (!/^[a-zA-Z0-9_.-]+:[a-zA-Z0-9_.-]+$/.test(ollamaName)) throw new Error('Use a library model:tag, for example gemma3:27b.');
  const [name, tag] = ollamaName.split(':');
  const models = process.env.OLLAMA_MODELS || join(homedir(), '.ollama', 'models');
  const manifest = JSON.parse(await readFile(join(models, 'manifests', 'registry.ollama.ai', 'library', name, tag), 'utf8'));
  const layer = manifest.layers.find(l => l.mediaType === 'application/vnd.ollama.image.model');
  if (!layer || !/^sha256:[a-f0-9]{64}$/.test(layer.digest)) throw new Error('No local GGUF weights found.');
  modelPath = join(models, 'blobs', layer.digest.replace(':', '-')); modelName = `${ollamaName}.gguf`;
}
if (modelPath) {
  modelPath = resolve(modelPath);
  const f = await open(modelPath, 'r');
  try { const b = Buffer.alloc(4); await f.read(b, 0, 4, 0); if (b.toString() !== 'GGUF') throw new Error('The selected model is not GGUF.'); } finally { await f.close(); }
}
const port = Number(option('--port') || process.env.PORT || 8787);
const types = { '.html': 'text/html; charset=utf-8', '.js': 'text/javascript; charset=utf-8', '.css': 'text/css; charset=utf-8', '.json': 'application/json', '.svg': 'image/svg+xml' };
const server = createServer(async (req, res) => {
  try {
    if (!['GET', 'HEAD'].includes(req.method)) { res.writeHead(405); return res.end(); }
    if (![`127.0.0.1:${port}`, `localhost:${port}`, `[::1]:${port}`].includes(req.headers.host)) { res.writeHead(403); return res.end('Invalid Host'); }
    if (req.headers.origin && ![`http://127.0.0.1:${port}`, `http://localhost:${port}`].includes(req.headers.origin)) { res.writeHead(403); return res.end('Cross-origin access denied'); }
    const url = new URL(req.url, `http://127.0.0.1:${port}`);
    res.setHeader('X-Content-Type-Options', 'nosniff'); res.setHeader('Referrer-Policy', 'no-referrer');
    if (url.pathname === '/model-info') {
      res.setHeader('Content-Type', 'application/json'); res.setHeader('Cache-Control', 'no-store');
      return res.end(JSON.stringify(modelPath ? { url: '/model.gguf', name: modelName || 'Local GGUF', size: (await stat(modelPath)).size } : null));
    }
    const isModel = url.pathname === '/model.gguf';
    let file = isModel ? modelPath : resolve(root, '.' + decodeURIComponent(url.pathname === '/' ? '/index.html' : url.pathname));
    if (!file || (!isModel && !file.startsWith(root + sep))) { res.writeHead(404); return res.end(); }
    const info = await stat(file);
    if (!info.isFile()) { res.writeHead(404); return res.end(); }
    res.setHeader('Content-Type', isModel ? 'application/octet-stream' : types[extname(file)] || 'application/octet-stream');
    res.setHeader('Accept-Ranges', 'bytes');
    res.setHeader('Cache-Control', isModel ? 'no-store' : 'no-cache');
    let start = 0, end = info.size - 1;
    if (req.headers.range) {
      const m = /^bytes=(\d+)-(\d+)$/.exec(req.headers.range);
      if (!m || !Number.isSafeInteger(Number(m[1])) || !Number.isSafeInteger(Number(m[2])) || Number(m[1]) > Number(m[2]) || Number(m[2]) >= info.size || (isModel && Number(m[2]) - Number(m[1]) + 1 > 64 * 1024 ** 2)) {
        res.writeHead(416, { 'Content-Range': `bytes */${info.size}` }); return res.end();
      }
      start = Number(m[1]); end = Number(m[2]); res.statusCode = 206;
      res.setHeader('Content-Range', `bytes ${start}-${end}/${info.size}`);
    } else if (isModel && req.method !== 'HEAD') {
      res.writeHead(400); return res.end('Use HTTP Range requests, or select this GGUF file in the browser.');
    }
    res.setHeader('Content-Length', end - start + 1);
    if (req.method === 'HEAD') return res.end();
    await pipeline(createReadStream(file, { start, end }), res);
  } catch (e) { if (!res.headersSent) res.writeHead(e.code === 'ENOENT' ? 404 : 500); res.end(); }
});
server.listen(port, '127.0.0.1', () => {
  console.log(`WebGPU: http://127.0.0.1:${port}`);
  console.log(modelPath ? 'Serving local model bytes. All inference executes inside the browser.' : 'Select a GGUF file in the browser.');
});
