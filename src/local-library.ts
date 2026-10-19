import './local-library.css';
import { makeSource, type SourceInput } from './core/source.ts';
import { peekGGUF } from './core/gguf.ts';

/** One row in the local library. Weights are never held in memory here; a folder
 * library keeps the file handle and asks for a fresh File only when it is needed. */
export interface LibraryEntry {
  /** `sha256:<hex>` for an Ollama blob, or `file:<relative path>` for a plain GGUF. */
  digest: string;
  name: string;
  size: number;
  architecture: string;
  quantization: string;
  supported: boolean;
  reason: string;
  handle?: FileSystemFileHandle;
  /** Only the no-File-System-Access fallback stores the File: there is no handle to keep. */
  file?: File;
}

const DB_NAME = 'webgpu-library';
const STORE_NAME = 'handles';
const SAVED_KEY = 'models';
const MANIFEST_DIRS = ['manifests', 'manifests-v2'] as const;
const MODEL_MEDIA_TYPE = 'application/vnd.ollama.image.model';
const DIGEST_PATTERN = /^sha256:[a-f0-9]{64}$/;
const SUPPORTED_ARCHITECTURES = new Set(['gemma3', 'qwen35', 'llama', 'qwen2', 'qwen3']);
const MAX_ENTRIES = 2000;
const MAX_MANIFEST_PARTS = 8;
const MAX_MANIFEST_BYTES = 1024 ** 2;
const MAX_GGUF_DEPTH = 5;
const MAX_READS = 2;

/** Ollama manifest paths are `<registry>/<namespace>/<name…>/<tag>`. The default
 * registry and library namespace are implicit, exactly as server/ollama.mjs lists them. */
export function modelNameFromManifestPath(pathParts: string[]): string {
  if (pathParts.length < 3 || pathParts.some(part => typeof part !== 'string' || !part)) throw new Error('Invalid Ollama manifest path.');
  const [registry, namespace, ...tail] = pathParts;
  const tag = tail.pop()!;
  const base = tail.join('/');
  return `${registry === 'registry.ollama.ai' ? '' : registry + '/'}${namespace === 'library' ? '' : namespace + '/'}${base}:${tag}`;
}

/** Finds the local weight layer. Cloud-only manifests have no model layer and are skipped. */
export function manifestModelLayer(manifest: unknown): { digest: string; size: number } | undefined {
  if (!manifest || typeof manifest !== 'object') return undefined;
  const layers = (manifest as { layers?: unknown }).layers;
  if (!Array.isArray(layers)) return undefined;
  const layer = layers.find(candidate => !!candidate && typeof candidate === 'object' && (candidate as { mediaType?: unknown }).mediaType === MODEL_MEDIA_TYPE);
  if (!layer || typeof layer !== 'object') return undefined;
  const { digest, size } = layer as { digest?: unknown; size?: unknown };
  if (typeof digest !== 'string' || !DIGEST_PATTERN.test(digest) || !Number.isSafeInteger(size) || (size as number) < 0) return undefined;
  return { digest, size: size as number };
}

/** `sha256:<hex>` in a manifest names the blob `sha256-<hex>` on disk. */
export function blobNameForDigest(digest: string): string {
  if (!DIGEST_PATTERN.test(digest)) throw new Error('Invalid Ollama model digest.');
  return digest.replace(':', '-');
}

/** The browser engine supports exactly these GGUF architectures. */
export function isSupportedArchitecture(architecture: string): boolean {
  return SUPPORTED_ARCHITECTURES.has(architecture);
}

function cleanerName(candidate: string, current: string): boolean {
  return candidate.length < current.length || (candidate.length === current.length && candidate.localeCompare(current) < 0);
}

/** One digest can be tagged under several names. Keep the least-qualified name,
 * then list supported models first. */
export function mergeModelEntries(entries: LibraryEntry[]): LibraryEntry[] {
  const merged = new Map<string, LibraryEntry>();
  for (const entry of entries) {
    const current = merged.get(entry.digest);
    if (!current) { merged.set(entry.digest, entry); continue; }
    if (!cleanerName(entry.name, current.name)) continue;
    if (!entry.handle && !entry.file && (current.handle || current.file)) {
      merged.set(entry.digest, {
        ...entry, handle: current.handle, file: current.file, size: current.size,
        supported: current.supported, reason: current.reason,
        architecture: current.architecture || entry.architecture, quantization: current.quantization || entry.quantization,
      });
    } else merged.set(entry.digest, entry);
  }
  return [...merged.values()].sort((a, b) => Number(b.supported) - Number(a.supported) || a.name.localeCompare(b.name));
}

function isPending(entry: LibraryEntry): boolean {
  return entry.supported && !entry.architecture && !entry.reason;
}

function formatSize(bytes: number): string {
  if (!Number.isSafeInteger(bytes) || bytes < 0) return '';
  if (bytes === 0) return '0 B';
  if (bytes >= 1024 ** 3) return `${(bytes / 1024 ** 3).toFixed(2)} GiB`;
  if (bytes >= 1024 ** 2) return `${(bytes / 1024 ** 2).toFixed(1)} MiB`;
  if (bytes < 1024) return `${bytes} B`;
  return `${Math.round(bytes / 1024)} KiB`;
}

function errorMessage(error: unknown, fallback: string): string {
  return error instanceof Error && error.message ? error.message : typeof error === 'string' && error ? error : fallback;
}

function isAbortError(error: unknown): boolean {
  return !!error && typeof error === 'object' && 'name' in error && (error as { name?: unknown }).name === 'AbortError';
}

function isNotFound(error: unknown): boolean {
  return !!error && typeof error === 'object' && 'name' in error && (error as { name?: unknown }).name === 'NotFoundError';
}

interface DiscoveredFile { parts: string[]; handle: FileSystemFileHandle }
interface ReadPermissionDescription { mode: 'read' }
interface PermissionHandle extends FileSystemHandle {
  queryPermission?(description?: ReadPermissionDescription): Promise<PermissionState>;
  requestPermission?(description?: ReadPermissionDescription): Promise<PermissionState>;
}

async function permissionState(handle: FileSystemDirectoryHandle, request: boolean): Promise<PermissionState> {
  const permitted = handle as unknown as PermissionHandle;
  const description: ReadPermissionDescription = { mode: 'read' };
  const query = async (): Promise<PermissionState> => {
    if (typeof permitted.queryPermission !== 'function') return 'granted';
    try { return await permitted.queryPermission(description); }
    catch { try { return await permitted.queryPermission(); } catch { return 'prompt'; } }
  };
  const current = await query();
  if (current !== 'prompt' || !request || typeof permitted.requestPermission !== 'function') return current;
  try { return await permitted.requestPermission(description); }
  catch { try { return await permitted.requestPermission(); } catch { return 'prompt'; } }
}

function libraryDatabase(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open(DB_NAME, 1);
    request.onupgradeneeded = () => { if (!request.result.objectStoreNames.contains(STORE_NAME)) request.result.createObjectStore(STORE_NAME); };
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error ?? new Error('Could not open the model library database.'));
  });
}

async function readSavedDirectory(): Promise<FileSystemDirectoryHandle | undefined> {
  if (typeof indexedDB === 'undefined') return undefined;
  let database: IDBDatabase | undefined;
  try {
    database = await libraryDatabase();
    const value = await new Promise<unknown>((resolve, reject) => {
      const request = database!.transaction(STORE_NAME, 'readonly').objectStore(STORE_NAME).get(SAVED_KEY);
      request.onsuccess = () => resolve(request.result);
      request.onerror = () => reject(request.error ?? new Error('Could not read the saved model folder.'));
    });
    if (!value || typeof value !== 'object') return undefined;
    const handle = value as Partial<FileSystemDirectoryHandle>;
    return handle.kind === 'directory' && typeof handle.name === 'string' ? value as FileSystemDirectoryHandle : undefined;
  } catch { return undefined; }
  finally { database?.close(); }
}

async function saveSavedDirectory(handle: FileSystemDirectoryHandle): Promise<void> {
  if (typeof indexedDB === 'undefined') return;
  const database = await libraryDatabase();
  try {
    await new Promise<void>((resolve, reject) => {
      const request = database.transaction(STORE_NAME, 'readwrite').objectStore(STORE_NAME).put(handle, SAVED_KEY);
      request.onsuccess = () => resolve();
      request.onerror = () => reject(request.error ?? new Error('Could not save the model folder.'));
    });
  } finally { database.close(); }
}

async function walkManifests(directory: FileSystemDirectoryHandle, parts: string[], signal: AbortSignal, found: DiscoveredFile[]): Promise<void> {
  if (parts.length > MAX_MANIFEST_PARTS || found.length >= MAX_ENTRIES) return;
  for await (const [name, handle] of directory.entries()) {
    signal.throwIfAborted();
    if (handle.kind === 'directory') await walkManifests(handle, [...parts, name], signal, found);
    else found.push({ parts: [...parts, name], handle });
    if (found.length >= MAX_ENTRIES) return;
  }
}

async function walkGGUFs(directory: FileSystemDirectoryHandle, parts: string[], signal: AbortSignal, found: DiscoveredFile[]): Promise<void> {
  for await (const [name, handle] of directory.entries()) {
    signal.throwIfAborted();
    if (found.length >= MAX_ENTRIES) return;
    const next = [...parts, name];
    if (handle.kind === 'directory') {
      if (next.length < MAX_GGUF_DEPTH) await walkGGUFs(handle, next, signal, found);
    } else if (/\.gguf$/i.test(name) && next.length <= MAX_GGUF_DEPTH) {
      found.push({ parts: next, handle });
    }
  }
}

async function manifestEntry(handle: FileSystemFileHandle, parts: string[], blobs: FileSystemDirectoryHandle | undefined, signal: AbortSignal): Promise<LibraryEntry | undefined> {
  try {
    const manifestFile = await handle.getFile();
    if (manifestFile.size > MAX_MANIFEST_BYTES) return undefined;
    const manifest = JSON.parse(await manifestFile.text());
    const layer = manifestModelLayer(manifest);
    if (!layer) return undefined;
    let weights: FileSystemFileHandle | undefined;
    if (blobs) {
      try { weights = await blobs.getFileHandle(blobNameForDigest(layer.digest)); }
      catch (error) { if (!isNotFound(error)) throw error; }
    }
    signal.throwIfAborted();
    return {
      digest: layer.digest, name: modelNameFromManifestPath(parts), size: layer.size,
      architecture: '', quantization: '', supported: !!weights,
      reason: weights ? '' : 'The weight file is missing from the blobs folder.', handle: weights,
    };
  } catch (error) {
    if (signal.aborted) throw error;
    return undefined; // A malformed or partially written manifest must not hide other models.
  }
}

async function ggufEntry(handle: FileSystemFileHandle, parts: string[], signal: AbortSignal): Promise<LibraryEntry | undefined> {
  try {
    const file = await handle.getFile();
    signal.throwIfAborted();
    return {
      digest: 'file:' + parts.join('/'), name: parts[parts.length - 1] ?? file.name, size: file.size,
      architecture: '', quantization: '', supported: true, reason: '', handle,
    };
  } catch (error) {
    if (signal.aborted) throw error;
    return undefined;
  }
}

async function discoverDirectory(root: FileSystemDirectoryHandle, signal: AbortSignal): Promise<LibraryEntry[]> {
  const entries: LibraryEntry[] = [];
  let blobs: FileSystemDirectoryHandle | undefined;
  try { blobs = await root.getDirectoryHandle('blobs'); }
  catch (error) { if (!isNotFound(error)) throw error; }
  for (const manifestDir of MANIFEST_DIRS) {
    let directory: FileSystemDirectoryHandle;
    try { directory = await root.getDirectoryHandle(manifestDir); }
    catch (error) { if (isNotFound(error)) continue; throw error; }
    const manifests: DiscoveredFile[] = [];
    await walkManifests(directory, [], signal, manifests);
    for (const item of manifests) {
      signal.throwIfAborted();
      if (item.parts.length < 3) continue;
      if (entries.length >= MAX_ENTRIES) break;
      const entry = await manifestEntry(item.handle, item.parts, blobs, signal);
      if (entry) entries.push(entry);
    }
  }
  const files: DiscoveredFile[] = [];
  await walkGGUFs(root, [], signal, files);
  for (const item of files) {
    signal.throwIfAborted();
    if (entries.length >= MAX_ENTRIES) break;
    const entry = await ggufEntry(item.handle, item.parts, signal);
    if (entry) entries.push(entry);
  }
  return mergeModelEntries(entries);
}

function pathSegments(file: File): string[] {
  return (file.webkitRelativePath || file.name).split('/').filter(Boolean);
}

function relativeSegments(file: File): string[] {
  const segments = pathSegments(file);
  return file.webkitRelativePath ? segments.slice(1) : segments;
}

function manifestParts(segments: string[]): string[] | undefined {
  const index = segments.findIndex(segment => segment === 'manifests' || segment === 'manifests-v2');
  return index < 0 ? undefined : segments.slice(index + 1);
}

async function discoverFiles(files: File[], signal: AbortSignal): Promise<LibraryEntry[]> {
  const entries: LibraryEntry[] = [];
  const byName = new Map<string, File>();
  const byPath = new Map<string, File>();
  for (const file of files) {
    byName.set(file.name, file);
    byPath.set(pathSegments(file).join('/'), file);
    byPath.set(relativeSegments(file).join('/'), file);
  }
  const findBlob = (name: string, segments: string[]): File | undefined => {
    const index = segments.findIndex(segment => segment === 'manifests' || segment === 'manifests-v2');
    const prefix = index > 0 ? segments.slice(0, index) : [];
    return byName.get(name)
      ?? byPath.get([...prefix, 'blobs', name].join('/'))
      ?? byPath.get(['blobs', name].join('/'));
  };
  for (const file of files) {
    signal.throwIfAborted();
    if (entries.length >= MAX_ENTRIES) break;
    const segments = relativeSegments(file);
    const fullSegments = pathSegments(file);
    if (/\.gguf$/i.test(file.name) && segments.length <= MAX_GGUF_DEPTH) {
      entries.push({ digest: 'file:' + segments.join('/'), name: file.name, size: file.size, architecture: '', quantization: '', supported: true, reason: '', file });
      continue;
    }
    const parts = manifestParts(fullSegments);
    if (parts) {
      if (parts.length < 3 || file.size > MAX_MANIFEST_BYTES) continue;
      try {
        const manifest = JSON.parse(await file.text());
        const layer = manifestModelLayer(manifest);
        if (!layer) continue;
        const blob = findBlob(blobNameForDigest(layer.digest), fullSegments);
        const incomplete = !!blob && blob.size !== layer.size;
        entries.push({
          digest: layer.digest, name: modelNameFromManifestPath(parts), size: blob?.size ?? layer.size,
          architecture: '', quantization: '', supported: !!blob && !incomplete,
          reason: !blob ? 'The weights are missing from the folder.' : incomplete ? `The weights are ${formatSize(blob.size)} but the manifest declares ${formatSize(layer.size)}.` : '',
          file: blob,
        });
      } catch { /* Ignore malformed manifests. */ }
      continue;
    }
  }
  return mergeModelEntries(entries);
}

type DirectoryPicker = (options?: { id?: string; mode?: 'read' | 'readwrite' }) => Promise<FileSystemDirectoryHandle>;
type DirectoryWindow = Window & { showDirectoryPicker?: DirectoryPicker };

export function createLibraryPicker(load: (source: SourceInput) => Promise<void>, chooseFile: () => void, isBusy: () => boolean) {
  const dialog = document.createElement('dialog');
  dialog.className = 'library-dialog';
  dialog.setAttribute('aria-labelledby', 'library-title');
  dialog.setAttribute('aria-describedby', 'library-intro');
  dialog.innerHTML = `<div class="library-heading"><div><span class="eyebrow"><span></span>ON YOUR DISK</span><h2 id="library-title">Open local models</h2></div><button type="button" class="text-button" id="library-close" aria-label="Close local model library">✕</button></div>
    <p class="library-intro" id="library-intro">Open an Ollama model folder, or any folder of GGUF files. Model weights are read in this tab and never uploaded.</p>
    <div class="library-toolbar"><input id="library-search" type="search" placeholder="Search by name, architecture or quantization…" aria-label="Search local models" autocomplete="off" spellcheck="false"><button type="button" class="secondary" id="library-saved" hidden></button><button type="button" class="primary" id="library-choose">Choose folder</button><button type="button" class="secondary" id="library-rescan" disabled>Rescan</button></div>
    <p id="library-status" role="status" aria-live="polite"></p>
    <div id="library-list" class="library-list" aria-label="Local models"></div>
    <div id="library-empty" class="library-empty"><div class="library-empty-symbol" aria-hidden="true">▤</div><h3>Your models, straight from disk</h3><p>Choose the folder where Ollama stores its models, or any folder with GGUF files. Nothing is uploaded and Ollama can stay closed.</p><p class="field-help" id="library-empty-help">With the File System Access API, the folder is remembered and can be reopened with one click. Other browsers can still browse a folder.</p></div>
    <div class="library-footer"><span>Files stay on this computer.</span><button type="button" class="text-button" id="library-file">Choose a single GGUF file instead</button></div>
    <input id="library-directory" type="file" webkitdirectory multiple hidden tabindex="-1" aria-hidden="true">`;
  document.body.append(dialog);
  const $ = <T extends HTMLElement = HTMLElement>(selector: string) => dialog.querySelector<T>(selector)!;
  const search = $<HTMLInputElement>('#library-search');
  const status = $('#library-status');
  const list = $('#library-list');
  const empty = $('#library-empty');
  const emptyHelp = $('#library-empty-help');
  const savedButton = $<HTMLButtonElement>('#library-saved');
  const rescanButton = $<HTMLButtonElement>('#library-rescan');
  const chooseButton = $<HTMLButtonElement>('#library-choose');
  const directoryInput = $<HTMLInputElement>('#library-directory');
  const pickerWindow = window as unknown as DirectoryWindow;
  const picker = pickerWindow.showDirectoryPicker;
  const supportsFolders = typeof picker === 'function';
  const showDirectoryPicker = supportsFolders ? (picker as DirectoryPicker).bind(window) : undefined;
  let models: LibraryEntry[] = [];
  let rows = new Map<string, HTMLButtonElement>();
  let queue: LibraryEntry[] = [];
  let activeReads = 0;
  let scanning = false;
  let scanned = false;
  let message = '';
  let messageError = false;
  let generation = 0;
  let scanController: AbortController | undefined;
  let currentHandle: FileSystemDirectoryHandle | undefined;
  let savedHandle: FileSystemDirectoryHandle | undefined;
  let lastFiles: File[] = [];
  let lastFolderName = '';

  function setMessage(text: string, error = false): void {
    message = text; messageError = error && !!text;
  }

  const visibleModels = (): LibraryEntry[] => {
    const query = search.value.trim().toLowerCase();
    return query ? models.filter(entry => `${entry.name} ${entry.architecture} ${entry.quantization}`.toLowerCase().includes(query)) : models;
  };

  function updateControls(): void {
    rescanButton.disabled = scanning || (!currentHandle && !lastFiles.length);
    list.setAttribute('aria-busy', String(scanning));
  }

  function updateStatus(): void {
    status.dataset.state = message ? (messageError ? 'error' : 'info') : '';
    if (message) { status.textContent = message; return; }
    if (scanning) { status.textContent = lastFolderName ? `Scanning “${lastFolderName}”…` : 'Scanning your folder…'; return; }
    const total = models.length;
    const visible = visibleModels();
    if (!total) { status.textContent = scanned && lastFolderName ? `0 models · no GGUF files or Ollama manifests found in “${lastFolderName}”.` : ''; return; }
    if (search.value.trim()) { status.textContent = `${visible.length} of ${total} model${total === 1 ? '' : 's'} ${total === 1 ? 'matches' : 'match'} your search.`; return; }
    const pending = models.filter(isPending).length;
    const unavailable = models.filter(entry => !entry.supported).length;
    const count = `${total} model${total === 1 ? '' : 's'}`;
    status.textContent = pending
      ? `${count} · checking ${pending} for compatibility…`
      : unavailable ? `${count} · ${total - unavailable} ready · ${unavailable} unavailable` : `${count} · select one to open`;
  }

  function rowElement(entry: LibraryEntry): HTMLButtonElement {
    let row = rows.get(entry.digest);
    if (!row) {
      row = document.createElement('button');
      row.type = 'button';
      row.className = 'library-row';
      row.innerHTML = '<span class="library-row-main"><strong class="library-name"></strong><span class="library-meta"></span><span class="library-hint"></span></span><span class="library-state"><span class="subtle-badge library-badge"></span><span class="library-action" aria-hidden="true"></span></span>';
      row.onclick = () => {
        const current = models.find(model => model.digest === entry.digest);
        if (current) void openEntry(current);
      };
      rows.set(entry.digest, row);
    }
    const pending = isPending(entry);
    const unavailable = !entry.supported;
    row.disabled = unavailable;
    row.dataset.state = pending ? 'checking' : entry.supported ? 'supported' : 'unsupported';
    row.setAttribute('aria-label', pending
      ? `Open ${entry.name}. Compatibility is still being checked.`
      : entry.supported ? `Open ${entry.name}${entry.architecture ? ` (${entry.architecture}, ${entry.quantization})` : ''}.` : `Unavailable ${entry.name}. ${entry.reason}`);
    const name = row.querySelector<HTMLElement>('.library-name')!, meta = row.querySelector<HTMLElement>('.library-meta')!, hint = row.querySelector<HTMLElement>('.library-hint')!;
    const badge = row.querySelector<HTMLElement>('.library-badge')!, action = row.querySelector<HTMLElement>('.library-action')!;
    name.textContent = entry.name;
    meta.textContent = [formatSize(entry.size), entry.architecture, entry.quantization].filter(Boolean).join(' · ');
    hint.textContent = pending ? 'Checking architecture and quantization…' : entry.supported ? 'Ready to open in your browser' : entry.reason;
    badge.textContent = pending ? 'Checking…' : entry.supported ? 'Supported' : entry.architecture ? 'Unsupported' : 'Unavailable';
    badge.dataset.state = pending ? 'checking' : entry.supported ? 'supported' : 'unsupported';
    action.textContent = pending ? '…' : entry.supported ? '↗' : '';
    return row;
  }

  function render(): void {
    const visible = visibleModels();
    const shown = new Set(visible.map(entry => entry.digest));
    for (const [digest, row] of rows) { if (!shown.has(digest)) { row.remove(); rows.delete(digest); } }
    for (const [index, entry] of visible.entries()) {
      const next = rowElement(entry);
      if (list.children[index] !== next) list.insertBefore(next, list.children[index] ?? null);
    }
    empty.hidden = models.length > 0 || scanning;
    updateControls();
    updateStatus();
  }

  function beginScan(label: string): { id: number; signal: AbortSignal } {
    scanController?.abort();
    scanController = new AbortController();
    generation++;
    queue = [];
    scanning = true;
    scanned = false;
    models = [];
    setMessage('');
    lastFolderName = label;
    render();
    return { id: generation, signal: scanController.signal };
  }

  async function openEntry(entry: LibraryEntry): Promise<void> {
    if (isBusy() || !entry.supported) return;
    try {
      const file = entry.file ?? (entry.handle ? await entry.handle.getFile() : undefined);
      if (!file) throw new Error(entry.reason || 'The model file is no longer available.');
      if (isBusy()) return;
      dialog.close();
      void load(file);
    } catch (error) {
      setMessage(errorMessage(error, 'The model file could not be opened.'), true);
      render();
    }
  }

  async function inspectEntry(entry: LibraryEntry, signal: AbortSignal): Promise<void> {
    try {
      const file = entry.file ?? (entry.handle ? await entry.handle.getFile() : undefined);
      if (!file) throw new Error(entry.reason || 'The model file is no longer available.');
      signal.throwIfAborted();
      if (entry.digest.startsWith('sha256:') && entry.size !== file.size) throw new Error(`The weights on disk are ${formatSize(file.size)}, but the manifest declares ${formatSize(entry.size)}. The model is incomplete.`);
      const peek = await peekGGUF(makeSource(file), signal);
      signal.throwIfAborted();
      entry.architecture = peek.architecture;
      entry.quantization = peek.quantization;
      entry.supported = isSupportedArchitecture(peek.architecture);
      entry.reason = entry.supported ? '' : `The ${peek.architecture} architecture is not supported by this browser runtime.`;
      if (entry.digest.startsWith('sha256:')) entry.size = file.size;
    } catch (error) {
      if (signal.aborted) return;
      entry.architecture = '';
      entry.quantization = '';
      entry.supported = false;
      entry.reason = errorMessage(error, 'This model could not be inspected.');
    }
    render();
  }

  function pumpPeeks(id: number, signal: AbortSignal): void {
    while (activeReads < MAX_READS && queue.length && id === generation && !signal.aborted) {
      const entry = queue.shift()!;
      activeReads++;
      void inspectEntry(entry, signal).finally(() => {
        activeReads--;
        if (id !== generation) return;
        const current = scanController;
        if (current && !current.signal.aborted) pumpPeeks(id, current.signal);
      });
    }
  }

  function resumePeeks(): void {
    if (!queue.length) queue = models.filter(isPending);
    if (!queue.length) return;
    if (!scanController) scanController = new AbortController();
    pumpPeeks(generation, scanController.signal);
  }

  async function scanHandle(handle: FileSystemDirectoryHandle, label = handle.name): Promise<void> {
    const { id, signal } = beginScan(label);
    currentHandle = handle;
    lastFiles = [];
    try {
      const entries = await discoverDirectory(handle, signal);
      if (signal.aborted || id !== generation) return;
      models = entries;
      scanned = true;
      scanning = false;
      queue = models.filter(isPending);
      render();
      pumpPeeks(id, signal);
    } catch (error) {
      if (signal.aborted || id !== generation) return;
      scanning = false;
      models = [];
      setMessage(isNotFound(error) ? `“${label}” is no longer available. Choose the folder again.` : errorMessage(error, 'The folder could not be read.'), true);
      render();
    } finally {
      if (id === generation) {
        scanning = false;
        if (dialog.open) render();
        else updateControls();
      }
    }
  }

  async function scanFiles(files: File[], label: string): Promise<void> {
    const { id, signal } = beginScan(label);
    currentHandle = undefined;
    lastFiles = files.slice();
    try {
      const entries = await discoverFiles(files, signal);
      if (signal.aborted || id !== generation) return;
      models = entries;
      scanned = true;
      scanning = false;
      queue = models.filter(isPending);
      render();
      pumpPeeks(id, signal);
    } catch (error) {
      if (signal.aborted || id !== generation) return;
      scanning = false;
      models = [];
      setMessage(errorMessage(error, 'The selected folder could not be read.'), true);
      render();
    } finally {
      if (id === generation) {
        scanning = false;
        if (dialog.open) render();
        else updateControls();
      }
    }
  }

  function rememberFolder(handle: FileSystemDirectoryHandle): void {
    savedHandle = handle;
    savedButton.hidden = false;
    savedButton.textContent = `Reopen “${handle.name}”`;
    savedButton.setAttribute('aria-label', `Reopen your saved model folder ${handle.name}`);
    savedButton.title = `Open your saved model folder ${handle.name}`;
    void saveSavedDirectory(handle).catch(() => { /* Private browsing can disable IndexedDB. */ });
  }

  async function chooseFolder(): Promise<void> {
    if (isBusy()) return;
    if (!showDirectoryPicker) {
      directoryInput.value = '';
      directoryInput.click();
      return;
    }
    try {
      const handle = await showDirectoryPicker({ id: 'webgpu-models', mode: 'read' });
      if (!dialog.open) return;
      rememberFolder(handle);
      currentHandle = handle;
      await scanHandle(handle);
    } catch (error) {
      if (isAbortError(error) || !dialog.open) return;
      setMessage(errorMessage(error, 'The folder could not be opened. Choose it again.'), true);
      render();
    }
  }

  async function reopenSaved(): Promise<void> {
    if (isBusy() || !savedHandle) return;
    try {
      const state = await permissionState(savedHandle, true);
      if (!dialog.open) return;
      if (state !== 'granted') {
        setMessage(state === 'denied' ? `Read access to “${savedHandle.name}” was denied. Choose the folder again and allow access.` : `Read access to “${savedHandle.name}” was not granted. Choose the folder again.`, true);
        render();
        return;
      }
      currentHandle = savedHandle;
      await scanHandle(savedHandle);
    } catch (error) {
      setMessage(errorMessage(error, 'The saved folder could not be opened. Choose it again.'), true);
      render();
    }
  }

  async function rescan(): Promise<void> {
    if (isBusy() || scanning) return;
    if (currentHandle) {
      const handle = currentHandle;
      try {
        const state = await permissionState(handle, true);
        if (!dialog.open) return;
        if (state !== 'granted') {
          setMessage(`Read access to “${handle.name}” is required to rescan. Choose the folder again.`, true);
          render();
          return;
        }
        await scanHandle(handle);
      } catch (error) {
        setMessage(errorMessage(error, 'The folder could not be rescanned.'), true);
        render();
      }
    } else if (lastFiles.length) {
      await scanFiles(lastFiles, lastFolderName || 'your folder');
    }
  }

  async function restoreSaved(): Promise<void> {
    if (!supportsFolders) return;
    const handle = await readSavedDirectory();
    if (!handle || !dialog.open) return;
    savedHandle = handle;
    savedButton.hidden = false;
    savedButton.textContent = `Reopen “${handle.name}”`;
    savedButton.setAttribute('aria-label', `Reopen your saved model folder ${handle.name}`);
    savedButton.title = `Open your saved model folder ${handle.name}`;
    const state = await permissionState(handle, false);
    if (!dialog.open) return;
    if (state === 'granted') { currentHandle = handle; await scanHandle(handle); }
    else if (state === 'denied') {
      setMessage(`Read access to “${handle.name}” is blocked. Choose the folder again and allow access.`, true);
      render();
    } else {
      setMessage(`Access to “${handle.name}” needs to be granted again. Choose “Reopen” or pick the folder again.`, false);
      render();
    }
  }

  $('#library-close').onclick = () => dialog.close();
  chooseButton.onclick = () => { void chooseFolder(); };
  savedButton.onclick = () => { void reopenSaved(); };
  rescanButton.onclick = () => { void rescan(); };
  search.oninput = () => render();
  directoryInput.addEventListener('change', () => {
    const files = Array.from(directoryInput.files ?? []);
    directoryInput.value = '';
    if (!files.length || isBusy()) return;
    const label = files[0].webkitRelativePath.split('/')[0] || 'your folder';
    void scanFiles(files, label);
  });
  $('#library-file').onclick = () => { if (isBusy()) return; dialog.close(); chooseFile(); };
  dialog.addEventListener('close', () => {
    scanController?.abort();
    scanController = undefined;
    queue = [];
  });

  if (!supportsFolders) {
    emptyHelp.textContent = 'This browser cannot remember a folder between visits. Use “Choose folder” to pick it each time; models are still read locally and nothing is uploaded.';
  }

  return () => {
    if (isBusy()) return;
    search.value = '';
    if (!dialog.open) dialog.showModal();
    search.focus();
    if (models.length) {
      render();
      if (models.some(isPending)) resumePeeks();
      return;
    }
    if (scanning) { render(); return; }
    setMessage(supportsFolders ? '' : 'This browser does not support the File System Access API. Use “Choose folder” to browse a folder; models still open locally and nothing is uploaded.');
    render();
    if (supportsFolders) void restoreSaved();
  };
}
