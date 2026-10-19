/** Persistent, copyable API access key for the in-browser request bridge. */

export interface KeyStore { getItem(key: string): string | null; setItem(key: string, value: string): void }

export const API_KEY_ENTRY = 'webgpu.apiKey';

const API_KEY_PREFIX = 'wgpu_sk_';
const API_KEY_BYTES = 20;
const API_KEY_PATTERN = /^wgpu_sk_[0-9a-f]{40}$/;

const memory = new Map<string, string>();
const memoryStore: KeyStore = {
  getItem: key => memory.get(key) ?? null,
  setItem: (key, value) => { memory.set(key, value); },
};
/** A stable in-memory fallback for stores that deny reads or throw on quota. */
const fallbackStores = new WeakMap<KeyStore, KeyStore>();

function fallbackFor(store: KeyStore): KeyStore {
  let fallback = fallbackStores.get(store);
  if (!fallback) {
    const values = new Map<string, string>();
    fallback = { getItem: key => values.get(key) ?? null, setItem: (key, value) => { values.set(key, value); } };
    fallbackStores.set(store, fallback);
  }
  return fallback;
}

/** localStorage is absent in workers and can throw when access is denied by the browser. */
function defaultStore(): KeyStore {
  try {
    if (typeof localStorage !== 'undefined' && localStorage) return localStorage;
  } catch { /* Private mode or blocked storage: use memory. */ }
  return memoryStore;
}

function hex(bytes: Uint8Array): string {
  let value = '';
  for (const byte of bytes) value += byte.toString(16).padStart(2, '0');
  return value;
}

function strongRandom(): Uint8Array {
  const bytes = new Uint8Array(API_KEY_BYTES);
  try {
    if (typeof crypto !== 'undefined' && typeof crypto.getRandomValues === 'function') {
      crypto.getRandomValues(bytes);
      return bytes;
    }
  } catch { /* Fall back to Math.random when Web Crypto is unavailable. */ }
  for (let i = 0; i < bytes.length; i++) bytes[i] = Math.floor(Math.random() * 256);
  return bytes;
}

function isApiKey(value: unknown): value is string {
  return typeof value === 'string' && API_KEY_PATTERN.test(value);
}

function storedValue(store: KeyStore): { value: string | null; failed: boolean } {
  try { return { value: store.getItem(API_KEY_ENTRY), failed: false }; }
  catch { return { value: null, failed: true }; }
}

function persist(store: KeyStore, key: string): void {
  try {
    store.setItem(API_KEY_ENTRY, key);
    fallbackStores.delete(store);
  } catch {
    try { fallbackFor(store).setItem(API_KEY_ENTRY, key); } catch { /* Memory is the last resort. */ }
  }
}

export function generateApiKey(random?: (bytes: Uint8Array) => Uint8Array): string {
  const buffer = new Uint8Array(API_KEY_BYTES);
  if (random) {
    const supplied = random(buffer);
    const bytes = supplied instanceof Uint8Array && supplied.length >= API_KEY_BYTES ? supplied.subarray(0, API_KEY_BYTES) : buffer;
    return API_KEY_PREFIX + hex(bytes);
  }
  return API_KEY_PREFIX + hex(strongRandom());
}

/** Read the persisted key, creating and persisting one on first use. */
export function readApiKey(store: KeyStore = defaultStore()): string {
  const current = storedValue(store);
  if (!current.failed && isApiKey(current.value)) return current.value;

  // A missing value (or an unreadable store) may still have a key cached when
  // persistence failed earlier. A malformed value is always repaired.
  if (current.failed || current.value === null) {
    const fallback = fallbackFor(store);
    let cached: string | null = null;
    try { cached = fallback.getItem(API_KEY_ENTRY); } catch { /* Ignore a broken fallback. */ }
    if (isApiKey(cached)) {
      try { store.setItem(API_KEY_ENTRY, cached); } catch { /* Remain on the in-memory key. */ }
      return cached;
    }
  }
  const key = generateApiKey();
  persist(store, key);
  return key;
}

export function rotateApiKey(store: KeyStore = defaultStore()): string {
  const key = generateApiKey();
  persist(store, key);
  return key;
}

/** Keep at most four characters of the secret itself at each end. */
export function maskApiKey(key: string): string {
  if (typeof key !== 'string' || !key) return '…';
  const prefixed = key.startsWith(API_KEY_PREFIX);
  const secret = prefixed ? key.slice(API_KEY_PREFIX.length) : key;
  if (secret.length <= 8) return (prefixed ? API_KEY_PREFIX : '') + '…';
  return `${prefixed ? API_KEY_PREFIX : ''}${secret.slice(0, 4)}…${secret.slice(-4)}`;
}

/** Constant-time-ish comparison: every character and both lengths are always inspected. */
export function keysMatch(expected: string, provided: unknown): boolean {
  if (typeof expected !== 'string' || typeof provided !== 'string') return false;
  const length = Math.max(expected.length, provided.length);
  let difference = expected.length ^ provided.length;
  for (let i = 0; i < length; i++) {
    const left = i < expected.length ? expected.charCodeAt(i) : 0;
    const right = i < provided.length ? provided.charCodeAt(i) : 0;
    difference |= left ^ right;
  }
  return difference === 0;
}
