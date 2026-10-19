/** A random-access byte source. No operation materializes an entire model. */
export interface ByteSource {
  name: string;
  size: number;
  read(offset: number, length: number, signal?: AbortSignal): Promise<Uint8Array<ArrayBuffer>>;
}
export type SourceInput = File | { url: string; name?: string; size: number };
export function makeSource(input: SourceInput): ByteSource {
  const file = typeof File !== 'undefined' && input instanceof File;
  const descriptor = input as { url: string; name?: string; size: number };
  const size = input.size;
  if (!Number.isSafeInteger(size) || size < 24) throw new Error('Invalid model file size.');
  return {
    name: file ? (input as File).name : descriptor.name || 'local-model.gguf', size,
    async read(offset, length, signal) {
      signal?.throwIfAborted();
      if (!Number.isSafeInteger(offset) || !Number.isSafeInteger(length) || offset < 0 || length < 1 || offset + length > size || length > 64 * 1024 ** 2)
        throw new Error('Invalid model byte range (maximum read: 64 MiB).');
      if (file) {
        const bytes = new Uint8Array(await (input as File).slice(offset, offset + length).arrayBuffer());
        signal?.throwIfAborted();
        if (bytes.length !== length) throw new Error('The local file changed or could not be read.');
        return bytes;
      }
      const response = await fetch(descriptor.url, { headers: { Range: `bytes=${offset}-${offset + length - 1}` }, signal });
      if (response.status !== 206 || response.headers.get('Content-Range') !== `bytes ${offset}-${offset + length - 1}/${size}`) {
        await response.body?.cancel();
        throw new Error('The file server must support exact HTTP Range requests. Select the local GGUF file instead.');
      }
      const bytes = new Uint8Array(await response.arrayBuffer());
      if (bytes.length !== length) throw new Error('Truncated model range.');
      return bytes;
    },
  };
}
