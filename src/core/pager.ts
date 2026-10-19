import { Compute, MiB } from './gpu.ts';
import type { ByteSource } from './source.ts';
import type { Tensor } from './gguf.ts';

export interface PagingStats { readBytes: number; readMs: number; uploads: number; cacheHits: number; cacheBytes: number; gpuBytes: number; peakGpuBytes: number; pageBytes: number }
/** Stable resident prefix + one reusable streaming slot avoids LRU scan thrashing. */
export class WeightPager {
  private cache = new Map<string, GPUBuffer>();
  private scratch: GPUBuffer;
  private cacheBudget: number;
  readonly pageBytes: number;
  stats: PagingStats = { readBytes: 0, readMs: 0, uploads: 0, cacheHits: 0, cacheBytes: 0, gpuBytes: 0, peakGpuBytes: 0, pageBytes: 0 };
  private gpu: Compute;
  private source: ByteSource;
  constructor(gpu: Compute, source: ByteSource, pageSizeMiB = 16) {
    this.gpu = gpu; this.source = source;
    if (![16, 64].includes(pageSizeMiB)) throw new Error('Weight pages must be 16 or 64 MiB.');
    this.pageBytes = Math.min(pageSizeMiB * MiB, gpu.device.limits.maxStorageBufferBindingSize, gpu.device.limits.maxBufferSize);
    this.stats.pageBytes = this.pageBytes;
    this.scratch = gpu.buffer(this.pageBytes, 'streaming weight page');
    // Leave headroom for readback and transient uniforms. Driver allocations are outside this counter.
    this.cacheBudget = Math.max(0, gpu.limit - gpu.allocated - 16 * MiB);
  }
  rowsPerPage(t: Tensor) { return Math.max(1, Math.min(this.gpu.device.limits.maxComputeWorkgroupsPerDimension, Math.floor(this.pageBytes / t.rowBytes))); }
  async page(t: Tensor, row: number, count: number, signal?: AbortSignal): Promise<GPUBuffer> {
    signal?.throwIfAborted();
    const key = `${t.name}:${row}:${count}`, hit = this.cache.get(key);
    if (hit) { this.stats.cacheHits++; return hit; }
    const length = count * t.rowBytes;
    if (length > this.pageBytes) throw new Error(`One row of ${t.name} exceeds the paging slot.`);
    const started = performance.now();
    const data = await this.source.read(t.offset + row * t.rowBytes, length, signal);
    this.stats.readMs += performance.now() - started;
    this.stats.readBytes += length;
    const padded = length % 4 ? new Uint8Array(Math.ceil(length / 4) * 4) : data;
    if (padded !== data) padded.set(data);
    let buffer = this.scratch;
    if (this.stats.cacheBytes + padded.length <= this.cacheBudget) {
      buffer = this.gpu.buffer(padded.length, `resident ${key}`);
      this.cache.set(key, buffer); this.stats.cacheBytes += padded.length;
    }
    this.gpu.write(buffer, padded); this.stats.uploads++;
    return buffer;
  }
  snapshot(): PagingStats { return { ...this.stats, gpuBytes: this.gpu.allocated, peakGpuBytes: this.gpu.peak }; }
  dispose() { for (const b of this.cache.values()) this.gpu.destroy(b); this.cache.clear(); this.gpu.destroy(this.scratch); }
}
