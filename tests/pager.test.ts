import { test } from 'node:test';
import assert from 'node:assert/strict';
import { WeightPager } from '../src/core/pager.ts';
import { Compute, MiB } from '../src/core/gpu.ts';
import { PagedEngine } from '../src/core/engine.ts';
import type { Tensor } from '../src/core/gguf.ts';

function gpuFixture(limitMiB = 256, bindingMiB = 128) {
  const sizes = new Map<object, number>();
  const gpu = {
    allocated: 32 * MiB, peak: 32 * MiB, limit: limitMiB * MiB,
    device: { limits: { maxStorageBufferBindingSize: bindingMiB * MiB, maxBufferSize: bindingMiB * MiB, maxComputeWorkgroupsPerDimension: 65535 } },
    buffer(size: number) { assert.ok(gpu.allocated + size <= gpu.limit, 'GPU allocation stays within budget'); const b = { size, first: -1 }; sizes.set(b, size); gpu.allocated += size; gpu.peak = Math.max(gpu.peak, gpu.allocated); return b; },
    write(b: { size: number; first: number }, data: Uint8Array) { assert.ok(data.length <= b.size); b.first = data[0]; },
    destroy(b: object) { const size = sizes.get(b); if (size !== undefined) { gpu.allocated -= size; sizes.delete(b); } },
  };
  return gpu;
}
const tensor: Tensor = { name: 'test.weight', dims: [256, 200000], type: 0, offset: 0, rowBytes: 1024, bytes: 200000 * 1024 };
test('64 MiB pages preserve cached weights, avoid reads on cache hits, and keep streaming inside the GPU budget', async () => {
  const gpu = gpuFixture(); let reads = 0;
  const source = { name: 'fixture', size: tensor.bytes, async read(offset: number, length: number) { reads++; assert.ok(length <= 64 * MiB); const data = new Uint8Array(length); data[0] = offset % 251; return data; } };
  const pager = new WeightPager(gpu as unknown as Compute, source, 64), rows = pager.rowsPerPage(tensor);
  const first = await pager.page(tensor, 0, rows);
  await pager.page(tensor, rows, rows);
  const streaming = await pager.page(tensor, 2 * rows, rows);
  assert.notEqual(first, streaming);
  assert.equal(await pager.page(tensor, 0, rows), first); assert.equal(reads, 3);
  assert.equal((first as unknown as { first: number }).first, 0);
  assert.equal(await pager.page(tensor, 2 * rows, rows), streaming); assert.equal(reads, 4);
  assert.equal(pager.snapshot().cacheHits, 1); assert.equal(pager.snapshot().cacheBytes, 2 * 65535 * 1024);
  assert.equal(pager.snapshot().readBytes, 4 * 65535 * 1024); assert.equal(pager.snapshot().pageBytes, 64 * MiB);
  assert.ok(pager.snapshot().peakGpuBytes <= gpu.limit);
  pager.dispose(); assert.equal(gpu.allocated, 32 * MiB);
});
test('Faster pages respect adapter limits, align complete quantized rows, and reject oversized rows', async () => {
  const gpu = gpuFixture(256, 32); let readLength = 0;
  const source = { name: 'fixture', size: 128 * MiB, async read(_offset: number, length: number) { readLength = length; return new Uint8Array(length); } };
  const pager = new WeightPager(gpu as unknown as Compute, source, 64);
  const quantized = { ...tensor, rowBytes: 2880 };
  const rows = pager.rowsPerPage(quantized); await pager.page(quantized, 0, rows);
  assert.equal(readLength % 2880, 0); assert.ok(readLength <= 32 * MiB);
  assert.equal(pager.snapshot().pageBytes, 32 * MiB);
  assert.equal(pager.rowsPerPage({ ...tensor, rowBytes: 32 }), 65535, 'small rows must stay within the GPU dispatch limit');
  await assert.rejects(pager.page({ ...tensor, rowBytes: 40 * MiB }, 0, 1), /exceeds/);
  pager.dispose();
});
test('performance options reject invalid modes and unsupported batch sizes before accessing a model', async () => {
  const engine = new PagedEngine(), file = new File([new Uint8Array(24)], 'fixture.gguf');
  await assert.rejects(engine.load(file, { performance: 'turbo' as 'fast' }), /Performance must/);
  await assert.rejects(engine.load(file, { performance: 'fast', batchSize: 256 }), /Batch size/);
  await assert.rejects(engine.load(file, { performance: 'fast', gpuBudgetMiB: 32768 }), /GPU budget/);
});
