export const MiB = 1024 ** 2;
export class Compute {
  device!: GPUDevice;
  allocated = 0;
  peak = 0;
  limit: number;
  adapterName = '';
  private sizes = new Map<GPUBuffer, number>();
  private fault: Error | null = null;
  constructor(budgetMiB: number) { this.limit = budgetMiB * MiB; }
  async init() {
    if (!navigator.gpu) throw new Error('WebGPU is unavailable. Open this app in a WebGPU-enabled browser on localhost or HTTPS.');
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
    if (!adapter) throw new Error('The browser could not access a WebGPU adapter. Check hardware acceleration.');
    this.adapterName = adapter.info.description || adapter.info.device || adapter.info.architecture || 'WebGPU adapter';
    this.device = await adapter.requestDevice({ requiredLimits: {
      maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
      maxBufferSize: adapter.limits.maxBufferSize,
    } });
    this.device.lost.then(info => { this.fault = new Error(`GPU device lost: ${info.message || info.reason}. Reload with a smaller memory budget.`); });
    this.device.addEventListener('uncapturederror', e => { this.fault = new Error(e.error.message); });
  }
  check() { if (this.fault) throw this.fault; }
  buffer(size: number, label: string, usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC) {
    this.check(); size = Math.max(4, Math.ceil(size / 4) * 4);
    if (this.allocated + size > this.limit) throw new Error('GPU budget exceeded. Reduce context length or increase the memory budget.');
    if (size > this.device.limits.maxBufferSize || ((usage & GPUBufferUsage.STORAGE) && size > this.device.limits.maxStorageBufferBindingSize)) throw new Error(`Buffer ${label} exceeds this browser's GPU limits.`);
    const b = this.device.createBuffer({ size, label, usage });
    this.sizes.set(b, size); this.allocated += size; this.peak = Math.max(this.peak, this.allocated); return b;
  }
  destroy(b: GPUBuffer) { const size = this.sizes.get(b); if (size !== undefined) { b.destroy(); this.allocated -= size; this.sizes.delete(b); } }
  write(b: GPUBuffer, data: ArrayBuffer | ArrayBufferView<ArrayBuffer>, offset = 0) { this.device.queue.writeBuffer(b, offset, data); }
  async pipeline(code: string, label: string) {
    const module = this.device.createShaderModule({ code, label });
    const info = await module.getCompilationInfo();
    const errors = info.messages.filter(m => m.type === 'error');
    if (errors.length) throw new Error(`${label}: ${errors.map(e => `${e.lineNum}: ${e.message}`).join('; ')}`);
    return await this.device.createComputePipelineAsync({ label, layout: 'auto', compute: { module, entryPoint: 'main' } });
  }
  run(pipeline: GPUComputePipeline, buffers: GPUBuffer[], params: (number | { f: number })[], x: number, y = 1) {
    this.check();
    const data = new ArrayBuffer(Math.ceil(Math.max(16, params.length * 4) / 16) * 16);
    const u = new Uint32Array(data), f = new Float32Array(data);
    params.forEach((v, i) => { if (typeof v === 'number') u[i] = v; else f[i] = v.f; });
    const uniform = this.buffer(data.byteLength, 'dispatch parameters', GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
    this.write(uniform, data);
    const bg = this.device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [...buffers, uniform].map((buffer, binding) => ({ binding, resource: { buffer } })) });
    const encoder = this.device.createCommandEncoder(), pass = encoder.beginComputePass();
    pass.setPipeline(pipeline); pass.setBindGroup(0, bg); pass.dispatchWorkgroups(x, y); pass.end();
    this.device.queue.submit([encoder.finish()]); this.destroy(uniform);
  }
  copy(src: GPUBuffer, dst: GPUBuffer, size: number, srcOffset = 0, dstOffset = 0) {
    const e = this.device.createCommandEncoder(); e.copyBufferToBuffer(src, srcOffset, dst, dstOffset, size); this.device.queue.submit([e.finish()]);
  }
  async done() { await this.device.queue.onSubmittedWorkDone(); this.check(); }
  async read(b: GPUBuffer, size: number) {
    const rb = this.buffer(size, 'readback', GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST);
    try { this.copy(b, rb, size); await rb.mapAsync(GPUMapMode.READ); this.check(); const result = rb.getMappedRange().slice(0); rb.unmap(); return result; }
    finally { this.destroy(rb); }
  }
  dispose() { for (const b of [...this.sizes.keys()]) this.destroy(b); this.device?.destroy(); }
}
