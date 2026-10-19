/** GGML block layouts, decoded without expanding whole weight matrices. */
export function halfToFloat(h: number): number {
  const sign = h & 0x8000 ? -1 : 1, exponent = (h >>> 10) & 31, fraction = h & 1023;
  return exponent === 0 ? sign * 2 ** -14 * fraction / 1024 : exponent === 31 ? (fraction ? NaN : sign * Infinity) : sign * 2 ** (exponent - 15) * (1 + fraction / 1024);
}
export function decodeWeight(data: Uint8Array, type: number, i: number): number {
  const b = (p: number) => data[p];
  const f16 = (p: number) => halfToFloat(b(p) | b(p + 1) << 8);
  if (type === 0) return new DataView(data.buffer, data.byteOffset, data.byteLength).getFloat32(i * 4, true);
  if (type === 1) return f16(i * 2);
  if (type === 2) { const p = Math.floor(i / 32) * 18, j = i % 32; return f16(p) * (((b(p + 2 + j % 16) >>> (j < 16 ? 0 : 4)) & 15) - 8); }
  if (type === 8) { const p = Math.floor(i / 32) * 34, q = b(p + 2 + i % 32); return f16(p) * (q > 127 ? q - 256 : q); }
  if (type === 12 || type === 13) {
    const p = Math.floor(i / 256) * (type === 12 ? 144 : 176), j = i % 256, s = Math.floor(j / 32);
    const sc = s < 4 ? b(p + 4 + s) & 63 : (b(p + 8 + s) & 15) | ((b(p + s) >>> 6) << 4);
    const mn = s < 4 ? b(p + 8 + s) & 63 : (b(p + 8 + s) >>> 4) | ((b(p + 4 + s) >>> 6) << 4);
    let q = (b(p + (type === 12 ? 16 : 48) + Math.floor(j / 64) * 32 + j % 32) >>> (s % 2 * 4)) & 15;
    if (type === 13) q |= ((b(p + 16 + j % 32) >>> s) & 1) << 4;
    return f16(p) * sc * q - f16(p + 2) * mn;
  }
  if (type === 14) {
    const p = Math.floor(i / 256) * 210, j = i % 256, segment = Math.floor(j / 128), quarter = Math.floor(j % 128 / 32), lane = j % 32;
    const low = (b(p + segment * 64 + quarter % 2 * 32 + lane) >>> (Math.floor(quarter / 2) * 4)) & 15;
    const high = (b(p + 128 + segment * 32 + lane) >>> (quarter * 2)) & 3;
    const scale = b(p + 192 + segment * 8 + quarter * 2 + Math.floor(lane / 16));
    return f16(p + 208) * (scale > 127 ? scale - 256 : scale) * ((low | high << 4) - 32);
  }
  throw new Error(`Unsupported quantization ${type}.`);
}
