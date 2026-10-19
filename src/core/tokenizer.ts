import type { Metadata } from './gguf.ts';

/** SentencePiece BPE merge scores from GGUF, with UTF-8 byte fallback. */
export class Tokenizer {
  tokens: string[];
  scores: number[];
  types: number[];
  ids = new Map<string, number>();
  bos: number;
  eos: Set<number>;
  private prefix: boolean;
  constructor(m: Metadata) {
    if (m.get('tokenizer.ggml.model') !== 'llama') throw new Error('Only the Gemma SentencePiece tokenizer is supported.');
    this.tokens = m.get('tokenizer.ggml.tokens') as string[];
    this.scores = m.get('tokenizer.ggml.scores') as number[];
    this.types = m.get('tokenizer.ggml.token_type') as number[];
    if (!Array.isArray(this.tokens) || !Array.isArray(this.scores) || this.tokens.length !== this.scores.length) throw new Error('Invalid tokenizer metadata.');
    this.tokens.forEach((s, i) => this.ids.set(s, i));
    this.bos = Number(m.get('tokenizer.ggml.bos_token_id') ?? 2);
    this.eos = new Set([Number(m.get('tokenizer.ggml.eos_token_id') ?? 1), this.special('<end_of_turn>')]);
    // GGUF's SentencePiece default is true unless the model explicitly overrides it.
    this.prefix = m.get('tokenizer.ggml.add_space_prefix') !== false;
  }
  special(s: string): number { const id = this.ids.get(s); if (id === undefined) throw new Error(`Tokenizer lacks ${s}.`); return id; }
  encodeText(text: string): number[] {
    // Gemma uses identity normalization and U+2581 to encode spaces.
    const symbols = Array.from((this.prefix ? ' ' : '') + text).map(s => s === ' ' ? '\u2581' : s);
    while (symbols.length > 1) {
      let best = -Infinity, index = -1;
      for (let i = 0; i < symbols.length - 1; i++) {
        const id = this.ids.get(symbols[i] + symbols[i + 1]);
        if (id !== undefined && (this.types?.[id] ?? 1) === 1 && this.scores[id] > best) { best = this.scores[id]; index = i; }
      }
      if (index < 0) break;
      symbols.splice(index, 2, symbols[index] + symbols[index + 1]);
    }
    const result: number[] = [], encoder = new TextEncoder();
    for (const symbol of symbols) {
      const id = this.ids.get(symbol);
      if (id !== undefined && (this.types?.[id] ?? 1) === 1) result.push(id);
      else for (const byte of encoder.encode(symbol)) result.push(this.special(`<0x${byte.toString(16).toUpperCase().padStart(2, '0')}>`));
    }
    return result;
  }
  chat(messages: { role: 'user' | 'assistant'; content: string }[]): number[] {
    const ids = [this.bos];
    for (const message of messages) {
      ids.push(this.special('<start_of_turn>'), ...this.encodeText(`${message.role === 'assistant' ? 'model' : 'user'}\n${message.content.trim()}`), this.special('<end_of_turn>'), ...this.encodeText('\n'));
    }
    ids.push(this.special('<start_of_turn>'), ...this.encodeText('model\n'));
    return ids;
  }
  decode(ids: number[]): string {
    const bytes: number[] = [], encoder = new TextEncoder();
    for (const id of ids) {
      const t = this.tokens[id] ?? '';
      if (/^<0x[\dA-F]{2}>$/.test(t)) bytes.push(parseInt(t.slice(3, 5), 16));
      else if ((this.types?.[id] ?? 1) !== 3) bytes.push(...encoder.encode(t.replaceAll('\u2581', ' ')));
    }
    return new TextDecoder().decode(new Uint8Array(bytes), { stream: true });
  }
}
