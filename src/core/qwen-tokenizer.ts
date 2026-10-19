import type { Metadata } from './gguf.ts';

// Qwen3.8's Unicode pre-tokenizer, followed by GPT-2 byte encoding and ranked BPE.
// Contractions alone are case-insensitive; the Unicode letter/mark classes are not.
const pieces = /'[sStT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD]|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+/gu;
export function byteAlphabet(): string[] {
  const kept = Array.from({ length: 256 }, (_, b) => b).filter(b => b >= 33 && b <= 126 || b >= 161 && b <= 172 || b >= 174);
  const codes = new Set(kept); let extra = 256;
  return Array.from({ length: 256 }, (_, b) => String.fromCodePoint(codes.has(b) ? b : extra++));
}
export class QwenTokenizer {
  readonly tokens: string[];
  readonly eos: Set<number>;
  private ids = new Map<string, number>();
  private ranks = new Map<string, number>();
  private alphabet = byteAlphabet();
  private bytes = new Map<string, number>();
  private types: number[];
  constructor(m: Metadata) {
    if (m.get('tokenizer.ggml.model') !== 'gpt2' || m.get('tokenizer.ggml.pre') !== 'qwen35') throw new Error('Qwen3.8 requires the qwen35 byte-BPE tokenizer.');
    this.tokens = m.get('tokenizer.ggml.tokens') as string[];
    const merges = m.get('tokenizer.ggml.merges');
    this.types = m.get('tokenizer.ggml.token_type') as number[];
    if (!Array.isArray(this.tokens) || !Array.isArray(merges) || !Array.isArray(this.types) || this.types.length !== this.tokens.length || merges.some(s => typeof s !== 'string' || !s.includes(' '))) throw new Error('Invalid Qwen tokenizer metadata.');
    this.tokens.forEach((s, id) => this.ids.set(s, id));
    merges.forEach((s, rank) => this.ranks.set(s, rank));
    this.alphabet.forEach((s, b) => this.bytes.set(s, b));
    this.eos = new Set([this.special('<|im_end|>'), this.special('<|endoftext|>')]);
  }
  private special(s: string): number { const id = this.ids.get(s); if (id === undefined) throw new Error(`Qwen tokenizer lacks ${s}.`); return id; }
  encodeText(text: string): number[] {
    const result: number[] = [], encoder = new TextEncoder();
    for (const piece of text.normalize('NFC').matchAll(pieces)) {
      const symbols = Array.from(encoder.encode(piece[0]), b => this.alphabet[b]);
      while (symbols.length > 1) {
        let best = Infinity, at = -1;
        for (let i = 0; i < symbols.length - 1; i++) {
          const rank = this.ranks.get(symbols[i] + ' ' + symbols[i + 1]);
          if (rank !== undefined && rank < best) { best = rank; at = i; }
        }
        if (at < 0) break;
        symbols.splice(at, 2, symbols[at] + symbols[at + 1]);
      }
      for (const symbol of symbols) {
        const id = this.ids.get(symbol);
        if (id === undefined || this.types[id] !== 1) throw new Error('Invalid byte-BPE merge result.');
        result.push(id);
      }
    }
    return result;
  }
  chat(messages: { role: 'user' | 'assistant'; content: string }[]): number[] {
    const ids: number[] = [];
    const emptyThinking = () => [this.special('<think>'), ...this.encodeText('\n\n'), this.special('</think>'), ...this.encodeText('\n\n')];
    for (const m of messages) {
      ids.push(this.special('<|im_start|>'), ...this.encodeText(m.role + '\n'));
      if (m.role === 'assistant') ids.push(...emptyThinking());
      ids.push(...this.encodeText(m.content.trim()), this.special('<|im_end|>'), ...this.encodeText('\n'));
    }
    // Explicit non-thinking mode saves the small browser context for the answer.
    ids.push(this.special('<|im_start|>'), ...this.encodeText('assistant\n'), ...emptyThinking());
    return ids;
  }
  decode(ids: number[]): string {
    const bytes: number[] = [];
    for (const id of ids) {
      if (this.types[id] !== 1) continue;
      for (const ch of this.tokens[id] ?? '') { const b = this.bytes.get(ch); if (b === undefined) throw new Error('Invalid byte-BPE output token.'); bytes.push(b); }
    }
    return new TextDecoder().decode(new Uint8Array(bytes), { stream: true });
  }
}
