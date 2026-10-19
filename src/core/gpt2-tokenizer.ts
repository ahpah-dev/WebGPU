import type { Metadata } from './gguf.ts';
import { byteAlphabet, mergeSymbols } from './bpe.ts';

/** Chat framings used by the byte-BPE (gpt2 vocabulary) model families. */
export type ChatStyle = 'llama3' | 'chatml' | 'mistral';

// llama.cpp's pre-tokenizer patterns. Inline modifier groups are avoided because
// older Safari builds reject them, so contractions are spelled out instead.
const contraction = `'[sStT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD]`;
const patterns: Record<string, string> = {
  'llama-bpe': `(?:${contraction})|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+`,
  'llama3': `(?:${contraction})|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+`,
  'smollm': `(?:${contraction})|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+`,
  'deepseek-llm': `(?:${contraction})|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+`,
  'qwen2': `(?:${contraction})|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+`,
  'qwen35': `(?:${contraction})|[^\\r\\n\\p{L}\\p{N}]?[\\p{L}\\p{M}]+|\\p{N}| ?[^\\s\\p{L}\\p{M}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+`,
  'gpt2': `(?:${contraction})| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+`,
  'default': `(?:${contraction})| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+`,
};
const fallback = patterns['llama-bpe'];
const preTokenizers = new Map<string, RegExp>();

export function preTokenizer(name: unknown): RegExp {
  const key = typeof name === 'string' && patterns[name] ? name : 'default';
  let pattern = preTokenizers.get(key);
  if (!pattern) { pattern = new RegExp(patterns[key] ?? fallback, 'gu'); preTokenizers.set(key, pattern); pattern.lastIndex = 0; }
  return pattern;
}

/** Picks a chat framing from the GGUF chat template, falling back to the tokens
 * actually present in the vocabulary so a missing template cannot produce a
 * silently wrong prompt. */
export function chatStyleFor(m: Metadata, ids: Map<string, number>): ChatStyle {
  const template = m.get('tokenizer.chat_template');
  if (typeof template === 'string') {
    if (template.includes('<|start_header_id|>')) return 'llama3';
    if (template.includes('<|im_start|>')) return 'chatml';
    if (template.includes('[INST]')) return 'mistral';
  }
  if (ids.has('<|start_header_id|>')) return 'llama3';
  if (ids.has('<|im_start|>')) return 'chatml';
  if (ids.has('[INST]')) return 'mistral';
  throw new Error('This model does not declare a chat template this runtime understands. Use a chat or instruct GGUF (Llama 3, Qwen2/3, Mistral).');
}

const END_TOKENS = ['<|eot_id|>', '<|eom_id|>', '<|end_of_text|>', '<|im_end|>', '<|endoftext|>', '<|end|>', '</s>', '<end_of_turn>'];

/** Byte-level BPE with the GPT-2 unicode alphabet: Llama 3.x, Mistral, SmolLM and
 * Qwen2/3 all ship a `gpt2` tokenizer model with ranked merges. */
export class Gpt2Tokenizer {
  readonly tokens: string[];
  readonly eos: Set<number>;
  readonly style: ChatStyle;
  private ids = new Map<string, number>();
  private ranks = new Map<string, number>();
  private alphabet = byteAlphabet();
  private bytes = new Map<string, number>();
  private types: number[];
  private pattern: RegExp;
  constructor(m: Metadata) {
    if (m.get('tokenizer.ggml.model') !== 'gpt2') throw new Error('This model does not use a byte-level BPE tokenizer.');
    this.tokens = m.get('tokenizer.ggml.tokens') as string[];
    const merges = m.get('tokenizer.ggml.merges');
    const types = m.get('tokenizer.ggml.token_type');
    if (!Array.isArray(this.tokens) || !Array.isArray(merges) || merges.some(s => typeof s !== 'string' || !s.includes(' '))) throw new Error('Invalid byte-BPE tokenizer metadata.');
    this.types = Array.isArray(types) ? types as number[] : [];
    this.tokens.forEach((s, id) => this.ids.set(s, id));
    merges.forEach((s, rank) => this.ranks.set(s as string, rank));
    this.alphabet.forEach((s, b) => this.bytes.set(s, b));
    this.pattern = preTokenizer(m.get('tokenizer.ggml.pre'));
    this.eos = new Set<number>();
    for (const key of ['tokenizer.ggml.eos_token_id', 'tokenizer.ggml.eot_token_id']) {
      const id = Number(m.get(key)); if (Number.isSafeInteger(id) && id >= 0) this.eos.add(id);
    }
    const many = m.get('tokenizer.ggml.eos_token_ids');
    if (Array.isArray(many)) for (const id of many) if (Number.isSafeInteger(id)) this.eos.add(id as number);
    for (const name of END_TOKENS) { const id = this.ids.get(name); if (id !== undefined) this.eos.add(id); }
    this.style = chatStyleFor(m, this.ids);
  }
  private id(name: string): number | undefined { return this.ids.get(name); }
  private control(name: string, ids: number[]) {
    const id = this.ids.get(name);
    if (id !== undefined) ids.push(id);
    else ids.push(...this.encodeText(name));
  }
  encodeText(text: string): number[] {
    const result: number[] = [], encoder = new TextEncoder();
    for (const piece of text.matchAll(this.pattern)) {
      const symbols = mergeSymbols(Array.from(encoder.encode(piece[0]), b => this.alphabet[b]), (a, b) => {
        const rank = this.ranks.get(`${a} ${b}`);
        return rank === undefined ? undefined : -rank;
      });
      for (const symbol of symbols) {
        const id = this.ids.get(symbol);
        if (id === undefined) throw new Error('Invalid byte-BPE merge result.');
        result.push(id);
      }
    }
    return result;
  }
  chat(messages: { role: 'user' | 'assistant'; content: string }[]): number[] {
    const ids: number[] = [], text = (s: string) => ids.push(...this.encodeText(s));
    if (this.style === 'llama3') {
      const bos = this.id('<|begin_of_text|>'); if (bos !== undefined) ids.push(bos);
      for (const m of messages) {
        this.control('<|start_header_id|>', ids); text(m.role === 'assistant' ? 'assistant' : 'user');
        this.control('<|end_header_id|>', ids); text('\n\n'); text(m.content.trim()); this.control('<|eot_id|>', ids);
      }
      this.control('<|start_header_id|>', ids); text('assistant'); this.control('<|end_header_id|>', ids); text('\n\n');
    } else if (this.style === 'chatml') {
      for (const m of messages) {
        this.control('<|im_start|>', ids); text(`${m.role}\n`); text(m.content.trim()); this.control('<|im_end|>', ids); text('\n');
      }
      this.control('<|im_start|>', ids); text('assistant\n');
    } else {
      const bos = this.id('<s>'); if (bos !== undefined) ids.push(bos);
      for (const m of messages) {
        if (m.role === 'user') { this.control('[INST]', ids); text(` ${m.content.trim()} `); this.control('[/INST]', ids); }
        else { text(m.content.trim()); const eot = this.id('</s>'); if (eot !== undefined) ids.push(eot); }
      }
    }
    return ids;
  }
  decode(ids: number[]): string {
    const bytes: number[] = [];
    for (const id of ids) {
      if ((this.types[id] ?? 1) === 3) continue;
      for (const ch of this.tokens[id] ?? '') { const b = this.bytes.get(ch); if (b !== undefined) bytes.push(b); }
    }
    return new TextDecoder().decode(new Uint8Array(bytes), { stream: true });
  }
}
