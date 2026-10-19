import { test } from 'node:test';
import assert from 'node:assert/strict';
import { Gpt2Tokenizer, chatStyleFor, preTokenizer } from '../src/core/gpt2-tokenizer.ts';
import { byteAlphabet } from '../src/core/bpe.ts';
const alphabet = byteAlphabet();
const merges = ['h e', 'he l', 'hel l', 'hell o', 'Ġ w', 'Ġw o', 'Ġwo r', 'Ġwor l', 'Ġworl d'];
const specials = ['<|im_start|>', '<|im_end|>', '<|eot_id|>', '<|start_header_id|>', '<|end_header_id|>', '<|begin_of_text|>', '[INST]', '[/INST]', '</s>', '<s>'];
/** A byte-level vocabulary: single-byte fallbacks, the merges under test, then control tokens. */
function metadata(control: string[], extra: Record<string, unknown> = {}) {
  const tokens = [...alphabet, ...merges.map(m => m.split(' ').join('')), ...control];
  const types = tokens.map((_, i) => i >= alphabet.length + merges.length ? 3 : 1);
  return new Map<string, unknown>([
    ['tokenizer.ggml.model', 'gpt2'], ['tokenizer.ggml.tokens', tokens], ['tokenizer.ggml.merges', merges],
    ['tokenizer.ggml.token_type', types], ['tokenizer.ggml.eos_token_id', tokens.indexOf('<|eot_id|>')],
    ...Object.entries(extra),
  ]);
}
const tokensOf = (m: Map<string, unknown>) => m.get('tokenizer.ggml.tokens') as string[];
test('byte-level BPE round-trips arbitrary Unicode through single-byte tokens', () => {
  const tokenizer = new Gpt2Tokenizer(metadata(specials));
  for (const text of ['hello world', 'Ünïcödé — ok', 'emoji 🚀 tail', '  double\n\nnewlines', '']) {
    assert.equal(tokenizer.decode(tokenizer.encodeText(text)), text, JSON.stringify(text));
  }
});
test('ranked merges are preferred over single bytes', () => {
  const tokenizer = new Gpt2Tokenizer(metadata(specials)), tokens = tokensOf(metadata(specials));
  assert.deepEqual(tokenizer.encodeText('hello'), [tokens.indexOf('hello')]);
  assert.deepEqual(tokenizer.encodeText('hello world'), [tokens.indexOf('hello'), tokens.indexOf('Ġworld')]);
  // 'wo' has no merge entry, so it stays as two byte tokens.
  assert.deepEqual(tokenizer.encodeText('wo'), [tokens.indexOf('w'), tokens.indexOf('o')]);
});
test('special tokens stay literal and are never produced by text encoding', () => {
  const tokenizer = new Gpt2Tokenizer(metadata(specials)), tokens = tokensOf(metadata(specials));
  assert.ok(!tokenizer.encodeText('<|im_end|>').includes(tokens.indexOf('<|im_end|>')));
  assert.equal(tokenizer.decode([tokens.indexOf('<|im_end|>')]), '');
});
test('ChatML framing is chosen from the template and uses vocabulary control tokens', () => {
  const chatml = metadata(specials, { 'tokenizer.chat_template': '{{ bos_token }}<|im_start|>{{ role }}\n{{ content }}<|im_end|>\n' });
  const tokenizer = new Gpt2Tokenizer(chatml), tokens = tokensOf(chatml);
  assert.equal(tokenizer.style, 'chatml');
  const ids = tokenizer.chat([{ role: 'user', content: 'hello' }]);
  assert.equal(ids[0], tokens.indexOf('<|im_start|>'));
  assert.equal(ids.at(-1), tokenizer.encodeText('assistant\n').at(-1));
});
test('Llama 3 header framing emits control tokens in order and opens the assistant turn', () => {
  const llama = metadata(specials, { 'tokenizer.chat_template': '<|begin_of_text|><|start_header_id|>{{ role }}<|end_header_id|>\n\n{{ content }}<|eot_id|>' });
  const tokenizer = new Gpt2Tokenizer(llama), tokens = tokensOf(llama);
  assert.equal(tokenizer.style, 'llama3');
  const ids = tokenizer.chat([{ role: 'user', content: 'hello' }, { role: 'assistant', content: 'hi' }, { role: 'user', content: 'hello' }]);
  const header = tokens.indexOf('<|start_header_id|>'), end = tokens.indexOf('<|end_header_id|>'), eot = tokens.indexOf('<|eot_id|>');
  const role = tokenizer.encodeText('user');
  assert.deepEqual(ids.slice(0, 2 + role.length), [tokens.indexOf('<|begin_of_text|>'), header, ...role]);
  assert.ok(ids.includes(end) && ids.filter(id => id === eot).length === 3);
  const tail = tokenizer.encodeText('\n\n');
  assert.deepEqual(ids.slice(-tail.length), tail);
});
test('Mistral instruction framing is detected and closes the assistant turn', () => {
  const mistral = metadata(specials, { 'tokenizer.chat_template': '<s>{% for m in messages %}[INST] {{ m.content }} [/INST]{% endfor %}' });
  const tokenizer = new Gpt2Tokenizer(mistral), tokens = tokensOf(mistral);
  assert.equal(tokenizer.style, 'mistral');
  const ids = tokenizer.chat([{ role: 'user', content: 'hello' }, { role: 'assistant', content: 'hi' }]);
  assert.equal(ids[0], tokens.indexOf('<s>'));
  assert.equal(ids[1], tokens.indexOf('[INST]'));
  assert.equal(ids.at(-1), tokens.indexOf('</s>'));
});
test('an unknown template without recognizable control tokens fails loudly', () => {
  assert.throws(() => new Gpt2Tokenizer(metadata([])), /chat template/);
  assert.throws(() => chatStyleFor(new Map(), new Map()), /chat template/);
});
test('end-of-turn tokens are stop tokens while other control tokens are not', () => {
  const tokenizer = new Gpt2Tokenizer(metadata(specials)), tokens = tokensOf(metadata(specials));
  assert.ok(tokenizer.eos.has(tokens.indexOf('<|eot_id|>')));
  assert.ok(!tokenizer.eos.has(tokens.indexOf('<|im_start|>')));
});
test('the pre-tokenizer splits contractions and falls back for unknown vocabulary names', () => {
  assert.deepEqual('hello world'.match(preTokenizer('llama-bpe'))!, ['hello', ' world']);
  assert.deepEqual('hello world'.match(preTokenizer('unheard-of'))!, ['hello', ' world']);
  assert.deepEqual("it's".match(preTokenizer('qwen2'))!, ['it', "'s"]);
});
