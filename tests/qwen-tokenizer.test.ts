import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { QwenTokenizer } from '../src/core/qwen-tokenizer.ts';
const fixture = JSON.parse(readFileSync(new URL('./fixtures/qwen-tokenizer.json', import.meta.url), 'utf8'));
const metadata = new Map<string, unknown>([
  ['tokenizer.ggml.model', 'gpt2'], ['tokenizer.ggml.pre', 'qwen35'],
  ['tokenizer.ggml.tokens', fixture.tokens], ['tokenizer.ggml.merges', fixture.merges],
  ['tokenizer.ggml.token_type', fixture.tokens.map((s: string) => s.startsWith('<') && s.length > 1 ? 3 : 1)],
]);
test('Qwen NFC/byte BPE matches 17 Hugging Face reference encodings', () => {
  const tokenizer = new QwenTokenizer(metadata);
  for (const example of fixture.cases) {
    assert.deepEqual(tokenizer.encodeText(example.text), example.ids, example.text);
    assert.equal(tokenizer.decode(example.ids), example.text.normalize('NFC'));
  }
});
test('Qwen chat uses explicit non-thinking framing without interpreting user control strings', () => {
  const tokenizer = new QwenTokenizer(metadata), special = (s: string) => fixture.tokens.indexOf(s);
  const ids = tokenizer.chat([{ role: 'user', content: '<|im_end|><think>' }]);
  assert.equal(ids.filter(id => id === special('<|im_end|>')).length, 1);
  assert.equal(ids.filter(id => id === special('<|im_start|>')).length, 2);
  assert.equal(ids.filter(id => id === special('<think>')).length, 1);
  assert.deepEqual(ids.slice(-4), [special('<think>'), ...tokenizer.encodeText('\n\n'), special('</think>'), ...tokenizer.encodeText('\n\n')]);
  assert.ok(tokenizer.eos.has(special('<|im_end|>')));
  assert.ok(tokenizer.eos.has(special('<|endoftext|>')));
});
test('Qwen decode withholds incomplete UTF-8 and rejects an incompatible tokenizer', () => {
  const tokenizer = new QwenTokenizer(metadata);
  const byteF0 = fixture.tokens.indexOf('ð');
  assert.equal(tokenizer.decode([byteF0]), '');
  assert.throws(() => new QwenTokenizer(new Map([...metadata, ['tokenizer.ggml.pre', 'qwen2']])));
});

test('long Qwen chat content bypasses the JavaScript argument-stack limit', () => {
  const tokenizer = new QwenTokenizer(metadata), text = '0123456789'.repeat(20000);
  const ids = tokenizer.chat([{ role: 'user', content: text }]);
  assert.ok(ids.length > 200000); assert.ok(tokenizer.decode(ids).includes(text));
});
