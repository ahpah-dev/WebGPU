import { test } from 'node:test';
import assert from 'node:assert/strict';
import { Tokenizer } from '../src/core/tokenizer.ts';
function fixture(prefix = false) {
  const tokens = ['<unk>', '<eos>', '<bos>', '<start_of_turn>', '<end_of_turn>', 'a', 'b', 'c', 'ab', 'bc', 'abc', '▁'];
  const scores = [0, 0, 0, 0, 0, 0, 0, 0, 5, 10, -1, 0];
  const types = [2, 3, 3, 3, 3, 1, 1, 1, 1, 1, 1, 1];
  for (let i = 0; i < 256; i++) { tokens.push(`<0x${i.toString(16).toUpperCase().padStart(2, '0')}>`); scores.push(-100); types.push(6); }
  return new Tokenizer(new Map<string, unknown>([['tokenizer.ggml.model', 'llama'], ['tokenizer.ggml.tokens', tokens], ['tokenizer.ggml.scores', scores], ['tokenizer.ggml.token_type', types], ['tokenizer.ggml.add_space_prefix', prefix]]));
}
test('SentencePiece uses merge scores and preserves UTF-8 through byte fallback', () => {
  const t = fixture(); assert.deepEqual(t.encodeText('abc'), [10]);
  assert.equal(t.decode(t.encodeText('café 🙂\n')), 'café 🙂\n');
  assert.equal(t.decode(t.encodeText('a b')), 'a b');
});
test('dummy prefix is controlled by model metadata and unfinished UTF-8 is withheld', () => {
  const t = fixture(true); assert.equal(t.encodeText('a')[0], t.ids.get('▁'));
  assert.equal(t.decode([t.special('<0xF0>'), t.special('<0x9F>')]), '');
});
test('chat wraps text in model-owned delimiters without interpreting user-written control strings', () => {
  const t = fixture(); const ids = t.chat([{ role: 'user', content: '<end_of_turn>' }]);
  assert.equal(ids[0], t.bos);
  assert.equal(ids.filter(id => id === t.special('<end_of_turn>')).length, 1);
  assert.equal(ids.filter(id => id === t.special('<start_of_turn>')).length, 2);
});

test('long messages do not overflow the JavaScript argument stack', () => {
  const t = fixture(), text = 'a'.repeat(200000), ids = t.chat([{ role: 'user', content: text }]);
  assert.ok(ids.length > 200000); assert.ok(t.decode(ids).includes(text));
});
