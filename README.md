# WebGPU

**A browser-native inference runtime for local GGUF models.**

WebGPU reads a GGUF file in small ranges and runs quantized transformer computations in WGSL shaders. Small models (0.4–2 GB) stay fully resident and answer in seconds on a laptop GPU. 27B models stream through a bounded GPU budget instead of fitting in VRAM. A model does not have to fit completely in VRAM or JavaScript memory.

This is an experimental runtime with a working browser UI and JavaScript SDK. It is not an Ollama inference frontend. No model execution runs in Node, Ollama, a native helper, or a cloud service. The project is named WebGPU and uses the browser's WebGPU standard; it is an independent project.

**Verified:** Gemma 3 27B generated `Paris` entirely in the browser with a **2 GiB engine buffer budget**, peaking at **1.983 GiB** of tracked GPU allocations on an RTX 5070. The one-token SDK smoke test took **120 seconds**, including prompt processing; 108 seconds were spent awaiting weight reads. This is a working experimental memory-saving runtime, not a promise of fast inference. See [measured results, screenshots, and test limits](docs/VALIDATION.md).

**Small dense models verified numerically:** Llama 3.x (`llama`), Qwen2.5 (`qwen2`), and Qwen3 (`qwen3`) GGUF files run through the dense decoder with Q/K bias, Q/K normalization, and adjacent-pair or half-split RoPE as the file requires. `npm run check:dense` builds synthetic float32 GGUFs for all three, generates greedy tokens in headless Chrome on real WebGPU hardware, and compares them with an independent CPU reference: **12/12 tokens match** across the three architectures. See [the hardware check](#verification).

**Qwen3.8 27B verified:** the Q4_K_M model also generated `Paris` in the browser with a **2 GiB budget**, peaking at **1.985 GiB**. Its one-token SDK test took **181.9 seconds**, including the 23-token prompt; 164.8 seconds were spent awaiting weight reads. [Raw Qwen result](docs/qwen38-benchmark.json).

## Use it

The built app is fully static. `dist/` can be served by any static file server or opened through the local byte server; there is no inference backend, so `npm start` only serves files. Inference, tokenization, paging, and the developer bridge all run inside the page.

Open [WebGPU](https://ahpah-dev.github.io/WebGPU/) or `http://127.0.0.1:8787` in a browser with WebGPU hardware acceleration, then choose **Open local models**. The folder picker uses the File System Access API: point it at any folder of GGUF files or your `~/.ollama/models` directory, and the browser remembers the handle for later visits. Picking a single GGUF file also works. The architecture is detected from the file and the file is read directly by a worker through the browser File API; it is never uploaded.

Start with a **512 MiB budget** and a small model: a 0.5–1B Q4_K_M file stays fully resident on almost any GPU and answers in seconds. The GPU must have additional headroom for the browser, driver, display, and temporary upload staging. The app does not automatically know how much free VRAM your browser can use.

### Presets

In **Run configuration → Performance**, select a preset, then **Apply & reload model**. If no model is loaded, select the preset before opening one. Changing the preset updates the budget control; you can lower that budget for your GPU before applying it.

| UI preset | GPU buffer budget | Maximum weight page | Prompt batch | Typical use |
| --- | --- | --- | --- | --- |
| Lite | 512 MiB | 16 MiB | 8 tokens | 0.5–1B models, integrated GPUs, minimum memory |
| Memory saver | 2 GiB | 16 MiB | 32 tokens | Default for 27B paging |
| Faster | 6 GiB | 64 MiB, capped by adapter limits | 128 tokens | Larger GPUs, models that nearly fit |

Lite keeps the engine's own GPU allocations under half a gigabyte, which leaves room for the browser on shared-memory laptops. Faster mode reserves more GPU memory for resident weights, reducing repeated disk reads after the cache warms. Larger file ranges reduce request overhead, and longer prompt batches reuse weights across more tokens. The model weights, context length, and decoder stay the same. The first prompt still needs to read the model.

Speed varies with GPU, storage, prompt length, and available memory; a 6 GiB engine budget also needs browser/driver headroom. A dense model that fits entirely in the budget runs resident: no weight pages are read during decoding, which is why small models feel like a normal chat app.

The GPU budget remains enforced. Upload scheduling fences after eight pages **or 128 MiB**, so larger pages do not multiply queued upload staging. The runtime panel shows the actual page size, prompt batch, and resident cache hits. Selecting **Memory saver** restores the 2 GiB preset.

In a short Qwen3.8-27B comparison on an RTX 5070, Memory saver took **220.6s** to the first token and **149.6s** for the next token; Faster took **136.7s** and **84.5s** respectively. Faster reduced the second pass's file reads by **30%** and peaked at **5.99 GiB** of tracked GPU buffers. Both modes generated identical tokens. This was a sequential two-token comparison with uncontrolled OS cache and background activity, not a general speed guarantee. See [the measurements](docs/VALIDATION.md#faster-preset-comparison) and [raw results](docs/speed-benchmark.json).

Paging makes the memory requirement smaller; it does **not** make 27 billion parameters cheap to compute. Large models can be very slow, especially on the first prompt. More GPU budget reduces disk reads. There are no inference API charges or subscriptions; you supply the hardware, storage, electricity, and model weights.

## Run from source

Requires Node.js 22.18+ (Node 24 LTS recommended) and a browser with WebGPU support.

```sh
git clone https://github.com/ahpah-dev/WebGPU.git
cd WebGPU
npm ci
npm run build
npm start
```

Open `http://127.0.0.1:8787` and click **Open local models**: pick a folder of GGUFs or your Ollama directory. The server binds to loopback and serves static assets and optional model bytes; all inference still runs in the browser. Because nothing on the server computes tokens, the same `dist/` works on GitHub Pages, a USB stick, or any static host.

For development, `npm run dev` runs Vite. Vite serves the app but not the optional `/model.gguf` route, so use the folder or file picker there.

### Use weights already downloaded by Ollama

The local app automatically discovers your Ollama library. No per-model startup argument, renaming, copying, or conversion is needed:

```sh
npm start
```

Click **Open local models** and choose `~/.ollama/models`, or click **Use the Ollama library instead** to browse served models by name. The picker displays each model's size and quantization, and models with unsupported architectures or quantization are hidden or explained. **Refresh** discovers models downloaded since the picker was opened. Ollama itself does not need to be running; WebGPU reads its saved manifests and weight files.

If you need weights first:

```sh
ollama pull llama3.2:1b
# Or:
ollama pull qwen2.5:0.5b
ollama pull qwen3.8:27b
```

The app respects `OLLAMA_MODELS`. For a custom directory, you can also use:

```sh
npm start -- --ollama-dir /absolute/path/to/ollama/models
```

On Windows and macOS it defaults to the current user's `.ollama/models`; on Linux it also recognizes the standard service location. The catalog includes models in custom namespaces and registries. It exposes opaque model URLs, and model reads still require bounded HTTP Range requests. The local API is `GET /api/ollama/models`; each returned entry includes a `url`, `size`, compatibility status, and display metadata for use with `model.load({url, size, name})`.

The GitHub Pages app has an **Open local Ollama library** link to the local app. Automatic discovery uses the local app; direct GGUF selection works on either site.

Qwen3.8 uses **non-thinking text chat** in this runtime, leaving the limited context for the answer. Its GGUF architecture is named `qwen35`; this is the correct architecture identifier for Qwen3.8, not a model substitution. The 64 text layers run; the optional MTP prediction layer and image projector are not evaluated.

The previous `--ollama-model qwen3.8:27b` option remains available for a preselected byte source. To serve a standalone GGUF:

```sh
npm start -- --model /absolute/path/to/gemma-3-27b-it-Q4_K_M.gguf
```

The Ollama Qwen3.8 27B GGUF is about **16.8 GB** (15.66 GiB); Gemma 3 27B is about **16.6 GB** (15.42 GiB). An Ollama pull can also download an image projector that this text-only engine does not use. You need disk space for the complete model. Weights are not bundled in this repository, and their own license applies.

## The mechanism

1. **Random-access GGUF reader.** Uses safe numeric file offsets beyond 4 GiB. Reads metadata separately and never allocates a model-sized ArrayBuffer. Malformed or unsupported models fail explicitly.
2. **Quantized kernels.** F32, F16, Q4_0, Q8_0, Q4_K, Q5_K, and Q6_K matrix multiplication decodes blocks on the GPU. The model's mixed tensor types remain quantized in storage and VRAM.
3. **Bounded weight pager.** A reusable 16 MiB slot (up to 64 MiB in Faster mode) streams uncached weight pages. A fixed resident prefix uses the remaining budget; it avoids the repeated full-cache eviction of naive LRU on sequential transformer scans. Pages respect storage/buffer limits and the GPU's maximum dispatch row count.
4. **Ordered uploads.** GPU writes and dispatches are ordered. Fences after at most eight pages or 128 MiB bound queued staging and allow uploads to overlap prior computation. Each transformer layer completes before progress advances.
5. **Batched prompt processing.** Up to 128 prompt tokens reuse each weight page in Faster mode. Gemma uses a GPU KV cache and conversation prefix reuse. Qwen retains KV state for full-attention layers and fixed-size recurrent/convolution state for its other layers. It rebuilds the prompt state for each new request so edited prompts, retries, and cancellation cannot reuse stale recurrent state.
6. **Dense decoder.** One implementation covers Llama 3.x, Mistral, SmolLM, Qwen2, and Qwen3: RMS normalization, linear Q/K/V with optional Qwen2 bias, optional Qwen3 per-head Q/K normalization, grouped-query causal attention, SwiGLU or a plain SiLU MLP when the file has no gate tensor, and tied or untied output projection. The file decides the RoPE pairing (`llama` stores adjacent pairs, Qwen uses the half-split NeoX layout) and whether a sliding window applies. Llama 3 and YaRN rope scaling factors are accepted but not yet applied to the rotation; the runtime documents this rather than pretending otherwise.
7. **Tokenizer selection.** `qwen35` uses the Qwen SentencePiece-derived tokenizer, `llama` files use a byte-level GPT-2 BPE with llama.cpp's pre-tokenization patterns, and Gemma files use the SentencePiece tokenizer. Chat framing comes from `tokenizer.chat_template` when present and is validated against the vocabulary's control tokens; an unknown template fails loudly instead of guessing.
8. **Gemma 3 decoder.** Implements embedding scaling, RMS normalization, Q/K normalization, local/global RoPE, grouped-query causal/sliding attention, residual connections, GELU feed-forward layers, and the tied output projection. The 27B attention scale is handled separately.
9. **Qwen3.8 decoder.** Implements causal depthwise convolution, L2-normalized Q/K, Gated DeltaNet with tiled GGUF value heads, gated RMS normalization, interleaved Q/gate projection, partial RoPE, gated grouped-query attention, SwiGLU, and the untied output projection. The tokenizer uses NFC normalization and ranked byte BPE with explicit non-thinking ChatML framing.
10. **Dedicated worker.** Sampling and tokenization run in JavaScript; transformer matrix/attention operations run on WebGPU. Both remain inside the browser. Stop cancels at the next asynchronous boundary, and disposal terminates the worker.

The GPU budget counts buffers allocated by this engine. It is **not a measurement of total process VRAM**: browser/driver allocations and temporary upload staging are additional. The runtime panel distinguishes file size, cumulative disk reads, resident cache hits, and tracked allocations.

## Browser SDK

`npm run build` emits ES modules and declarations to `dist/sdk/`. Host that folder alongside your app. No bundler or server inference route is required to consume it.

```js
import { WebGPU } from './sdk/sdk.js';

const model = new WebGPU();
model.onProgress = ({ detail, stats }) => console.log(detail, stats);

// file is a File from an <input type="file"> element.
const info = await model.load(file, {
  performance: 'fast',   // 'lite' (512 MiB), 'low-memory' (2 GiB, default) or 'fast' (6 GiB).
  gpuBudgetMiB: 6144,    // Optional explicit override to suit your GPU.
  contextLength: 512,
  batchSize: 128,        // Optional: 1, 8, 16, 32, 64 or 128.
});

for await (const chunk of model.generate(
  [{ role: 'user', content: 'Name the capital of France.' }],
  { maxTokens: 64, temperature: 0.7, topK: 40, topP: 0.9 },
)) {
  console.log(chunk.delta, chunk.stats.peakGpuBytes);
}

model.stop();     // Cancel a running operation.
await model.reset(); // Clear conversation KV reuse when idle.
model.dispose();  // Terminate the worker and release resources.
```

`load` also accepts `{url, size, name}` for a file server with exact HTTP 206 ranges. A server that ignores the Range header is rejected instead of downloading a multi-gigabyte file into memory.

This API is a browser JavaScript API, not a network REST server. Only one load or generation can run at a time. Context accepts any integer from **256 to 32,768 tokens**, bounded by the model's declared context and available GPU memory. `maxTokens` accepts **1–32,768**. The prompt, chat history, framing, and reply share the context: a 4,000-token prompt in a 32,768-token context leaves at most 28,768 reply tokens. EOS can end a reply earlier. Oversized prompts fail without silently dropping history.

For long conversations:

```js
await model.load(file, {
  performance: 'fast',
  gpuBudgetMiB: 6144,
  contextLength: 32768,
});
for await (const token of model.generate(messages, { maxTokens: 8192 })) {
  output.textContent = token.text;
  console.log(`${token.promptTokens + token.tokens}/${token.contextLength} context tokens`);
}
```

The UI defaults to a **4,096-token context / 2,048-token output limit** and offers 8K, 16K, and 32K settings. Apply & reload changes context while retaining the conversation in the current page; changing output length needs no reload. Opening a different model starts a fresh conversation. The SDK retains its 512-token context / 128-token output defaults for compatibility. Individual messages can contain up to 1,048,576 JavaScript characters; the tokenized context limit still applies.

Attention uses fixed 64-token tiles with online softmax, so shared shader memory does not grow with context. Gemma sliding-attention layers use ring buffers sized to the sliding window plus prompt-batch overlap. Full-attention KV state still grows with context. Qwen3.8 27B at 32K needs about **4.15 GiB for attention/recurrent state alone**, plus working buffers and cached weights; use the Faster 6 GiB preset on hardware with enough headroom. A longer context leaves less of the same GPU budget for weight caching and can slow generation. The engine checks the budget, model context, and browser buffer limits before allocating the cache.

`performance` accepts `'lite'`, `'low-memory'` (the default), or `'fast'`. The presets are 512 MiB / 8 prompt tokens, 2 GiB / 32 prompt tokens, and 6 GiB / 128 prompt tokens. Explicit `gpuBudgetMiB` and `batchSize` override the preset defaults; the low-memory SDK batch stays 16 for compatibility with the previous release. Supported batches are 1, 8, 16, 32, 64, and 128. `ModelInfo` reports `performance`, `batchSize`, and the actual `stats.pageBytes`.

## Developer API

Every page issues itself a bearer key and answers requests from other pages on the same origin. Open **Developer API** in the app to copy the key, a client snippet, and an iframe embedding snippet. The key looks like `wgpu_sk_…`, is generated with `crypto.getRandomValues`, and is stored in `localStorage` under `webgpu.apiKey`. **Generate new key** replaces it immediately and the previous key stops working.

The bridge is a `BroadcastChannel('webgpu-runtime')`, plus `window.postMessage` so an embedding page can drive the iframe. It lives only while the tab is open with a model loaded.

```js
const channel = new BroadcastChannel('webgpu-runtime');
const id = crypto.randomUUID();
channel.onmessage = ({ data }) => {
  if (data.id !== id) return;
  const { event } = data;
  if (event.type === 'chunk') output.textContent += event.delta;
  if (event.type === 'done') channel.close();
  if (event.type === 'error') console.error(event.message);
};
channel.postMessage({
  type: 'webgpu.request', id, apiKey, method: 'chat.completions',
  messages: [{ role: 'user', content: 'Hello' }], maxTokens: 64,
});
```

Events are `ready` (a `ping` reply with the loaded model, architecture, and context length), `chunk` (streamed deltas with token and context counts), `done`, and `error`. A wrong key, no loaded model, a busy runtime, or invalid messages all come back as `error`; one generation runs at a time. There is deliberately no HTTP endpoint: a browser tab cannot accept inbound connections, so a `http://…/v1/chat/completions` URL would be fiction. If you need a REST surface, run a small proxy that forwards into this channel, or use the SDK directly.

## Verification

```sh
npm test
npm run build
npm start -- --ollama-model gemma3:27b
```

Open `http://127.0.0.1:8787/verify.html` and select **Run kernel checks**. These checks execute real WebGPU kernels and compare their results with CPU references for all seven tensor types, batched/offset matrix operations, causal and sliding attention, RMS norm, and scaled RoPE. Qwen checks cover interleaved query/gates, causal convolution history, decay, recurrent state, batch versus sequential DeltaNet, gated normalization, SwiGLU, and partial RoPE. The standalone SDK smoke test loads whichever local model the byte server exposes and generates text; it is not a mocked inference test.

CPU tests cover independently packed quantization fixtures, non-finite sampling rejection, file bounds, HTTP range enforcement, invalid GGUF headers, real Qwen3.8 tensor shapes, MTP exclusion, and 17 reference tokenizer cases from Hugging Face's tokenizer. GitHub Actions runs these tests and the production build before publishing the static application.

Ollama integration tests exercise namespaced discovery, unsupported and incomplete models, malformed manifests, tag changes, selected-model HTTP byte ranges, opaque IDs, and origin restrictions. The browser picker has been checked with an actual installed `qwen3.8:27b`, including search, refresh, and loading into the runtime.

Pager tests check larger pages, fixed cache reuse, bounded GPU allocations, adapter/dispatch limits, and invalid performance settings. GPU matrix and recurrent checks run 128-token batches against independent CPU references. The verification page also offers **Compare speed presets (2 tokens each)**: it runs the same local model and greedy prompt sequentially in both presets and reports token times, read bytes, cache hits, and tracked memory. This is a short benchmark rather than a universal speed guarantee.

### Dense decoder hardware check

`npm run check:dense` builds three synthetic float32 GGUFs in the OS temp directory and runs them through the real browser runtime in headless Chrome. Each fixture is a two-layer, 64-wide decoder with grouped-query attention, a gated MLP, and its own 259-token byte-level vocabulary; an independent float64 CPU reference computes the expected greedy continuation. The fixtures exercise the `llama` adjacent-pair RoPE path, the `qwen2` Q/K/V bias path, and the `qwen3` per-head Q/K normalization plus half-split RoPE path.

On an RTX 5070 (adapter `blackwell`) **12/12 generated token ids matched** across a 24-token ChatML prompt — processed in prompt batches of eight — and three decode steps per model. The smallest top-2 logit margin in the reference was 0.19, so the comparison does not hinge on float32 rounding. The script prints the CPU tokens and skips the GPU half when no Chrome is installed; set `CHROME_PATH` to point at one. It is a correctness check, not a speed benchmark, and the synthetic models share no weights with real checkpoints.

## Current scope

- Implements dense text inference for `llama` (Llama 3.x, Mistral, SmolLM), `qwen2` (Qwen2/2.5), `qwen3`, `gemma3`, and hybrid `qwen35` (Qwen3.8 27B) architectures. Other architecture identifiers, including mixture-of-experts models, are rejected with an explicit message. Other sizes sharing a supported architecture are not all validated on hardware.
- Chat framing is taken from `tokenizer.chat_template` and cross-checked against the vocabulary; a file with an unrecognized template fails instead of producing a malformed prompt. Llama 3 and YaRN `rope.scaling` factors are accepted and currently left unapplied to the rotation, so extremely long contexts on those models lose some positional accuracy.
- Qwen chat runs in non-thinking mode. Images, video, tool execution, speculative MTP decoding, and extended reasoning controls are not exposed.
- Does not implement arbitrary model architectures, image/audio encoders, split GGUF files, or every GGML quantization format.
- Does not provide a CPU-only fallback if WebGPU is unavailable.
- Uses f32 attention state. Gemma local layers use bounded sliding caches; full-attention layers retain the configured context. Dense models keep a KV cache for the configured context, so a 32K context costs more GPU memory than a 2K one. Qwen allocates KV only for full-attention layers, plus fixed recurrent state. Runtime context is capped at 32,768 tokens even if a model advertises more. Hardware may require a lower setting.
- Conversations live in page memory and disappear on reload. Use Export to save a transcript.
- Model loading, token generation, and paging performance vary substantially with hardware and browser. It does not promise fast 27B inference on every GPU.

## References and licensing

The runtime is implemented in TypeScript and WGSL. GGUF and quantization block layouts follow [GGML/llama.cpp](https://github.com/ggml-org/llama.cpp); Gemma 3 computations follow its [model implementation](https://github.com/ggml-org/llama.cpp/blob/master/src/models/gemma3.cpp) and [Google's Gemma reference](https://github.com/google/gemma_pytorch). [svenflow/webgpu-gemma](https://github.com/svenflow/webgpu-gemma) was a design reference for browser Gemma execution; this project's bounded paging engine, block kernels, and SDK are implemented here.

Qwen's architecture and tokenizer follow the [official Qwen3.8 model](https://huggingface.co/Qwen/Qwen3.8-27B), [Transformers reference](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_5/modeling_qwen3_5.py), and [GGUF conversion conventions](https://github.com/ggml-org/llama.cpp/blob/master/conversion/qwen.py), including its value-head permutation and already adjusted normalization weights.

Code: [MIT](LICENSE). Model weights retain their original terms, including [Qwen3.8's Apache 2.0 license](https://huggingface.co/Qwen/Qwen3.8-27B/blob/main/LICENSE) and [Gemma's terms](https://ai.google.dev/gemma/terms). No model weights are redistributed by this repository. The small tokenizer test fixture contains vocabulary entries and merge rules derived from the Apache-licensed Qwen tokenizer for interoperability checks.
