# WebGPU

**A browser-native, paged inference runtime for Qwen3.8 27B and Gemma 3.**

WebGPU reads a local GGUF in small ranges and runs quantized transformer computations in WGSL shaders. It keeps a bounded portion of the model on the GPU and streams the rest from the local file. A model does not have to fit completely in VRAM or JavaScript memory.

This is an experimental runtime with a working browser UI and JavaScript SDK. It is not an Ollama inference frontend. No model execution runs in Node, Ollama, a native helper, or a cloud service. The project is named WebGPU and uses the browser's WebGPU standard; it is an independent project.

**Verified:** Gemma 3 27B generated `Paris` entirely in the browser with a **2 GiB engine buffer budget**, peaking at **1.983 GiB** of tracked GPU allocations on an RTX 5070. The one-token SDK smoke test took **120 seconds**, including prompt processing; 108 seconds were spent awaiting weight reads. This is a working experimental memory-saving runtime, not a promise of fast inference. See [measured results, screenshots, and test limits](docs/VALIDATION.md).

**Qwen3.8 27B verified:** the Q4_K_M model also generated `Paris` in the browser with a **2 GiB budget**, peaking at **1.985 GiB**. Its one-token SDK test took **181.9 seconds**, including the 23-token prompt; 164.8 seconds were spent awaiting weight reads. [Raw Qwen result](docs/qwen38-benchmark.json).

## Use it

Open [WebGPU](https://ahpah-dev.github.io/WebGPU/) in a browser with WebGPU hardware acceleration. Choose a local **Qwen3.8 27B or Gemma 3 GGUF** file, select your GPU allocation budget, and send a message. The model architecture is detected from the file. The selected file is read directly by a worker through the browser File API. It is never uploaded.

Start with a **2 GiB budget** and **512-token context**. The GPU must have additional headroom for the browser, driver, display, and temporary upload staging. The app does not automatically know how much free VRAM your browser can use.

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

Open `http://127.0.0.1:8787`, click **Open from Ollama**, and select an installed model by name. You can also choose a GGUF file. The server binds to loopback and serves static assets and model bytes; all inference still runs in the browser.

For development, `npm run dev` runs Vite. Use the file picker in development; the optional model byte route is provided by `npm start`.

### Use weights already downloaded by Ollama

The local app automatically discovers your Ollama library. No per-model startup argument, renaming, copying, or conversion is needed:

```sh
npm start
```

Click **Open from Ollama**, search for a model such as `qwen3.8:27b`, then select it. The picker displays its size and quantization. Models with unsupported architectures or quantization are shown with an explanation. **Refresh** discovers models downloaded since the picker was opened. Ollama itself does not need to be running; WebGPU reads its saved manifests and weight files.

If you need weights first:

```sh
ollama pull qwen3.8:27b
# Or:
ollama pull gemma3:27b
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
3. **Bounded weight pager.** A reusable 16 MiB slot streams uncached weight pages. A fixed resident prefix uses the remaining budget; it avoids the repeated full-cache eviction of naive LRU on sequential transformer scans.
4. **Ordered uploads.** GPU writes and dispatches are ordered. Fences after at most eight pages bound queued staging and allow uploads to overlap prior computation. Each transformer layer completes before progress advances.
5. **Batched prompt processing.** Up to 32 prompt tokens reuse each weight page. Gemma uses a GPU KV cache and conversation prefix reuse. Qwen retains KV state for full-attention layers and fixed-size recurrent/convolution state for its other layers. It rebuilds the prompt state for each new request so edited prompts, retries, and cancellation cannot reuse stale recurrent state.
6. **Gemma 3 decoder.** Implements embedding scaling, RMS normalization, Q/K normalization, local/global RoPE, grouped-query causal/sliding attention, residual connections, GELU feed-forward layers, and the tied output projection. The 27B attention scale is handled separately.
7. **Qwen3.8 decoder.** Implements causal depthwise convolution, L2-normalized Q/K, Gated DeltaNet with tiled GGUF value heads, gated RMS normalization, interleaved Q/gate projection, partial RoPE, gated grouped-query attention, SwiGLU, and the untied output projection. The tokenizer uses NFC normalization and ranked byte BPE with explicit non-thinking ChatML framing.
8. **Dedicated worker.** Sampling and tokenization run in JavaScript; transformer matrix/attention operations run on WebGPU. Both remain inside the browser. Stop cancels at the next asynchronous boundary, and disposal terminates the worker.

The GPU budget counts buffers allocated by this engine. It is **not a measurement of total process VRAM**: browser/driver allocations and temporary upload staging are additional. The runtime panel distinguishes file size, cumulative disk reads, resident cache hits, and tracked allocations.

## Browser SDK

`npm run build` emits ES modules and declarations to `dist/sdk/`. Host that folder alongside your app. No bundler or server inference route is required to consume it.

```js
import { WebGPU } from './sdk/sdk.js';

const model = new WebGPU();
model.onProgress = ({ detail, stats }) => console.log(detail, stats);

// file is a File from an <input type="file"> element.
const info = await model.load(file, {
  gpuBudgetMiB: 2048,
  contextLength: 512,
  batchSize: 32,
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

This API is a browser JavaScript API, not a network REST server. Only one load or generation can run at a time. Supported context lengths are 256, 512, 1024, and 2048 tokens. When a prompt does not fit, the caller receives an error; history is not silently discarded. When generation reaches the remaining context limit, it ends.

## Verification

```sh
npm test
npm run build
npm start -- --ollama-model gemma3:27b
```

Open `http://127.0.0.1:8787/verify.html` and select **Run kernel checks**. These checks execute real WebGPU kernels and compare their results with CPU references for all seven tensor types, batched/offset matrix operations, causal and sliding attention, RMS norm, and scaled RoPE. Qwen checks cover interleaved query/gates, causal convolution history, decay, recurrent state, batch versus sequential DeltaNet, gated normalization, SwiGLU, and partial RoPE. The standalone SDK smoke test loads whichever local model the byte server exposes and generates text; it is not a mocked inference test.

CPU tests cover independently packed quantization fixtures, non-finite sampling rejection, file bounds, HTTP range enforcement, invalid GGUF headers, real Qwen3.8 tensor shapes, MTP exclusion, and 17 reference tokenizer cases from Hugging Face's tokenizer. GitHub Actions runs these tests and the production build before publishing the static application.

Ollama integration tests exercise namespaced discovery, unsupported and incomplete models, malformed manifests, tag changes, selected-model HTTP byte ranges, opaque IDs, and origin restrictions. The browser picker has been checked with an actual installed `qwen3.8:27b`, including search, refresh, and loading into the runtime.

## Current scope

- Implements **Qwen3.8 27B** (`qwen35` GGUF) hybrid text inference and **Gemma 3** dense text inference. Other sizes sharing those architectures are not all validated on hardware. Ordinary Qwen3, Qwen3 MoE, and other architecture identifiers are rejected.
- Qwen chat runs in non-thinking mode. Images, video, tool execution, speculative MTP decoding, and extended reasoning controls are not exposed.
- Does not implement arbitrary model architectures, image/audio encoders, split GGUF files, or every GGML quantization format.
- Does not provide a CPU-only fallback if WebGPU is unavailable.
- Uses f32 attention state. Gemma caches all layers, including local layers; Qwen allocates KV only for full-attention layers, plus fixed recurrent state. Context is limited to 256–2048 tokens here, regardless of a model's advertised context.
- Conversations live in page memory and disappear on reload. Use Export to save a transcript.
- Model loading, token generation, and paging performance vary substantially with hardware and browser. It does not promise fast 27B inference on every GPU.

## References and licensing

The runtime is implemented in TypeScript and WGSL. GGUF and quantization block layouts follow [GGML/llama.cpp](https://github.com/ggml-org/llama.cpp); Gemma 3 computations follow its [model implementation](https://github.com/ggml-org/llama.cpp/blob/master/src/models/gemma3.cpp) and [Google's Gemma reference](https://github.com/google/gemma_pytorch). [svenflow/webgpu-gemma](https://github.com/svenflow/webgpu-gemma) was a design reference for browser Gemma execution; this project's bounded paging engine, block kernels, and SDK are implemented here.

Qwen's architecture and tokenizer follow the [official Qwen3.8 model](https://huggingface.co/Qwen/Qwen3.8-27B), [Transformers reference](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_5/modeling_qwen3_5.py), and [GGUF conversion conventions](https://github.com/ggml-org/llama.cpp/blob/master/conversion/qwen.py), including its value-head permutation and already adjusted normalization weights.

Code: [MIT](LICENSE). Model weights retain their original terms, including [Qwen3.8's Apache 2.0 license](https://huggingface.co/Qwen/Qwen3.8-27B/blob/main/LICENSE) and [Gemma's terms](https://ai.google.dev/gemma/terms). No model weights are redistributed by this repository. The small tokenizer test fixture contains vocabulary entries and merge rules derived from the Apache-licensed Qwen tokenizer for interoperability checks.
