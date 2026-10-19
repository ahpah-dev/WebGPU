# Validation — 2026-10-19

## Qwen3.8 27B generation

The real `qwen3.8:27b` Q4_K_M GGUF generated `Paris` using the standalone browser SDK. The full text model ran in WGSL/JavaScript inside the browser; the loopback server supplied file ranges only. [Raw benchmark record](qwen38-benchmark.json).

| Item | Observed value |
| --- | --- |
| Model | Qwen3.8 27B 0814, mixed Q4_K / Q6_K / F32 GGUF |
| Architecture | `qwen35`: 48 recurrent + 16 full-attention layers |
| Text parameters evaluated | 26,895,998,464; optional MTP layer excluded |
| GGUF file | 16,810,714,464 bytes |
| SHA-256 | `f5f1dd8920d417aac2718b0bda3403da274301efdd6760b4f0f4b864ff2ad57d` |
| Host | Windows, RTX 5070, 12 GB VRAM, 16 GB RAM |
| GPU buffer budget / context | 2,048 MiB / 512 tokens |
| Prompt batch | 32 maximum; 23 prompt tokens, non-thinking ChatML |
| Prompt | `Name the capital of France. Answer with one word.` |
| Sampling | Greedy; one-token SDK smoke test |
| Output | `Paris` (token 57590) |
| Time to first token | 181.9001 seconds, including prompt |
| Time awaiting weight reads | 164.7527 seconds |
| Weight bytes read | 15,810,662,400 |
| Page uploads | 1,223 |
| Resident weight cache | 1,859,842,176 bytes |
| Peak tracked GPU buffers | 2,131,695,744 bytes (1.985 GiB) |

The file contains an optional prediction layer, and input embedding rows are read individually; these are not traversed as full matrices. Consequently the bytes read are smaller than the complete GGUF. This is a smoke test, not a broad quality, long-context, or throughput benchmark. The test used a 12 GB GPU with a 2 GiB engine budget, not an actual 2 GB GPU. Browser/driver overhead and upload staging are additional.

A separate UI run selected the actual weight file through the browser File API, without HTTP requests for model weights. With the same 2 GiB budget, 512-token context, and greedy prompt, it generated `Paris` after **160.2 seconds**, then evaluated the next token and ended its turn normally. This exercises both batched prefill and a subsequent decoding step carrying recurrent/convolution state. The UI displayed **1.99 GiB** peak tracked allocations. These two input paths were tested in different runs; their timings are not a controlled performance comparison.

![Qwen3.8 direct-file chat completed in the browser](qwen38-browser.jpg)

![Qwen3.8 memory allocations and hybrid layers](qwen38-memory.jpg)

## Gemma 3 27B generation

The model was evaluated inside the browser using this repository's TypeScript/WGSL runtime. No requests were sent to an Ollama inference endpoint. Ollama was used to download the weights only.

| Item | Observed value |
| --- | --- |
| Model | Gemma 3 27B, mixed Q4_K / Q6_K / F32 GGUF |
| Parameters | 27,009,007,616 |
| Weight file | 16,555,661,472 bytes (15.42 GiB) |
| GGUF SHA-256 | `3eb75691d2db8bb04109c517466bb10bc0e4f035227d1a46cbba2f1d8e6afa45` |
| Host | Windows, RTX 5070 with 12 GB VRAM, 16 GB system RAM |
| Browser adapter | `blackwell`, Chromium-based browser |
| Configured GPU buffer budget | 2,048 MiB |
| Context | 512 tokens |
| SDK prompt batch | 32 tokens maximum; 21 prompt tokens in this run |
| Temperature | 0 (greedy) |
| Prompt | `Name the capital of France. Answer with one word.` |
| Generated token | `Paris` (token 50429) |
| Time to first token, including prompt | 120.2693 seconds |
| Cumulative time awaiting weight reads | 107.8082 seconds |
| Weight bytes read | 16,534,430,010 |
| Page uploads | 1,121 |
| Resident weight cache | 1,572,541,992 bytes |
| Peak engine-tracked GPU buffers | 2,129,510,960 bytes (1.983 GiB) |

The SDK smoke test intentionally stops after one generated token. This validates the full 62-layer forward pass and output projection, but is not a long-conversation, quality, or throughput benchmark. The file bytes in this SDK run came from a loopback **static range server**, which does no inference. The worker performed tokenization, every transformer layer, sampling, and decoding in the browser.

An earlier UI test selected the same local file directly through the browser File API, with **no model-file HTTP requests**. It generated `Paris.` (two tokens) with peak tracked allocations displayed as **1.98 GiB**. Before the scheduling/batching change, that test took 269.0 seconds to the first token and 405.9 seconds to the second; it was then cancelled. These are different runs and input paths, not a controlled speed comparison.

![Direct-file browser inference](browser-27b.jpg)

![Measured memory usage](runtime-memory.jpg)

The budget counts engine-created GPU buffers, not total process VRAM. Browser/driver allocations, upload staging, and the OS file cache are additional. The test shows a 27B model running within a small **engine allocation budget on a 12 GB GPU**; it does not establish that an actual 2 GB card will run it, or that all weak GPUs will perform similarly.

## Automated checks

- **21 Node tests passed:** independently packed quantization fixtures; half floats; invalid logits; Unicode fallback and SentencePiece merging; control token isolation; File API bounds; HTTP 206 enforcement; truncated ranges; invalid GGUF files; real Qwen3.8 tensor shapes and MTP exclusion; 17 independent Hugging Face tokenizer examples including NFC, multilingual text, combining marks, emoji, and whitespace; Ollama catalog and selected-model byte-serving integration tests.
- **12 real WebGPU checks passed:** all seven supported matrix tensor formats; batched matrix input/output offsets; causal grouped-query attention; sliding windows; RMS normalization; QK normalization and scaled RoPE; ordered reuse of a weight buffer between dispatches.
- **Qwen GPU checks passed:** interleaved query/gate split, sigmoid output gating, causal depthwise convolution with carried history, L2 Q/K normalization, stable decay and beta, tiled value heads, batched versus sequential Gated DeltaNet with nonzero initial state, state clearing, gated RMS norm, SwiGLU, and partial RoPE (64 of 256 dimensions) at nonzero positions.
- **Type check, production bundle, standalone SDK emission passed.**
- **Byte-server checks passed:** valid 206 range; rejection of a whole-model GET; invalid-range 416; cross-origin 403.
- The standalone SDK was loaded as native browser ES modules and generated the result above.

## Opening installed Ollama models

The local app discovered the installed `qwen3.8:27b` from its Ollama manifest and opened its existing 16,810,714,464-byte GGUF by name. Browser checks covered opening the picker, search, no results, Refresh, and loading the selected model until the runtime reported ready. No file dialog, copy, conversion, or per-model server argument was used. Model reads use the catalog's bounded range route; inference remains in the browser.

Automated integration checks use isolated model-store fixtures for multiple models and namespaces, unsupported architectures, missing weights, malformed manifests, changed tag identities, correct 206 byte ranges, invalid ranges, unknown model IDs, and cross-origin rejection.

![Open installed models directly from Ollama](ollama-library.jpg)

## Remaining limits

No broad model-quality evaluation, long-context stress test, actual 2 GB GPU test, or cross-browser/hardware matrix was performed. Arbitrary architectures, image/audio encoders, split GGUF files, and unimplemented quantization formats are not supported. The first run is slow; the measured workload spent most of its time reading model weights.
