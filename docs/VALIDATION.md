# Validation — 2026-10-19

## Real 27B generation

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

- **12 Node tests passed:** independently packed quantization fixtures; half floats; invalid logits; Unicode fallback and SentencePiece merging; control token isolation; File API bounds; HTTP 206 enforcement; truncated ranges; invalid GGUF files.
- **12 real WebGPU checks passed:** all seven supported matrix tensor formats; batched matrix input/output offsets; causal grouped-query attention; sliding windows; RMS normalization; QK normalization and scaled RoPE; ordered reuse of a weight buffer between dispatches.
- **Type check, production bundle, standalone SDK emission passed.**
- **Byte-server checks passed:** valid 206 range; rejection of a whole-model GET; invalid-range 416; cross-origin 403.
- The standalone SDK was loaded as native browser ES modules and generated the result above.

## Remaining limits

No broad model-quality evaluation, long-context stress test, actual 2 GB GPU test, or cross-browser/hardware matrix was performed. Arbitrary architectures, image/audio encoders, split GGUF files, and unimplemented quantization formats are not supported. The first run is slow; the measured workload spent most of its time reading model weights.
