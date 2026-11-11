# Case 3: modify existing GGUF layers

## Goal

Measure how a controlled change to existing layer weights changes loading, memory, speed, and output quality. Preserve the number of blocks and the original downloads. Small-artifact acquisition, inventory and a byte-identical no-change rewrite are complete; source/control inference is verified; the norm edit is serialized and verified, with edited-model inference pending.

## Method

GGUF describes named tensors, their dimensions/types, and architecture metadata. A tensor change must respect the block encoding and alignment of its quantization type. Use a fresh output file and record which tensor elements changed. [GGUF format specification](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md)

Proposed first experiment: choose one small existing floating-point normalization tensor, then compare an exact no-change rewrite with a single documented scaling perturbation. This separates file-writing damage from an intentional weight change. A later selected quantized tensor experiment would dequantize only that tensor, apply a defined change, and requantize with a supported encoder. These are proposed controls, not validated implementations.

## Tools and versions

Python 3.12.13 and gguf 0.19.0 are installed. GGUFReader/GGUFWriter and llama.cpp's quantization utilities are candidate tools. Inference uses llama-cpp-python 0.3.36 and its vendored llama.cpp commit from case 1. Pin any separate standalone llama.cpp build to the same source revision before using it. [gguf-py tools](https://github.com/ggml-org/llama.cpp/tree/master/gguf-py)

## Feasibility on the 8 GB Air

Memory-mapped metadata inspection and one small tensor copy should fit more readily than loading the whole model as floats. This remains a budget to measure. A complete BF16 language weight copy is about 15.23 GB; full-model dequantization is outside the local budget. Keep one inference process and enough disk for the unchanged original plus a complete edited artifact. Rewriting a 3.81 GB file still requires substantial disk I/O.

## Concrete plan

1. Inventory original architecture metadata, tensor names, shapes, and quantization types without modifying them.
2. Choose the smallest meaningful target tensor and predeclare the operation and expected numerical range.
3. Write a no-change copy; validate tensor metadata and data hashes against the source. Confirm loading and output equivalence on a fixed smoke prompt.
4. Write one edited copy, preserving all unrelated tensor bytes and required metadata. Record source/output SHA256, exact operation, seed, and target tensor statistics.
5. Run the same evaluation protocol as case 1 with one artifact loaded at a time. Compare unchanged-copy controls before attributing effects to the perturbation.

## Risks and open questions

A model can load successfully while producing damaged text. Quantization scales and packing make byte edits unsafe. A no-change rewrite must preserve tokenizer and chat metadata too. Which weight to perturb, the magnitude, and encoder support for its actual quant type require inspection. This experiment does not substitute for meaningful training.

## Findings

### 2026-11-11: small immutable source acquired

`src/case3_acquire.py` downloaded one small F16 GGUF directly into the local models folder, without a shared cache. The source is `bartowski/SmolLM2-135M-Instruct-GGUF`, revision `09816acd5d99df7be770d85ea30822623dab342c`, file `SmolLM2-135M-Instruct-f16.gguf`. Its 270,885,952 bytes match upstream LFS SHA256 `f535f83ec568d040f88ddc04a199fa6da90923bbb41d4dcaed02caa924d6ef57`. F16 avoids packed quantization for subsequent tensor and numerical experiments and remains modest on disk. This is a separate conversion of the tiny text-model family used in case 2, not the trained adapter or the 7B VL model. Equivalence to the case 2 BF16 checkpoint is not assumed.

Read-only gguf 0.19.0 inspection found 272 tensors, 134,515,008 stored elements, 211 F16 tensors and 61 F32 tensors. Metadata declares Llama architecture, 30 blocks, hidden width 576, feed-forward width 1,536, 9 attention heads, 3 KV heads, 64 RoPE dimensions, theta 100,000, RMSNorm epsilon approximately 1e-5, vocabulary 49,152 and context 8,192. `results/case3-source.json` preserves every tensor name/shape/type/byte count/hash and metadata value hashes, including tokenizer arrays and chat template. These hashes provide a reference for verifying unrelated bytes after rewrites.

Predeclared edit: multiply the existing F32 `blk.0.attn_norm.weight` by 1.01 in float32, after a no-change GGUFWriter control. This tensor has 576 elements, currently ranging from -0.3046875 to 0.314453125. No edit or rewrite has happened yet. Next: preserve all metadata and tensor bytes in a no-change rewrite, verify them, then check original/control loading and fixed-prompt output before the perturbation.

Acquisition plus hash/inventory took 14.788 seconds; peak process RSS was 494,092,288 bytes. Before/after snapshots were normal pressure with 69%/68% reported free. Download chunks and pre-inspection checks required normal pressure and at least 15% free; these are acquisition snapshots, not a continuous inference memory trace. RSS does not cover all unified-memory allocations. No model execution, server or training process started. `results/case3-acquire.runtime.log` preserves the measurement summary. Original downloads and all previous evidence remain unchanged.

Sources: [pinned conversion repository](https://huggingface.co/bartowski/SmolLM2-135M-Instruct-GGUF/tree/09816acd5d99df7be770d85ea30822623dab342c), [GGUF specification](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md). The repository API supplied the pinned revision and LFS digest; local bytes were independently checked. Acquisition is preparation, not a completed case 3 experiment.

## Sources consulted

The GGUF specification and gguf-py project linked above were consulted on 2026-11-11. The inference architecture reference is [llama.cpp Qwen2-VL](https://github.com/ggml-org/llama.cpp/blob/master/src/models/qwen2vl.cpp). Proposed experimental controls are project plans, not claims of upstream support.

Validation: independent full-file and per-tensor/metadata hash rechecks, preservation of all prior tracked results and the original manifest, Python compilation, notebook schema/code checks, dependency compatibility (131 packages), exactly seven main docs and diff whitespace checks passed. No background processes were started.

### 2026-11-11: no-change rewrite control verified

`src/case3_rewrite.py` used gguf 0.19.0 GGUFWriter to write `models/SmolLM2-135M-Instruct-control.gguf`, preserving metadata value types (including array element types), tensor order, shapes and data types. It refuses existing output/evidence paths. A fresh read verified all 272 tensor hashes and 36 reader fields (33 metadata entries plus three GGUF header fields) against the source; the source inventory also matched the saved acquisition evidence. The complete rewritten file is byte-identical: 270,885,952 bytes and SHA256 `f535f83ec568d040f88ddc04a199fa6da90923bbb41d4dcaed02caa924d6ef57`. The original was rehashed and remains unchanged. No packed quantized data was edited.

Total rewrite/verification time was 5.110 seconds, including 0.199 seconds writing. Peak process RSS was 909,770,752 bytes. All 276 phase/tensor-boundary snapshots showed normal pressure; reported free memory was at least 68%. These are snapshots rather than continuous monitoring, and RSS excludes some unified-memory allocations. No model was loaded or background process started. Raw evidence: `results/case3-rewrite.json` and `results/case3-rewrite.runtime.log`.

This establishes serialization preservation, not an observed inference comparison. Next, run original/control fixed-prompt inference sequentially, then create the predeclared 1.01 norm perturbation and compare outputs. Case 3 remains incomplete. Implementation followed the [upstream GGUFWriter source](https://github.com/ggml-org/llama.cpp/blob/master/gguf-py/gguf/gguf_writer.py), consulted on 2026-11-11, and the installed pinned implementation.


### 2026-11-11: sequential inference control verified

`src/case3_control.py` loaded the immutable original and byte-identical rewrite sequentially, explicitly closing each before the next. Both hashes were verified before and after execution. The fixed smoke input is dataset index 6, with the case 1 system prompt and only the scene description in messages. References never enter prompts. Settings: CPU-only mmap, context 2,048, batch 128, four prompt/decode threads, seed 42, greedy decoding, top-p/top-k/repeat penalty 1 and a 128-new-token cap. The embedded chat template was used and preserved in evidence. This short control is separate from the 1,024-token case 1 protocol and the MLX case 2 results.

| Measurement | Original | Rewrite control |
| --- | ---: | ---: |
| Constructor seconds | 0.08589 | 0.08109 |
| Input wall seconds | 0.94285 | 0.96484 |
| Prompt evaluation tokens | 74 | 74 |
| Generation evaluation tokens | 127 | 127 |
| Generation evaluation seconds | 0.83345 | 0.85919 |
| Process high-water RSS bytes | 460,947,456 | 463,208,448 |
| Finish reason | length | length |

Both outputs are exactly equal and repeatedly emit the Naisha/Zain tea-stall scene title rather than useful shot structure. This establishes successful loading and text equality on one smoke prompt, not numerical logit parity, quality, or a general speed comparison. The 128-token cap includes the first generated token while llama.cpp generation evaluation counts the subsequent 127 evaluations. Process RSS is cumulative across sequential loads and excludes some unified-memory allocations.

All 12 phase/periodic readings showed normal pressure and 69% reported free, including before load, before input, during generation and after close. A 0.5-second watcher stops streaming on critical pressure or below 8% free (and on unreadable headroom); the existing normal-pressure/15% guard applies before loading and each input. Total execution plus hash checks took 2.870 seconds. Both models closed, the watcher joined, and the process exited. Raw outputs, template, runtime versions, timings and readings are in `results/case3-control.json`; upstream runtime whitespace is preserved in `results/case3-control.runtime.log`.

The [llama-cpp-python API reference](https://llama-cpp-python.readthedocs.io/en/latest/api-reference/) was consulted for sequential loading and streamed chat completion. Next: create the predeclared float32 norm scaling, verify only that tensor changes, then compare using these same settings and prompt. Case 3 remains incomplete until that measured perturbation runs.


### 2026-11-11: predeclared norm perturbation serialized and verified

`src/case3_perturb.py` created `models/SmolLM2-135M-Instruct-norm101.gguf` with the predeclared float32 multiplication by 1.01 on `blk.0.attn_norm.weight` (shape 576, F32). All 576 elements changed; the range changed from [-0.3046875, 0.314453125] to [-0.3077343702, 0.3175976574], with maximum absolute delta 0.0031445324. No packed quantized bytes were edited. The script follows the installed GGUFWriter implementation and previously verified rewrite control, preserving metadata types, tensor order, shapes and types.

Fresh-reader verification found exactly one changed tensor, 271 byte-identical unrelated tensors and all 36 reader fields unchanged, including architecture/tokenizer metadata. The edited tensor exactly equals the prescribed float32 operation. Target SHA256 changed from `2469079d969a361823be8145fa36e15e4c1b933b5261d669357b3deead825cba` to `1f841112c39dddf5fb3943db40d972a9ae1ff3036cedee6973041b19a2e21145`. The edited file remains 270,885,952 bytes; its SHA256 is `336f76366de192df5dc66185ea0caf98b00a1b75ba03d9089aa4dd481358b49e`. Source hash and acquisition inventory were rechecked unchanged. Existing artifact/evidence paths are refused rather than overwritten.

Write time was 0.208 seconds; total rewrite/verification took 4.923 seconds. Process peak RSS was 909,721,600 bytes. All 276 phase/tensor snapshots showed normal pressure and at least 69% reported free memory. Snapshots are not continuous monitoring; RSS excludes some unified-memory allocations. No inference model or background process started. Raw measurements and hashes are in `results/case3-perturb.json` and its runtime log.

Case 3 remains incomplete: the edited file has not loaded or generated output. Next, compare its load and fixed index-6 smoke output with `results/case3-control.json`, using identical CPU/settings/prompt and the same memory guards. No behavior or quality change is established by the artifact edit alone.
