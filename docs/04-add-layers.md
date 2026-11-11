# Case 4: add transformer layers

## Goal

Test whether adding trainable capacity improves the scene task. Distinguish a file that loads, a forward pass that preserves behavior, and a trained model that improves quality. The local experiment begins with a tiny text-model GGUF duplication control.

## Method

Block expansion has research precedent: LLaMA Pro adds transformer blocks and tunes the expanded blocks. That evidence concerns its studied LLaMA setup, not an automatic guarantee for Qwen2.5-VL or this dataset. [LLaMA Pro, version 2](https://arxiv.org/abs/2401.02415v2)

For this project, start with an architecture-aware floating-point implementation. Add one compatible decoder block with a deliberately chosen initialization. Investigate residual-branch initialization intended to preserve the original function, and verify it numerically before training. Only after the expanded model behaves correctly should conversion to GGUF be attempted. Update block count, tensor indexes, and every affected execution assumption together; editing a metadata count alone adds no useful computation.

## Tools and versions

Baseline: Python 3.12.13, gguf 0.19.0, llama-cpp-python 0.3.36, and the llama.cpp commit recorded in case 1. MLX 0.32.4 and MLX-LM 0.32.2 are installed for case 2; this bounded artifact experiment uses gguf-py only. The [Qwen configuration](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/blob/main/config.json) and [llama.cpp implementation](https://github.com/ggml-org/llama.cpp/blob/master/src/models/qwen2vl.cpp) are the architecture references.

## Feasibility on the 8 GB Air

Adding blocks increases weight, KV-cache, and compute costs. It may also push the language artifact beyond the 4 GB target, which must be reported rather than hidden by changing the quant. Local 7B floating-point expansion and training exceed practical RAM. A tiny architecture demonstrator could be local; the full 7B expansion/training route needs larger external hardware. Tiny artifact expansion has now started; full 7B expansion/training remains unattempted.

## Concrete plan

1. Establish forward-logit parity for the unmodified decoded architecture as described in case 6.
2. Determine exact block tensor shapes and attention/position requirements. Freeze a test input set and choose where one block will be inserted.
3. Build one initialized block in floating point. Check the intended initial output equivalence and verify gradients on a small synthetic case.
4. Freeze original weights if the research method requires it, then train only the chosen new parameters on a properly split dataset with adequate hardware.
5. Convert to a new GGUF, update all layer indexes/counts, and check loader compatibility. If the runtime needs source changes, record and pin those changes explicitly.
6. Compare file size, peak memory, time, and held-out quality with the same baseline settings and human rubric.

## Risks and open questions

Randomly inserting a block can destroy learned behavior. A suitable initialization for this architecture must be proven by a forward check. Extra depth may not help 150 examples and may overfit. Loader assumptions, exporter support, insertion location, training budget, and the desired file-size constraint after expansion are open questions.

## Findings

### 2026-11-11: tiny GGUF block duplication, serialization only

The bounded local route deliberately differs from the longer function-preserving training proposal above. It appends an exact copy of the last decoder block of the immutable case 3 SmolLM2-135M F16 source. This is an untrained depth-change control, not LLaMA Pro initialization or 7B VL expansion. The insertion was chosen before expanded-model output inspection. Blocks 0-29 retain their indexes; new block 30 copies block 29. All nine added tensors retain source shapes, F16/F32 types and exact bytes. A duplicated residual block is not an identity mapping, so behavior preservation is not expected or claimed.

Command: `.venv/bin/python src/case4_expand.py > results/case4-expand.runtime.log 2>&1`. Source repository/revision/hash remain those in `results/case3-source.json`. The script refuses overwrites, verifies source integrity against the saved inventory, and writes a separate artifact. The pinned [llama.cpp Llama implementation](https://github.com/ggml-org/llama.cpp/blob/0c1e57098bba43ac29e6e3b677cdceebdd22334f/src/models/llama.cpp), consulted on 2026-11-11, constructs and executes indexed blocks using the declared layer count; this informed the metadata/tensor map, but does not establish loader success.

| Measurement | Observed |
| --- | --- |
| Blocks / tensors | 30 to 31 / 272 to 281 |
| Added stored elements / tensor bytes | 3,540,096 / 7,082,496 |
| Source / expanded file bytes | 270,885,952 / 277,968,992 |
| File increase including tensor records and alignment | 7,083,040 bytes |
| Changed reader fields | `llama.block_count` 30 to 31; `GGUF.tensor_count` 272 to 281 |
| Preservation | All 272 original tensors and 34 other reader fields unchanged |
| Rewrite plus verification / writing | 5.054 / 0.198 seconds |
| Peak process RSS | 917,307,392 bytes |
| Memory snapshots | 285 readings; normal pressure throughout; at least 69% reported free |

Expanded artifact: `models/SmolLM2-135M-Instruct-plus1.gguf`, SHA256 `bb7594b5fbcddc3dda1b958d762a3df77624fd2ca7cf048c4d27ccf1561cc664`. Raw evidence `results/case4-expand.json` records the full block/tensor map, added tensor hashes, memory readings and tool version; the runtime log is retained. The original file hash remains unchanged. Phase/tensor snapshots are not continuous monitoring; RSS omits some unified-memory allocations.

No model was loaded, no logits/output were measured, and no parameters were trained. Loading and comparison were pending at this serialization checkpoint; the measured follow-up is below. A successful file rewrite proves neither runtime compatibility nor useful added capacity. No background process was started.

### 2026-11-11: expanded-model CPU smoke comparison

Command: `.venv/bin/python src/case3_control.py --expanded > results/case4-expanded.runtime.log 2>&1`. The shared guarded inference code verifies the expanded artifact and saved source-control evidence hashes, then checks identical scene index 6, system prompt, context, greedy decoding, seed, CPU settings, runtime version and chat template. It reuses the saved source output rather than loading both models together. References stay outside prompts. Raw evidence is `results/case4-expanded.json`.

| Measurement | Original source control | Expanded model |
| --- | ---: | ---: |
| Constructor seconds | 0.08589 | 0.07885 |
| Input wall seconds | 0.94285 | 0.81778 |
| Prompt / generation evaluation tokens | 74 / 127 | 74 / 127 |
| Output words / SHOT headings | 66 / 0 | 103 / 0 |
| Unigram F1 | 0.08560 | 0.11978 |
| Finish reason | length | length |
| Peak process RSS bytes | 460,947,456 | 472,039,424 |

The 31-block artifact loaded successfully and changed greedy text. Instead of the source's repeated scene title, it invents a tea spill and repeats waiting/anxiety and dialogue-like fragments. Neither output provides the requested shot structure. Increased lexical overlap does not demonstrate semantic improvement or useful capacity. No training or logit parity was attempted. These separate single runs have uncontrolled cache/thermal conditions; their timings do not establish a speed advantage from adding depth.

Expanded prompt/generation evaluation took 0.065812/0.733039 seconds; the complete run including hash checks took 1.422 seconds. All five before/during/after readings showed normal pressure and 69% reported free memory. RSS excludes some unified-memory allocations and short-run sampling can miss transients. The source and expanded hashes remain unchanged. The model explicitly closed and the watcher joined before process exit. This completes the bounded case 4 artifact/load/output experiment; useful added capacity and trained expansion remain unestablished. Cases 5-6 remain pending.

## Sources consulted

LLaMA Pro v2, the Qwen configuration, and the llama.cpp Qwen2-VL implementation linked above were consulted on 2026-11-11. The proposed sequence and feasibility estimates are project reasoning, not measured Qwen expansion results.
