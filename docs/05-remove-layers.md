# Case 5: remove transformer layers

## Goal

Measure the size/speed/quality trade-off from reducing decoder depth while preserving a valid model artifact. The first bounded experiment uses the tiny F16 text model from case 3.

## Method

ShortGPT proposes measuring block influence and removing low-influence layers. Its published results motivate an experiment, but do not identify safe layers in this Qwen VL model. [ShortGPT, version 3](https://arxiv.org/abs/2403.03853v3)

Use a training/calibration split to rank candidate blocks, keeping held-out evaluation separate. A simpler first control would remove one predetermined interior block and measure the damage. In a new artifact, remove all tensors belonging to the selected block, renumber remaining blocks consistently, and update the declared block count. Inspect runtime attention and position assumptions before relying on the edited file. A loader accepting the file is only the first check.

## Tools and versions

Installed inspection/inference stack: Python 3.12.13; gguf 0.19.0; llama-cpp-python 0.3.36 with the llama.cpp commit recorded in case 1. Activation-based ranking likely needs a floating-point model with hooks in PyTorch/Transformers or a dedicated runtime instrument. Those training/analysis packages are not installed or version-selected. [gguf-py](https://github.com/ggml-org/llama.cpp/tree/master/gguf-py) and [Qwen2-VL runtime source](https://github.com/ggml-org/llama.cpp/blob/master/src/models/qwen2vl.cpp) are implementation references.

## Feasibility on the 8 GB Air

A streaming file rewrite and inference of a smaller quantized model may fit. Loading the full original floating-point model to collect activations does not fit safely. Holding both original and pruned models in RAM is avoided. Activation capture can itself be large, so a smaller model or larger external machine is the likely calibration route. File rewrite and runtime peak memory must still be measured.

## Concrete plan

1. Inventory decoder blocks and identify a runtime-compatible removable block. Keep vision/projector layers outside the first experiment.
2. Freeze a removal rule without looking at held-out scores. Decide whether to use a simple fixed-block control or a calibrated influence ranking.
3. Write one new artifact with consistent tensor names/indexes and architecture metadata. Preserve the source model and record hashes and the old-to-new index map.
4. Check tensor inventory and loading, then run a one-prompt smoke check before the held-out comparison.
5. Measure quality, file size, load time, peak memory, and throughput with only one model loaded at a time. Stop removing blocks if behavior collapses; report the failure rather than only successful samples.
6. Consider recovery fine-tuning only as a later separately authorized training experiment, with training effects reported separately from pruning.

## Risks and open questions

Depth reduction can break factual recall, instruction following, and vision-language alignment even if text generation looks fluent. Calibration on 150 similar examples may rank layers poorly. Whether block removal is accepted by the exact converter/runtime, which blocks matter, and whether useful speed gains occur on this hardware remain unmeasured.

## Findings

### 2026-11-11: verified fixed interior-block removal

Before any pruned-output inspection, `src/case5_remove.py` fixes original block 15 for removal from the immutable case 3 SmolLM2-135M F16 GGUF. This is a simple damage control, without activation ranking, calibration, recovery training or held-out score selection. Original blocks 0-14 retain their indexes; original blocks 16-29 become new blocks 15-28. Embeddings, output norm and tokenizer remain unchanged. No Qwen VL weights are removed.

The pinned [Llama runtime source](https://github.com/ggml-org/llama.cpp/blob/0c1e57098bba43ac29e6e3b677cdceebdd22334f/src/models/llama.cpp), inspected before coding, constructs tensors and residual blocks by declared layer count and index. An unrecognized depth is labeled unknown model type; this alone does not prove loading succeeds. This artifact has homogeneous attention/FFN shapes and no per-layer position metadata requiring adjustment. Actual runtime loading is the next check.

| Measurement | Observed |
| --- | --- |
| Source | Case 3 immutable 30-block F16 artifact, SHA256 `f535f83ec568d040f88ddc04a199fa6da90923bbb41d4dcaed02caa924d6ef57` |
| Output | `models/SmolLM2-135M-Instruct-minus1.gguf` |
| Output SHA256 | `391046ca968852f26f72402386b5732f562fb37719e660d031c6f460629964c4` |
| Blocks / tensors | 30 / 272 before; 29 / 263 after |
| Removed | Nine block-15 tensors, 3,540,096 elements, 7,082,496 tensor bytes |
| File bytes | 263,802,944, a reduction of 7,083,008 (2.61%) |
| Metadata | Only `llama.block_count` and header tensor count changed; 34 other fields preserved |
| Retained tensor verification | All 263 names mapped, with matching shapes, types and byte hashes |
| Rewrite / total verification time | 0.199 / 5.263 seconds |
| Peak process RSS | 903,086,080 bytes |
| Memory readings | 267 snapshots, all normal pressure; minimum 70% reported free |

Run: `.venv/bin/python src/case5_remove.py > results/case5-remove.runtime.log 2>&1`. The script refuses to overwrite artifacts/evidence and rechecks the original full-file hash after writing. It uses gguf 0.19.0 and the existing memory guard. `results/case5-remove.json` preserves the complete index/tensor map, removed tensor inventory, hashes and memory readings. Phase/tensor snapshots are not continuous monitoring, and RSS excludes some unified-memory allocations. No model was loaded and no training ran. Loading, outputs and inference timing remain unmeasured, so case 5 is incomplete and no efficiency or quality improvement is claimed.

## Sources consulted

ShortGPT v3, gguf-py, and the Qwen2-VL runtime source linked above were consulted on 2026-11-11. The workflow and local feasibility statements are proposals and resource estimates, not completed pruning results.
