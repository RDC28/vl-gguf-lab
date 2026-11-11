# Case 5: remove transformer layers

## Goal

Measure the size/speed/quality trade-off from reducing decoder depth while preserving a valid model artifact. Planning only in this task.

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

Not run. Reserved: ranking method, calibration split hash, removed indexes, metadata changes, output hashes, load behavior, speed/memory, and quality deterioration. No layers have been removed.

## Sources consulted

ShortGPT v3, gguf-py, and the Qwen2-VL runtime source linked above were consulted on 2026-11-11. The workflow and local feasibility statements are proposals and resource estimates, not completed pruning results.
