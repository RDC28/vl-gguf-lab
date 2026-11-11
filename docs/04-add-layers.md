# Case 4: add transformer layers

## Goal

Test whether adding trainable capacity improves the scene task. Distinguish a file that loads, a forward pass that preserves behavior, and a trained model that improves quality. Planning only in this task.

## Method

Block expansion has research precedent: LLaMA Pro adds transformer blocks and tunes the expanded blocks. That evidence concerns its studied LLaMA setup, not an automatic guarantee for Qwen2.5-VL or this dataset. [LLaMA Pro, version 2](https://arxiv.org/abs/2401.02415v2)

For this project, start with an architecture-aware floating-point implementation. Add one compatible decoder block with a deliberately chosen initialization. Investigate residual-branch initialization intended to preserve the original function, and verify it numerically before training. Only after the expanded model behaves correctly should conversion to GGUF be attempted. Update block count, tensor indexes, and every affected execution assumption together; editing a metadata count alone adds no useful computation.

## Tools and versions

Baseline: Python 3.12.13, gguf 0.19.0, llama-cpp-python 0.3.36, and the llama.cpp commit recorded in case 1. A future PyTorch/Transformers or MLX implementation and adapter-training stack are not installed or version-selected. Pin them together after architecture support is established. The [Qwen configuration](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/blob/main/config.json) and [llama.cpp implementation](https://github.com/ggml-org/llama.cpp/blob/master/src/models/qwen2vl.cpp) are the architecture references.

## Feasibility on the 8 GB Air

Adding blocks increases weight, KV-cache, and compute costs. It may also push the language artifact beyond the 4 GB target, which must be reported rather than hidden by changing the quant. Local 7B floating-point expansion and training exceed practical RAM. A tiny architecture demonstrator could be local; the full 7B expansion/training route needs larger external hardware. Neither is started here.

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

Not run. Reserved: insertion map, new parameter count, initialization/parity checks, training settings, source revisions, artifact hashes, memory/time, and quality table. No layers have been added.

## Sources consulted

LLaMA Pro v2, the Qwen configuration, and the llama.cpp Qwen2-VL implementation linked above were consulted on 2026-11-11. The proposed sequence and feasibility estimates are project reasoning, not measured Qwen expansion results.
