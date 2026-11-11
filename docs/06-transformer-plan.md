# Case 6: rebuild the decoded transformer and train it

## Goal

Understand the downloaded model well enough to implement its forward architecture ourselves, check the implementation against a trusted runtime, and later train a model with our own code. Architecture reproduction and training from scratch are different milestones. Planning only in this task.

## Method

Read GGUF architecture metadata, tensor names/shapes/types, tokenizer, and chat template. Reconstruct computation from those observations together with the published Qwen configuration and runtime implementation. Weights do not reveal the original corpus, optimizer history, or all training choices.

The published Qwen2.5-VL-7B configuration lists 28 language blocks, hidden width 3,584, feed-forward width 18,944, 28 attention heads, 4 KV heads, vocabulary size 152,064, RMSNorm epsilon 0.000001, and MRoPE sections 16/24/24. Its vision configuration lists 32 blocks with hidden width 1,280 and output width 3,584. These are upstream configuration facts, not yet an inventory measured from our GGUF. The historical `transformers_version` field is not a recommendation for a current install. [Official configuration](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/blob/main/config.json)

Reproduce text tokenization, embeddings, RMSNorm, grouped-query attention, positional encoding, causal masking, gated feed-forward blocks, residuals, and output projection. Start with text-only forward parity. Image patch encoding, vision blocks, positional handling, and the projector are an additional milestone. Consult the [Qwen2-VL runtime implementation](https://github.com/ggml-org/llama.cpp/blob/master/src/models/qwen2vl.cpp) rather than inferring every operation from tensor shapes.

## Tools and versions

Python 3.12.13, gguf 0.19.0, and the case 1 llama-cpp-python 0.3.36/runtime commit are the installed inspection/reference stack. Choose either PyTorch or MLX for the new implementation after testing required operations; neither training framework is installed or version-selected. Before implementation, pin the framework and the exact trusted model/config/source revisions. The source configuration consulted here is on upstream main as of 2026-11-11 and must be pinned for a parity experiment.

## Feasibility on the 8 GB Air

Read-only GGUF metadata/tensor inspection and a tiny educational transformer are realistic candidates. A full 7B floating-point reconstruction needs roughly 14 GB of weights before execution buffers and exceeds available RAM. Training that architecture from scratch is far beyond this hardware and a 150-example dataset. Layer-by-layer checks might reduce resident weights, but require measurement and do not make full training feasible. Quantized decoded weights also differ from the original floating-point checkpoint.

## Concrete plan

1. Save an architecture/tensor inventory and compare it with the source configuration. Resolve naming, transposition, quantization, tokenizer, and position details explicitly.
2. Implement a tiny shape-compatible test block in our own framework code. Verify individual operations and gradients with small synthetic tensors.
3. For selected real tensors, check dequantization and intermediate activations against the same quantized reference. Define absolute/relative tolerances before judging parity.
4. Establish a one-block text forward check, then grow toward full forward parity on suitable hardware. A matching greedy sentence alone is insufficient evidence; compare logits and intermediate values.
5. Add vision encoding/projector parity only if image support becomes a requirement, with a separate image dataset.
6. Build an explicit training loop for a tiny model first. Freeze train/test data, record architecture, initialization, optimizer, loss, memory, speed, and held-out quality. Treat a full-scale reconstruction/training run as a separately budgeted project.

## Risks and open questions

Quantization error can be confused with implementation error. Missing tokenizer special tokens, wrong tensor orientation, MRoPE, attention masks, or KV-cache layout can produce plausible but incorrect outputs. Decide whether the target is Qwen reproduction or a new architecture, whether weights are reused or randomly initialized, whether vision is required, and what corpus/compute is available. The small scene dataset can evaluate a narrow task; it cannot pretrain a capable 7B model.

## Findings

Not run. Reserved: decoded inventory, confirmed architecture facts, operation tests, numerical tolerances/logit errors, framework versions, training configuration, and held-out results. No custom transformer has been implemented or trained.

## Sources consulted

The official Qwen configuration, [model card](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct), Qwen2-VL runtime implementation, and [GGUF specification](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md) were consulted on 2026-11-11. Proposed milestones and memory estimates are project reasoning rather than completed results.
