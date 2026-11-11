# Case 6: rebuild the decoded transformer and train it

## Goal

Understand the downloaded model well enough to implement its forward architecture ourselves, check the implementation against a trusted runtime, and later train a model with our own code. Architecture reproduction and training from scratch are different milestones. The bounded local target is the unchanged case 3 SmolLM2-135M F16 GGUF; the 7B VL model remains outside the reconstruction memory budget.

## Method

Read GGUF architecture metadata, tensor names/shapes/types, tokenizer, and chat template. Reconstruct computation from those observations together with the published Qwen configuration and runtime implementation. Weights do not reveal the original corpus, optimizer history, or all training choices.

The published Qwen2.5-VL-7B configuration lists 28 language blocks, hidden width 3,584, feed-forward width 18,944, 28 attention heads, 4 KV heads, vocabulary size 152,064, RMSNorm epsilon 0.000001, and MRoPE sections 16/24/24. Its vision configuration lists 32 blocks with hidden width 1,280 and output width 3,584. These are upstream configuration facts, not yet an inventory measured from our GGUF. The historical `transformers_version` field is not a recommendation for a current install. [Official configuration](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/blob/main/config.json)

Reproduce text tokenization, embeddings, RMSNorm, grouped-query attention, positional encoding, causal masking, gated feed-forward blocks, residuals, and output projection. Start with text-only forward parity. Image patch encoding, vision blocks, positional handling, and the projector are an additional milestone. Consult the [Qwen2-VL runtime implementation](https://github.com/ggml-org/llama.cpp/blob/master/src/models/qwen2vl.cpp) rather than inferring every operation from tensor shapes.

## Tools and versions

Python 3.12.13, gguf 0.19.0, and the case 1 llama-cpp-python 0.3.36/runtime commit are the installed inspection/reference stack. Choose either PyTorch or MLX for the new implementation after testing required operations; MLX 0.32.4 is already installed and pinned from case 2. Before implementation, pin the framework and the exact trusted model/config/source revisions. The source configuration consulted here is on upstream main as of 2026-11-11 and must be pinned for a parity experiment.

## Feasibility on the 8 GB Air

Read-only GGUF metadata/tensor inspection and a tiny educational transformer are realistic candidates. A full 7B floating-point reconstruction needs about 15.23 GB of language weights before execution buffers and exceeds available RAM. Training that architecture from scratch is far beyond this hardware and a 150-example dataset. Layer-by-layer checks might reduce resident weights, but require measurement and do not make full training feasible. Quantized decoded weights also differ from the original floating-point checkpoint.

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

### 2026-11-11: fixed-token reference capture

`src/case6_reference.py` saved the unchanged tiny F16 source's logits to `results/case6-reference-logits.npz`, with settings, token pieces, hashes and measurements in `results/case6-reference.json` and the raw runtime log. Source SHA256 remains `f535f83ec568d040f88ddc04a199fa6da90923bbb41d4dcaed02caa924d6ef57`; this is the case 3 GGUF, not the case 2 BF16 checkpoint or the 7B VL model.

Decoded metadata establishes 30 Llama blocks, width 576, feed-forward width 1,536, nine query heads, three KV heads, 64 dimensions per head, RoPE base 100,000, RMSNorm epsilon approximately 1e-5 and vocabulary 49,152. The existing inventory records F16 matrix weights, F32 norms and no separate output matrix; later implementation must resolve tied embeddings and GGUF RoPE weight layout against the runtime, rather than assume Hugging Face layout.

The fixed plain text `The camera moves slowly.` tokenizes without BOS or special handling to [504, 6715, 7488, 6351, 30]. No dataset or chat template is involved. Each saved row is the raw next-token logit vector after consuming that position (positions 0-4), before sampling or softmax. Sequential one-token evaluation uses CPU-only execution, float32 K/V cache, context 64, four threads, no operation/KQV offload and no flash attention. Float32 cache deliberately simplifies later comparison and differs from earlier text smoke settings.

| Measurement | Observed |
| --- | ---: |
| Logit array | 5 x 49,152 float32, all finite |
| Constructor / evaluation seconds | 0.07526 / 0.07423 |
| Total capture/check seconds | 0.46166 |
| Peak process RSS bytes | 422,625,280 |
| Before / minimum / after free | 70% / 70% / 70% |
| Pressure across eight readings | Normal throughout |

The compressed logits have SHA256 `627ac8a032c8b98645fb4b904d9ba5cb5d73f11efe6f8ee8d411732eb1248040`. The model explicitly closed and watcher joined before exit. RSS excludes some unified-memory allocations and snapshots can miss transients.

Before implementing the custom forward pass, acceptance is fixed as every logit satisfying `abs(custom-reference) <= 0.001 + 0.001*abs(reference)`. Also report maximum/RMS absolute error and per-position argmax agreement. This is a proposed numerical criterion, not a measured pass. Do not relax it silently if the implementation fails. No custom transformer, parity comparison, or training has run; case 6 is incomplete. Next implement the full tiny decoded architecture with the same weights, then compare to this saved reference with the reference runtime closed.

The pinned [llama-cpp-python evaluation implementation](https://github.com/abetlen/llama-cpp-python/blob/v0.3.36/llama_cpp/llama.py) was checked before capture: `logits_all=True` retains each evaluated position's scores. The installed implementation was also inspected directly. The pinned llama.cpp model-source fetch failed in this iteration; resolve that computation source before implementing unfamiliar RoPE/weight handling.

## Sources consulted

The official Qwen configuration, [model card](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct), Qwen2-VL runtime implementation, and [GGUF specification](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md) were consulted on 2026-11-11. Proposed milestones and memory estimates are project reasoning rather than completed results.

### 2026-11-11: independent full tiny-model forward comparison

`src/case6_forward.py` implements all 30 decoded blocks directly in NumPy 2.5.4, using the unchanged F16 GGUF matrices converted exactly to float32 and its F32 norms. It implements RMSNorm, nine-query/three-KV grouped attention, adjacent-pair RoPE with base 100,000, causal softmax, residual connections, SiLU-gated feed-forward layers and tied embedding output projection. GGUFReader arrays already reverse GGML dimension order; matrix multiplication uses their transpose. This is the full tiny source architecture, not a shape-reduced substitute or the 7B VL model. The code consumes the five saved token IDs, without reimplementing tokenization. Full-sequence attention is compared with the saved sequential reference; no reference model was loaded concurrently.

Before implementation, the pinned [Llama graph](https://github.com/ggml-org/llama.cpp/blob/0c1e57098bba43ac29e6e3b677cdceebdd22334f/src/models/llama.cpp) and [CPU rotary operations](https://github.com/ggml-org/llama.cpp/blob/0c1e57098bba43ac29e6e3b677cdceebdd22334f/ggml/src/ggml-cpu/ops.cpp) were fetched directly after browser fetch failure. Exact source copies are preserved as `results/case6-llama-source.cpp` and `results/case6-ops-source.cpp`. They establish tied output embeddings, pre-norm residual ordering, gated SiLU and adjacent-pair normal RoPE.

| Measurement | Observed |
| --- | ---: |
| Compared logits | 245,760 (5 x 49,152) |
| Maximum / RMS absolute error | 0.14151001 / 0.02623361 |
| Outside predeclared tolerance | 170,368 |
| Per-position argmax agreement | 5/5 |
| Custom forward / total seconds | 0.30590 / 1.70015 |
| Peak process RSS bytes | 603,013,120 |

**The declared numerical criterion failed.** Matching argmax IDs does not establish parity. The criterion remains unchanged; the underlying cause is unresolved. The custom path performs float32 matrix arithmetic on decoded weights, while reference kernel precision and accumulation behavior have not yet been matched. Position zero also differs (maximum error 0.14151), so positional rotation alone cannot explain all error. Investigate runtime matrix arithmetic and intermediate activations before claiming parity; do not attribute this to quantization without measurements. No repair or tolerance adjustment was attempted after observing the result in this iteration.

Raw custom logits, their hash, reference metadata hash, per-position errors, timing and memory evidence are in `results/case6-forward*`. All 37 readings showed normal pressure and at least 70% reported free memory. Arrays/readers were released and the watcher joined; RSS omits some unified-memory use and sampling can miss transients. The original source and reference evidence remain unchanged. Training, generation and tokenizer reconstruction remain pending. Case 6 is a measured partial attempt, not completed parity or training.
