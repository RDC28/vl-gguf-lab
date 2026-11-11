# Case 2: fine-tune and re-test

## Goal

Find out whether supervised adaptation improves scene fidelity and the reference shot-breakdown style over case 1. The overnight task authorizes a tiny local LoRA workflow demonstration. No training has run yet; it will use its own model baseline rather than claim adaptation of the 7B VL model.

## Method

Use the original floating-point Qwen checkpoint with a supported adapter training stack. Train on scene descriptions paired with target breakdowns, masking prompt tokens out of the supervised loss. Keep the vision component frozen for this text-only question. LoRA learns a small set of low-rank weight updates; QLoRA additionally uses a quantized training base. Neither makes all activation and operating-system memory disappear. GGUF is the inference artifact, not the normal starting format for this training path. [PEFT LoRA documentation](https://huggingface.co/docs/peft/v0.21.0/package_reference/lora)

## Tools and versions

| Tool | Version status | Role |
| --- | --- | --- |
| Existing baseline stack | Python 3.12.13, llama-cpp-python 0.3.36 | Controlled before/after inference |
| PEFT | 0.21.0 documentation consulted; not installed | Candidate LoRA adapter training |
| Transformers, PyTorch, bitsandbytes | Versions not selected or installed | Candidate cloud training stack; validate their compatibility together |
| MLX-VLM / MLX | Versions not selected or installed | Candidate Apple Silicon adaptation route |
| llama.cpp | Baseline commit recorded in case 1 | Convert/export compatible artifacts after training |

MLX-VLM advertises VL inference and fine-tuning on Apple Silicon, but that does not establish that this model and training setup fit 8 GB. Confirm architecture, text-only loss, and export support before selecting it. [MLX-VLM project](https://github.com/Neywa-Labs/mlx-vlm)

## Feasibility on the 8 GB Air

Full 7B fine-tuning is not feasible here: 15.23 GB of BF16 language weights alone (from the baseline tensor inventory) exceeds physical RAM, before gradients, optimizer states, activations, and the vision stack. A 7B adapter run is not assumed to fit safely while other work is running. Suitable cloud hardware is the practical 7B route; an explicitly smaller model could support a separate local feasibility study. These are resource estimates, not measured training results.

## Concrete plan

1. Check semantic duplicates, licensing, and target consistency. Freeze disjoint train/validation/test scene groups before training. The current 10-row exploratory sample must not silently become a trained model's unbiased test set.
2. Agree a human rubric: scene facts retained, invented events, continuity, four-shot structure, and useful camera instructions. Score the untrained model first.
3. Select and pin a compatible training stack on the chosen hardware. Measure a one-batch memory budget before a training run. Preserve the base revision and adapter configuration.
4. Tune only on the training/validation sets. Save adapter, seed, learning rate, loss curves, trainable parameter count, peak memory, and wall time.
5. Convert a supported merged checkpoint to the same inference quant. Re-run the held-out inputs with identical prompt/context/sampling. Compare adapters before merge where possible to separate training effects from conversion effects.

## Risks and open questions

Only 150 pairs invites overfitting and style memorization. Random row splits can leak similar scenes. Whether frozen vision weights retain image quality needs a separate image benchmark. Adapter export and quantization can change results. Cloud budget, data permission, best smaller local model, and the exact versions remain unresolved before any experiment.

## Findings

### 2026-11-11: frozen split before training

`src/freeze_split.py` reads the verified dataset and writes `results/case2-split.json` exclusively, refusing to overwrite it. The frozen split is 120 training, 15 validation, and 15 test rows. The SHA256 of the canonical index mapping is `3d4a1eff32d197aaab0c49a92171c31cdf0c7f557d1191afcce3752cf8f3ece0`; the artifact defines the exact hash encoding and includes dataset SHA256, indexes, group assignments, and selection seed 42. No framework or model was loaded.

All 150 input descriptions were reviewed for shared scene beats. Conservative manual groups keep related ferry, bridge, school, costume, radio, cosmic-separation, wedding, and other scenes together. Connected groups, not individual rows, are assigned. An additional all-pairs normalized word-set Jaccard screen at 0.45 found zero qualifying pairs; normalized exact duplicates were also zero. Lexical uniqueness alone would therefore miss the manually identified relationships. References are never inspected by the grouping implementation. Shared characters and story worlds still cross splits; this is a within-domain adaptation check, not an independence guarantee.

The ten scenes whose baseline outputs have already been inspected, and their connected groups, are forced into training. Test and validation each contain 15 other rows; no group crosses partitions. The explicit test indexes are [23, 34, 48, 55, 60, 81, 106, 107, 121, 123, 124, 125, 132, 141, 142]. Do not select hyperparameters or training examples based on these test results. Validation is for tuning only. Freeze this artifact before the model's own untrained held-out baseline and retain it unchanged through training/re-test.

Candidate for the next bounded experiment: `HuggingFaceTB/SmolLM2-135M-Instruct`, a tiny text-only Llama-family model with an Apache-2.0 model card, using MLX-LM LoRA. The [official model card](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct) and [MLX-LM LoRA documentation](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/LORA.md) were checked before selection. This candidate is deliberately much smaller than 3B to leave training headroom; compatibility and memory remain unmeasured. No download or dependency installation occurred this iteration. Next: pin a revision and training versions, download once with hashes, then perform a guarded one-batch preflight and its own before/after held-out evaluation. Training, loss, trainable parameters, and memory/time findings remain pending.

Validation: reproducible split reconstruction, exact coverage/disjointness, group containment, exploratory-row exclusion, hash consistency, exclusive-create refusal, Python/notebook syntax/schema, dependency consistency, and diff whitespace checks passed.

## Sources consulted

The PEFT 0.21.0 LoRA reference and MLX-VLM project linked above were consulted on 2026-11-11. The original [Qwen2.5-VL model card](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) supplies the base model identity. Compatibility and feasibility must be measured when versions are selected.

### 2026-11-11 17:56 UTC: tiny model acquired and inspected

Downloaded `HuggingFaceTB/SmolLM2-135M-Instruct` once at revision `12fd25f77366fa6b3b4b768ec3050bf629380bac` into `models/SmolLM2-135M-Instruct`, using direct HTTPS file reads without a shared model cache. The nine selected files total 272,444,345 bytes: weights, configuration, tokenizer assets, and the upstream model card. No upstream training-state or evaluation artifacts were downloaded. Original 7B files and their manifest remain unchanged, so baseline resume provenance stays valid. `results/case2-model-provenance.json` records every file size/SHA256 and the complete safetensors header.

| Observed item | Value |
| --- | --- |
| Weight file | `model.safetensors`, 269,060,552 bytes |
| Weight SHA256, also matched against upstream LFS | `5af571cbf074e6d21a03528d2330792e532ca608f24ac70a143f6b369968ab8c` |
| Stored tensors / elements / dtype | 272 / 134,515,008 / BF16 |
| Architecture | Llama causal LM, 30 blocks, hidden 576, feed-forward 1,536 |
| Attention / KV heads | 9 / 3, head dimension 64 |
| Vocabulary / tied embeddings | 49,152 / true |
| RMSNorm epsilon / RoPE theta / declared context | 0.00001 / 100,000 / 8,192 |
| Download plus hash/header inspection | 17.611 seconds |
| Acquisition process peak RSS | 57,802,752 bytes |

Stored BF16 weights occupy 269,030,016 tensor bytes; an FP32 copy alone would occupy 538,060,032 bytes. These estimates exclude adapters, gradients, optimizer states, activations, caches, and framework allocations. They justify trying a guarded tiny-model preflight, not a claim that training fits. Acquisition RSS measures only the downloader/header reader, not inference or training, and does not cover all unified-memory allocations. No model tensors were instantiated and no held-out outputs or references were inspected this iteration.

The pinned [upstream configuration](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct/blob/12fd25f77366fa6b3b4b768ec3050bf629380bac/config.json) and [model card](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct/blob/12fd25f77366fa6b3b4b768ec3050bf629380bac/README.md) identify a tiny text-only model, not a VL replacement. MLX-LM's [LoRA instructions](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/LORA.md) list Llama support, but compatibility of these exact files remains untested. Next: select/pin local MLX-LM dependencies, load this local directory without another download, measure a short preflight with the existing memory policy, and collect its own untrained baseline before adaptation. Training, loss, adapters, and quality changes remain pending.

Validation: all nine local file hashes and sizes rechecked, weight hash matched the upstream LFS digest, tensor byte spans/shapes and architecture inventory checked, frozen split reproduced unchanged, existing evidence verified against Git, notebook/Python syntax and dependency consistency checked. No inference, training, or server process was started.

### 2026-11-11: guarded MLX LoRA preflight

Installed MLX-LM 0.32.2 with its training extra into the existing `.venv`; MLX/MLX-Metal 0.32.4, Transformers 5.22.2 and all added dependencies are pinned in `requirements.txt`. No existing package version changed. The [official MLX-LM project](https://github.com/ml-explore/mlx-lm) and [LoRA guide](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/LORA.md) were consulted before implementation; the installed loader and adapter code were inspected for exact API behavior. The unchanged local BF16 checkpoint loads directly without conversion, remote code, or network downloads.

`src/case2_preflight.py` verifies all nine source file hashes and the frozen dataset hash, then uses only the first frozen training index (0). It freezes the base and adds rank-4, scale-8, zero-dropout adapters to query/value projections in the last two blocks: 15,360 trainable elements across eight arrays. It computes one batch-one forward/backward loss, with no optimizer update or adapter save. Seed is 42. This establishes gradient feasibility, not completed training or quality improvement.

| Measurement | Final preflight |
| --- | ---: |
| Prompt / total tokens | 96 / 129 |
| Model input / supervised answer targets | 128 / 33 |
| Model load | 0.144 seconds |
| Forward/backward | 0.062 seconds |
| Total process work | 1.316 seconds |
| Partial-answer cross entropy | 4.21875 |
| Gradient L2 / finite | 0.335437 / all finite |
| MLX peak allocated memory | 382,325,130 bytes |
| Peak process RSS | 426,573,824 bytes |
| Free before / minimum sampled / after release | 68% / 62% / 63% |
| Highest sampled pressure | 1 (normal) |

`results/case2-preflight.json` and its runtime log preserve the final run. Sampling occurs every 0.5 seconds plus stage boundaries; the existing normal-pressure/15% guard applies before loading and computation, and critical pressure or below 8% free requests a stop. A single short MLX graph cannot be interrupted between its internal kernels; the flag is checked after evaluation. Neither RSS nor MLX allocation alone measures all unified-memory use, and subsecond peaks can escape system sampling. Model references/cache were released and each process exited.

Two earlier attempts are preserved. `case2-preflight-64*` loaded safely but refused to compute because its 64-token cap contained no answer after the 96-token prompt. `case2-preflight-128-offset*` computed finite gradients in 2.017 seconds but omitted the first answer target due to a shifted masking boundary (32 supervised targets, loss 3.96875). The final mask includes all 33 answer targets and excludes all prompt targets. Timing differences include uncontrolled first-use kernel/cache overhead and different masks; they are not evidence of a speed improvement. Only the first 33 answer tokens are tested, not a whole shot breakdown. No test or validation references entered this preflight.

Validation: real local forward/backward execution, finite/nonzero gradients, source integrity, frozen split reproduction, Python compilation, notebook schema/syntax, dependency compatibility (131 packages), and whitespace checks passed. Held-out baseline, optimizer-state memory, actual LoRA training, adapters, and identical-settings re-test remain pending. Next: collect this tiny model's own untrained held-out outputs, then a short guarded training run; do not infer that longer sequences or the 7B model fit from this result.
