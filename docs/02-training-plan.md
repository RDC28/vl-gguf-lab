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
