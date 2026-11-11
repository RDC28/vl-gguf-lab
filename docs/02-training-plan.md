# Case 2: fine-tune and re-test

## Goal

Find out whether supervised adaptation improves scene fidelity and the reference shot-breakdown style over case 1. This is a research plan; no training is authorized in this setup task.

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

Full 7B fine-tuning is not feasible here: roughly 14 GB of BF16 language weights alone exceeds physical RAM, before gradients, optimizer states, activations, and the vision stack. A 7B adapter run is not assumed to fit safely while other work is running. Suitable cloud hardware is the practical 7B route; an explicitly smaller model could support a separate local feasibility study. These are resource estimates, not measured training results.

## Concrete plan

1. Check semantic duplicates, licensing, and target consistency. Freeze disjoint train/validation/test scene groups before training. The current 10-row exploratory sample must not silently become a trained model's unbiased test set.
2. Agree a human rubric: scene facts retained, invented events, continuity, four-shot structure, and useful camera instructions. Score the untrained model first.
3. Select and pin a compatible training stack on the chosen hardware. Measure a one-batch memory budget before a training run. Preserve the base revision and adapter configuration.
4. Tune only on the training/validation sets. Save adapter, seed, learning rate, loss curves, trainable parameter count, peak memory, and wall time.
5. Convert a supported merged checkpoint to the same inference quant. Re-run the held-out inputs with identical prompt/context/sampling. Compare adapters before merge where possible to separate training effects from conversion effects.

## Risks and open questions

Only 150 pairs invites overfitting and style memorization. Random row splits can leak similar scenes. Whether frozen vision weights retain image quality needs a separate image benchmark. Adapter export and quantization can change results. Cloud budget, data permission, best smaller local model, and the exact versions remain unresolved before any experiment.

## Findings

Not run. Reserved: split hashes, adapter settings, memory/time measurements, training curves, held-out table, human review, and conversion checks. No trained model or training claims exist yet.

## Sources consulted

The PEFT 0.21.0 LoRA reference and MLX-VLM project linked above were consulted on 2026-11-11. The original [Qwen2.5-VL model card](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) supplies the base model identity. Compatibility and feasibility must be measured when versions are selected.
