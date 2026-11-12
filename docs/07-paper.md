# Scene-to-shot generation: a reproducible Colab study

## Status and question

This is a methods-first paper draft. **Full Colab results are pending.** No table below contains projected outcomes. The historical 8 GB Mac work was an exploratory feasibility study: partial VL inference, tiny smoke training, single-prompt tensor edits and a failed strict numerical parity test. Its unchanged artifacts are retained under each case's `local-2026-11-11` folder and must not be presented as the new Colab experiment.

The study asks whether a vision-language decoder can produce structured cinematic shot breakdowns from text, how supervised training changes a smaller model's held-out behavior, how controlled GGUF block edits affect that behavior, and whether an independently decoded transformer can be trained on the same task. Since the dataset contains no images, the study makes no claim about visual understanding.

## Data and protocol

The committed private dataset contains 150 unique input/reference strings. Every reference contains four numbered shots, descriptive shot fields and continuity guidance. Provenance and the SHA256 are recorded in `data/README.md`. `src/freeze_split.py` freezes input-only story groups, lexical grouping and a seeded 120/15/15 train/validation/test split. Previously explored inputs are assigned to training. Related characters, templates and story worlds can still cross splits; the resulting held-out test is small and not a broad filmmaking benchmark.

Case 1 measures descriptive performance over all 150 inputs. Cases 2-6 evaluate the same 15 test scenes. Training uses only the 120 training scenes; validation remains reserved because the default budget is fixed before evaluation. No checkpoint, block position or scale is selected using test scores. Report all sweep variants, including damaging and null outcomes.

## Execution and reproducibility

There is exactly one notebook for each case. The notebooks call shared Python modules for model loading, scoring, editing, training, storage and summaries. Setup clones the private repository, installs pinned requirements and the official CUDA llama.cpp wheel, then verifies GPU support. Model downloads use immutable Hugging Face revisions. The default VL tier is Qwen2.5-VL-7B Q4_K_M for T4, with Q8_0 for L4 and BF16 for A100 as higher-precision options. GPU offload is required for full notebook runs. Actual VRAM fit and runtime must be reported from Colab, not inferred from filenames.

Google Drive receives a per-case timestamped run folder containing config, code revision, package versions, dataset/split hashes, output JSONL, metrics, edit maps and hashes, event/runtime logs, charts and training checkpoints. Scene rows and step logs save incrementally. Checkpoints include weights, optimizer and RNG state every 10 steps and at completion. Resume rejects changed config and provenance, skips completed scenes and replays training after the last durable checkpoint. Local fallback follows the same contract without Google dependencies. Remote Drive interruption can still affect durability; the workflow limits lost work rather than guaranteeing storage semantics.

## Six experiments

1. **VL baseline:** greedy decoding of all 150 scene inputs with 8,192 context tokens and a 4,096-token completion cap. The vision projector stays unloaded. Report completed scenes, finish reasons, truncation, elapsed time and quality proxies.
2. **Train and re-test:** pinned SmolLM2-1.7B with rank-16 query/value LoRA, before/after on the frozen test set. Three shuffled training epochs use response-only next-token cross entropy, 2,048-token training sequences, AdamW at 0.0001 and gradient clipping. Report supervised tokens and clipped training targets alongside losses.
3. **Modify layers:** pinned F16 SmolLM2-135M control. Multiply attention-norm weights independently by 0.9, 1.01 or 1.1 at layer 0, 15 or 29. Score all nine variants against the unchanged source.
4. **Add layers:** insert an exact copy immediately after original block 0, 15 or 29, independently. Update the block count and renumber tensors. Duplicating a residual block is not identity initialization; no quality gain is assumed.
5. **Remove layers:** independently delete original block 0, 15 or 29 and renumber retained blocks. Report quality and observed latency without extrapolating speed gains from a single timing.
6. **Own transformer:** decode F16 GGUF weights into an independent trainable PyTorch Llama-family decoder. Measure strict per-logit parity, evaluate before training, train all decoder parameters on the training split, and re-evaluate at identical settings. If parity fails, training results are explicitly exploratory.

Cases 3-5 begin with the unedited baseline and a no-change rewrite. Greedy control output must match before the sweep proceeds. Tensor hashes and block maps verify each rewritten artifact. These cases and case 6 use a small text-model control; results cannot be attributed to architectural edits on the 7B VL model. Model scale, training method and quantization differ across cases, so only the within-case paired comparisons support causal claims about those interventions.

## Metrics and uncertainty

Save raw generated text and per-scene Unicode word unigram/bigram F1. Parse numbered SHOT headings and time/lens/framing/movement fields. Shot-count agreement penalizes missing or extra shots relative to the reference. Capped field-count recall measures how many reference fields the parser finds in a prediction. Structure score averages count agreement and field recall. These proxies reward surface conformity and may favor boilerplate; they do not judge emotional intent, continuity correctness, dramatic staging or cinematic quality. Human inspection of raw answers is required before making those claims.

Report truncation separately, with `length` termination counted as capped, and retain token usage. A long cap reduces avoidable cutoffs but does not guarantee complete answers. Save paired per-scene deltas and descriptive bootstrap intervals for word F1 and structure score. The intervals resample scenes, not independent story worlds, and should not be interpreted as strong population-level significance on a dataset of this size.

For case 6, retain the original criterion: every logit must satisfy `abs(custom-reference) <= 0.001 + 0.001*abs(reference)`. Save maximum error, RMS error, violations and argmax agreement separately. Historical argmax agreement did not resolve the failed tolerance. The new implementation measures the gap again; operand precision and reduction-order differences are plausible contributors, not an established diagnosis. Local NumPy/PyTorch, cache and gradient tests do not substitute for parity on the actual model.

## Results to fill from full Colab artifacts

| Case | Coverage / training | Word overlap | Shot structure | Truncation | Main comparison |
| --- | --- | --- | --- | --- | --- |
| 1 VL baseline | PENDING | PENDING | PENDING | PENDING | All 150 scenes |
| 2 train/re-test | PENDING | PENDING | PENDING | PENDING | Held-out before/after |
| 3 norm edits | PENDING | PENDING | PENDING | PENDING | Each variant vs unchanged |
| 4 block insertion | PENDING | PENDING | PENDING | PENDING | Each variant vs unchanged |
| 5 block removal | PENDING | PENDING | PENDING | PENDING | Each variant vs unchanged |
| 6 own decoder | PENDING | PENDING | PENDING | PENDING | Strict parity and held-out before/after |

GPU model, tier, memory fit, elapsed time, training loss curves, training clipping, checkpoint/reconnect evidence, strict parity statistics and artifact run IDs: **PENDING COLAB RUN**. Populate these fields only from completed `summary.json`, per-scene rows, training summaries and parity artifacts. Preserve failed/incomplete runs and disclose their limits.

## Validation boundary

Local validation checks notebook JSON/schema and executes the CPU-safe path through real small-model download, generation, rewriting, training, charts and local persistence. Unit tests exercise partial-tail recovery, resumed evaluation, group separation, tensor mapping, independent decoder math, KV cache, gradients and exact optimizer-state resumption. Smoke outputs are setup evidence only and cannot populate the results table.

Only a real Colab session can establish CUDA wheel compatibility with its current image, GPU offload and VRAM fit, Drive authorization/mount/reconnect behavior and full-run feasibility and outcomes. No Colab execution is claimed by this repository preparation. The scientifically defensible result may include null or harmful interventions and a persistent parity failure; reporting those honestly is part of the experiment.
