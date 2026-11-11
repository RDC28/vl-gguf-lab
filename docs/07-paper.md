# Local VL model experiments for scene-to-shot generation

Draft paper skeleton, 2026-11-11. Observed setup facts are distinguished from planned experiments. This is not a completed research paper.

## Abstract

We investigate whether a compact, locally executed vision-language model can generate useful shot breakdowns from short text scene descriptions on an 8 GB Apple Silicon MacBook Air. The supplied dataset contains 150 text input/output pairs. The initial candidate is an Unsloth Qwen2.5-VL-7B-Instruct Q3_K_M GGUF language file under 4 GB. The planned sequence covers untrained inference, adapter fine-tuning, weight modification, block addition, block removal, and reconstruction/training with our own implementation. The current implementation covers baseline inference and resumable evidence collection. The first GPU output completed in 143.8 seconds at 6.46 evaluation tokens/s; all ten fixed CPU sample outputs have completed, six capped, with transient warning pressure in recent attempts; later stages remain plans. The text-only dataset cannot support conclusions about vision performance.

## Research questions

| Case | Question | Evidence required |
| --- | --- | --- |
| 1 | Does the untrained model produce usable shot breakdowns within local resource limits? | Raw outputs, timing, memory, structure/overlap metrics, human review |
| 2 | Does supervised adaptation improve held-out quality? | Frozen splits, training settings, before/after results |
| 3 | What changes when existing weights are perturbed? | No-change rewrite control, exact edit manifest, output comparison |
| 4 | Can added blocks preserve initial behavior and learn useful capacity? | Architecture consistency, initial parity, trained held-out comparison |
| 5 | Can blocks be removed with a useful efficiency/quality trade-off? | Removal rule/index map, size/speed/memory, held-out damage |
| 6 | Can we reconstruct computation and train with our own code? | Tensor inventory, operation/logit parity, documented training experiment |

## Setup

Hardware: Apple M1 MacBook Air, 8 cores, 8 GiB unified memory; other applications remain active. Dataset: 454,187-byte JSON list with 150 unique input strings and 150 unique references. Inputs average 17.87 word tokens; references average 434.57, with four numbered SHOT headings each. Exact-string uniqueness does not prove scene independence. Dataset SHA256 is recorded in `../results/dataset-summary.json`.

Model: `unsloth/Qwen2.5-VL-7B-Instruct-GGUF`, revision `68bb8bc4b7df5289c143aaec0ab477a7d4051aab`; language file `Qwen2.5-VL-7B-Instruct-Q3_K_M.gguf` (3,808,390,016 bytes). The separately downloaded F16 vision projector stays unloaded for text-only inference. See [case 1](01-baseline.md) for all artifact hashes, quant alternatives, and runtime provenance. The 4 GB cap refers to the language file, not total disk usage or RAM.

Python 3.12.13, llama-cpp-python 0.3.36, gguf 0.19.0, JupyterLab 4.6.6, and ipykernel 7.4.0 are installed. Metal support and project-kernel startup have been checked. Full package pins live in the root requirements file. Future training tools remain unselected and uninstalled.

## Methods

### Case 1: baseline inference

Use a seed-42 random sample of 10 rows, a fixed storyboard prompt, greedy decoding, 2,048 context tokens, and a 1,024-token generation cap. Run one process with the language model only. Save every raw output, timing, finish reason, and memory measurements. Measure word counts, numbered headings, and multiset unigram F1 against references. Flag truncation and inspect scene fidelity manually. Estimate full-set duration from observed sample wall time. [Protocol](01-baseline.md)

### Case 2: fine-tune and re-test

Freeze scene-group splits, select compatible adapter training on adequate hardware, then compare with the same held-out protocol and quantization. Record training and export effects separately. The split is now frozen for a tiny local text-model demonstration; no training has run. [Plan](02-training-plan.md)

### Case 3: modify existing layers

Preserve original artifacts; validate a no-change rewrite before perturbing one existing tensor. Record exact edits, hashes, and unchanged-tensor evidence. Avoid full-model dequantization locally. [Plan](03-modify-layers.md)

### Case 4: add layers

Establish architecture/parity checks, insert one compatible initialized block, train on adequate hardware, then export and compare. Treat successful loading and improved quality as separate outcomes. [Plan](04-add-layers.md)

### Case 5: remove layers

Predeclare a removal rule or calibration method, remove/renumber blocks consistently, and measure quality loss with resource savings. Keep calibration separate from evaluation. [Plan](05-remove-layers.md)

### Case 6: reconstruct and train

Decode tensor/configuration facts, implement operations, compare intermediate values and logits against a quantized reference, then investigate our own tiny-model training loop. Full 7B training is outside the local resource budget. [Plan](06-transformer-plan.md)

## Results

| Case | Current status | Result location |
| --- | --- | --- |
| 1 | One GPU output completed; resumed GPU probe stopped safely before generation; CPU-only fixed sample has 10 of 10 scenes completed, six capped; latest attempt had transient warning pressure | results/baseline.jsonl; case 1 findings |
| 2 | Split frozen (120/15/15); pinned 135M text model downloaded and inspected; training and model baseline pending | results/case2-split.json; case 2 findings |
| 3 | Not run | Future findings in case 3 |
| 4 | Not run | Future findings in case 4 |
| 5 | Not run | Future findings in case 5 |
| 6 | Not run | Future findings in case 6 |

Initial observed GPU result: load 4.882 seconds; one completed input in 143.792 seconds; 6.4555 generation evaluation tokens/s; peak process RSS 3,546,480,640 bytes; minimum sampled free percentage 16%, normal pressure throughout. The output has 621 words and six SHOT headings against 448 words and four headings in the reference; unigram F1 is 0.2320. It stopped naturally rather than reaching the output cap. A one-input extrapolation is roughly 6 hours for all 150; that duration has not been measured. The next input was deliberately interrupted for overnight handoff. See case 1 for raw evidence, corrected heading-count provenance, failed loads, and limits.

A bounded resume probe at 17:13 UTC reused row 6 and selected missing index 22. The verified original 7B file loaded on Metal in 4.283 seconds including backend initialization, starting at normal pressure and 70% reported free. Before generation, warning pressure level 2 with 15% free triggered the unchanged guard. No additional row completed; peak process RSS was 3,386,982,400 bytes, which does not include all Metal allocations. The model closed. Timestamped attempt metadata and the runtime log preserve the failure without overwriting the earlier evidence. A separate CPU-only probe of the same file subsequently completed; its measurements follow. Resume validation now protects dataset/model/settings provenance, skips completed indexes, and syncs each row to disk. See case 1 for exact artifacts and checks.

The CPU-only configuration probe completed row 6 in 114.320 seconds after a 4.256-second load, evaluating 1,023 generation tokens at 9.534 tokens/s. Pressure stayed normal across 60 samples; reported free memory went from 68% before loading to a minimum 42%, then 68% after closing. Peak process RSS was 4,330,668,032 bytes. Its 704-word output had eight headings versus four reference headings and unigram F1 0.22396. It repeated shots and ended mid-description at the token cap. This is a configuration comparison on the existing scene, not a second unique scene or evidence of quality improvement. CPU/GPU outputs differ; uncontrolled cache/thermal state and differing output lengths limit speed comparisons. A single-row CPU extrapolation is about 19 minutes for 10 inputs or 4.76 hours for 150, not a measured full run. Raw evidence is separate under `results/baseline-cpu-probe*`; the original GPU evidence is preserved. CPU sample continuation is next, without a fallback-model download. A follow-up validation accidentally selected row 28 by changing sample size to one; it was deliberately interrupted with no completed output, and its attempt metadata is retained.

A two-row CPU continuation appended indexes 22 and 26 without rerunning index 6. They took 125.468 and 134.755 seconds at 8.638 and 7.817 generation evaluation tokens/s, respectively; both hit the output cap with 1,023 evaluated generation tokens. Outputs had 661/673 words and ten headings each versus 440/437 words and four headings in their references; unigram F1 was 0.20890/0.19279. The 265.761-second attempt loaded in 5.669 seconds, reached peak process RSS 4,089,675,776 bytes, and stayed at normal pressure across 135 samples with minimum 42% reported free (69% before, 67% after close). RSS does not capture all Metal allocations. The model closed normally. Row 22 assumes missing photo continuity details and invents shopping actions; row 26 repeats photographing/reaction shots. Across three CPU rows, mean input time is 124.848 seconds and mean unigram F1 0.20855; all outputs are capped. Estimated full-set time is now 5.20 hours, excluding load, rather than a measured completion time. Seven fixed-sample rows remain. Raw evidence and cumulative metrics are linked in case 1 and `results/baseline-cpu-sample-summary.json`; cases 2-6 remain unattempted.

A second bounded CPU continuation completed indexes 28 and 35 in 83.654/118.406 seconds, with 692/680 evaluated generation tokens at 9.001/5.923 tokens/s. Both stopped naturally. Outputs had 466/458 words and six headings each versus 428/450 words and four reference headings; unigram F1 was 0.24161/0.25330. Load took 9.127 seconds and total attempt time was 211.060 seconds; peak process RSS was 3,969,187,840 bytes, excluding some Metal allocations. Of 108 memory samples, ten showed warning pressure during generation; neither input started under warning pressure, no critical pressure occurred, and reported free memory stayed at least 36% (68% before load, 62% after close). The unchanged guard permitted completion and the model closed normally. Index 28 omits an explicit action cue and invents a bow/exit; index 35 omits 15-second timing and invents power restoration while mostly repeating one framing. Across five CPU scenes, mean input time is 115.321 seconds, mean unigram F1 is 0.22411, and three outputs are capped. The revised full-set estimate is 4.81 hours excluding load, not a measured completion time. Five fixed-sample scenes remain. Raw outputs, per-attempt memory evidence, and summary are recorded in case 1. Cases 2-6 remain unattempted.

A third bounded CPU continuation completed indexes 57 and 62 in 132.444/81.823 seconds, evaluating 1,023/563 generation tokens at 8.177/7.186 tokens/s. Index 57 was capped; index 62 stopped naturally. Outputs had 769/383 words and six/five headings against 426/423 words and four reference headings; unigram F1 was 0.20418/0.23573. Load took 7.759 seconds and total attempt time was 221.885 seconds. Peak process RSS was 3,957,096,448 bytes, excluding some Metal allocations. Minimum free memory was 40% across 113 samples; three samples showed warning pressure during generation, while both inputs started under normal pressure. No critical pressure occurred and the model closed normally. Index 57 preserves the four named cats but cannot establish visual identity consistency; index 62 omits Ulluman's name and repeats movement despite maintaining a grounded atmosphere. Seven CPU scenes are complete, four capped; mean wall time is 112.981 seconds, mean unigram F1 is 0.22293, and the full-set extrapolation is 4.71 hours excluding load. Three sample scenes remain; cases 2-6 remain unattempted. Case 1 links raw outputs, memory evidence, and validation.

A final three-row continuation completed indexes 70, 108, and 139 in 126.178/147.191/159.017 seconds at 7.727/7.168/6.625 generation tokens/s. The latter two were capped. Load took 8.047 seconds, total attempt time was 440.320 seconds, and peak process RSS was 3,991,896,064 bytes, excluding some Metal allocations. Across 223 samples, minimum reported free memory was 39%, with three warning-pressure samples and no critical readings. All inputs started at normal pressure; after closing, pressure was normal with 63% free. Row 70 invents investigative equipment and a discovery; row 108 repeats lake/light views; row 139 proposes showing fire consuming Sati despite the non-graphic request. All ten fixed CPU sample scenes are now complete, six capped, with mean wall time 122.326 seconds and mean unigram F1 0.21726. Completed-input wall time totals 20.39 minutes excluding loading and incomplete attempts. The updated full-set extrapolation is 5.10 hours, not a measured duration. Case 1 and the cumulative summary preserve the raw evidence and validation. Cases 2-6 remain unattempted; the next step is a small case 2 experiment after checkpointing the sample.

Case 2 preparation froze 120/15/15 train/validation/test rows before training. Input-only manual scene groups keep related beats in one partition; the ten already inspected baseline rows and their groups are training-only. An all-pairs word-set Jaccard screen at 0.45 found no pairs, despite manually identified related scenes, illustrating the limits of lexical deduplication. Exact normalized duplicates were zero. The canonical index hash is `3d4a1eff32d197aaab0c49a92171c31cdf0c7f557d1191afcce3752cf8f3ece0`. Shared story worlds remain across partitions. SmolLM2-135M-Instruct with MLX-LM is a candidate for the next bounded experiment, not an installed or measured training system. No training or improvement is claimed. See case 2 and `results/case2-split.json` for selection rules and frozen indexes.

Case 2 acquired the tiny text-only `HuggingFaceTB/SmolLM2-135M-Instruct` at revision `12fd25f77366fa6b3b4b768ec3050bf629380bac`. Nine local files total 272,444,345 bytes; the 269,060,552-byte safetensors file matches upstream SHA256 `5af571cbf074e6d21a03528d2330792e532ca608f24ac70a143f6b369968ab8c`. Read-only inspection found 272 BF16 tensors containing 134,515,008 stored elements, 30 Llama blocks, hidden width 576, and tied embeddings. BF16 tensor bytes total 269,030,016; a full FP32 weight copy would require 538,060,032 bytes before any training overhead. Acquisition and verification took 17.611 seconds with 57,802,752-byte process peak RSS. These are download/header measurements, not execution or training memory evidence. The frozen split, original 7B artifacts, and baseline provenance remain unchanged. Training framework installation, guarded preflight, and before/after evaluation are still pending. Case 2 records primary sources and `results/case2-model-provenance.json` preserves file hashes and the complete tensor header.

## Discussion

The resource limit is unified memory shared with the operating system and other applications. A sub-4-GB file is not proof of a safe runtime fit. Quantization, sampling settings, output caps, prompt design, and software versions are confounders and must remain explicit. Unigram overlap and headings are weak indicators; fluent output may still invent details or violate continuity.

The small dataset supports a narrow exploratory task and invites overfitting in later adaptation. Text-only evaluation cannot establish multimodal capability. Cases 2-6 are authorized for modest local experiments; they still require appropriate resources and an unbiased evaluation protocol. Future conclusions should include failed loads, memory stops, quality regressions, and incomplete outputs.

## Conclusion placeholder

One coherent output establishes that text-only GPU generation can run after other applications are closed. The sample is too small to establish dataset quality or stable full-run resource use. The completed ten-scene CPU sample exposes frequent truncation, repetition, and scene-fidelity failures. Full-set measurements remain pending. Later-case conclusions remain empty until their experiments are performed.

## Sources and reproducibility

Primary sources consulted on 2026-11-11 are cited in each case: Unsloth artifact listings; Qwen configuration/model card; llama-cpp-python/llama.cpp/GGUF documentation; PEFT and MLX-VLM; LLaMA Pro v2; ShortGPT v3. Case 1 records exact downloaded artifact revisions/hashes; later experiments must do the same. Raw technical outputs belong in results, with concise interpretation in the matching Findings section.
