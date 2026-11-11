# Case 1: local baseline inference

## Goal

Generate shot breakdowns from text scene descriptions without training.

## Method and concrete plan

Use `src/baseline.py` or `notebooks/baseline.ipynb`. Both call the same Python implementation. Results contain the original row index, input, reference, output, finish reason, elapsed time, generation throughput, peak process RSS, word counts, SHOT heading counts, and unigram precision/recall/F1.

Prompt: a fixed storyboard assistant instruction asks for numbered SHOT headings, framing, camera movement, action, and sound. Only the input description follows it. The model's embedded chat template is used by llama-cpp-python. References never enter the prompt.

Settings: seed 42; temperature 0; top-p 1; top-k 1; repeat penalty 1; 2,048 context tokens; at most 1,024 new tokens; 128-token batch; 4 CPU threads for generation and prompt batches; all supported layers offloaded to Metal; memory mapping enabled. Context includes prompt plus output. Output caps can cut off a breakdown; the finish reason exposes that.

The sample is a sorted random sample without replacement of 10 of the 150 rows. Results are flushed after each row. The model is explicitly closed even on an error. The metadata file records errors and pressure samples. No parallel inference or image encoding runs.

Unigram F1 uses lowercased word-token multisets and clipped token counts. It measures shared vocabulary, not narrative correctness, camera quality, continuity, or semantic equivalence. SHOT heading counts detect numbered headings at line starts, allowing Markdown bold/headings and numbered or bulleted list prefixes. Length and structure are only diagnostic. Human review remains necessary.

Load time includes the Python backend import/system-info initialization plus the model constructor. Both components are also recorded separately. Total run time starts at model construction; add backend initialization for full runtime startup. It excludes dataset validation and artifact download.

Generation tokens/s uses llama.cpp evaluation counters, excluding prompt evaluation. Wall time includes prompt processing and Python overhead. Peak process RSS is the macOS process high-water mark, not total system RAM or a complete GPU allocation measure. Pressure samples are system-wide. Metal unified-memory buffers may be accounted differently; do not treat RSS as total application memory.

## Tools and versions

Python 3.12.13; llama-cpp-python 0.3.36; vendored llama.cpp commit `0c1e57098bba43ac29e6e3b677cdceebdd22334f`; gguf 0.19.0; huggingface_hub 2.4.0; JupyterLab 4.6.6; ipykernel 7.4.0. All installed dependencies are pinned in the root requirements file.

## Feasibility on the 8 GB Air

The 3.81 GB language file leaves limited headroom for the operating system, model working buffers, and the KV cache. File size alone does not prove that inference fits. The projector remains unloaded. Use one process, modest context, pre-load checks, and continuous pressure sampling. Stop on critical pressure instead of forcing completion. Observed memory and speed are recorded below; safe fit depends on other running work.

## Risks and open questions

The 10-row sample is exploratory and cannot establish reliability over all scenes. Quantization, prompt wording, truncation, and dataset style affect results. Greedy decoding reduces sampling variation but is not a promise of bit-identical output across software or hardware changes. The model may invent framing or sound not specified by the scene. Text-only results do not establish vision competence.

## Findings

The setup and attempts below are observed. The first successful output is exploratory; the overnight continuation will extend the sample.

### 2026-11-11: setup and model selection

Hardware: Apple Silicon MacBook Air (MacBookAir10,1), Apple M1, 8 CPU cores (4 performance and 4 efficiency), 8,589,934,592 bytes RAM (8 GiB). Other applications were running. Initial macOS memory pressure level was 1 (normal); `memory_pressure` reported 50% system-wide free percentage. That percentage includes reclaimable headroom and is not the same as unused physical pages.

Dataset: 454,187 bytes; 150 JSON objects; exactly two text fields. This experiment cannot establish vision quality because it contains no images. The language model is 7B class, which is relatively small among VL models but is not a tiny model for an 8 GB laptop.

The strict file limit is interpreted as 4,000,000,000 bytes, not 4 GiB. Hugging Face API listings checked on 2026-11-11:

| Unsloth repository | Quant | Model bytes | Fits under 4 GB |
| --- | --- | ---: | --- |
| Qwen2.5-VL-7B-Instruct-GGUF | Q3_K_S | 3,492,367,232 | Yes |
| Qwen2.5-VL-7B-Instruct-GGUF | Q3_K_M | 3,808,390,016 | Yes, selected |
| Qwen2.5-VL-7B-Instruct-GGUF | Q4_K_M | 4,683,072,384 | No |
| Qwen3-VL-8B-Instruct-GGUF | Q3_K_S | 3,769,613,152 | Yes |
| Qwen3-VL-8B-Instruct-GGUF | Q3_K_M | 4,124,162,912 | No |
| Qwen3-VL-8B-Instruct-GGUF | Q4_K_M | 5,027,785,568 | No |

Selected repository: `unsloth/Qwen2.5-VL-7B-Instruct-GGUF`.
Pinned revision: `68bb8bc4b7df5289c143aaec0ab477a7d4051aab`.
Selected file: `Qwen2.5-VL-7B-Instruct-Q3_K_M.gguf`.
Quantization: Q3_K_M (mixed K-quants, nominally 3-bit class; actual file includes higher precision tensors and metadata).

Q3_K_M preserves more weight precision than the smaller Q3_K_S variant while staying under the file cap. It sacrifices precision relative to 4-bit and floating-point weights. No controlled comparison of quants was run, so the quality cost is not measured. Qwen3-VL is a newer alternative, but its 8B Q3_K_M exceeds the cap. The requested 7B model fits without choosing a more aggressive fallback.

| Artifact | Bytes | SHA256 |
| --- | ---: | --- |
| Qwen2.5-VL-7B-Instruct-Q3_K_M.gguf | 3,808,390,016 | accacb52744cadbe58b49f37ebb44e094f8de7ee2b52eb0b51ed9289329ef0c1 |
| mmproj-F16.gguf | 1,354,163,040 | 987dd0733033fb5dd9b124d1ca926ae865572e432384eee7618b2eec3e735e17 |

The 4 GB limit applies to the language GGUF alone. Both downloaded artifacts together occupy 5,162,553,056 bytes. The F16 vision projector is downloaded for later image work and is not loaded in the text baseline. The manifest records upstream hashes; the downloader verifies actual bytes before accepting each artifact. Only one language quant is downloaded.

Sources: [Qwen2.5-VL Unsloth files](https://huggingface.co/unsloth/Qwen2.5-VL-7B-Instruct-GGUF/tree/68bb8bc4b7df5289c143aaec0ab477a7d4051aab), [Qwen3-VL Unsloth files](https://huggingface.co/unsloth/Qwen3-VL-8B-Instruct-GGUF/tree/b93a7ee713758252c555be4210c00540df954dc2).

### Runtime provenance

The environment uses Python 3.12.13. Package versions are pinned in `requirements.txt`. llama-cpp-python 0.3.36 vendors llama.cpp commit `0c1e57098bba43ac29e6e3b677cdceebdd22334f`, as recorded by the upstream tag's submodule reference. The inference metadata records runtime Metal support and package version. Runtime context, sampling, and memory policy are described in `01-baseline.md`.

Sources: [llama-cpp-python v0.3.36 vendor reference](https://github.com/abetlen/llama-cpp-python/tree/v0.3.36/vendor), [Metal installation instructions](https://github.com/abetlen/llama-cpp-python#installation).

### Verified language-file metadata

A read-only gguf 0.19.0 inspection found architecture `qwen2vl`, 28 blocks, hidden width 3,584, feed-forward width 18,944, 28 attention heads, 4 KV heads, and a declared 128,000-token model context. The baseline still uses only 2,048 tokens. The file has 339 tensors and 7,615,616,512 tensor elements. Tensor types: 141 F32, 113 Q3_K, 81 Q4_K, 3 Q5_K, and 1 Q6_K. This confirms that Q3_K_M mixes precisions; file size averages about 4 bits per tensor element including metadata. The raw inventory is `../results/model-metadata.json`. This inspection did not modify weights or construct a new model.

Casting all these language tensor elements to BF16 alone would require 15,231,233,024 bytes (15.23 GB, 14.19 GiB), before training state or runtime buffers. This is an estimate from measured tensor count, not a training memory measurement.

### Local attempts and first successful GPU output

All three attempts used the same 7B Q3_K_M file, full Metal offload (`n_gpu_layers=-1`), text only, 2,048 context tokens, 1,024-token generation cap, batch 128, and four CPU threads for prompt/generation. No critical pressure was observed. The guard requires normal pressure and at least 15% reported free memory before each input; during generation it stops on critical level 4 or below 8% reported free.

| Attempt / raw file stem | Free before | Lowest observed free | Highest pressure | Load seconds | Peak process RSS bytes | Completed outputs | Outcome |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| memory-stop | 46% | 11% at guard, 12% in periodic samples | 2 | 8.714 | 1,976,238,080 | 0 | Stopped safely before generation |
| memory-stop-retry | 66% | 11% at guard, 55% in periodic samples | 2 at guard | 3.052 | 3,615,211,520 | 0 | Same safe stop; coarse samples missed the low point |
| baseline, after other apps were closed | 76% | 16% in periodic samples | 1 | 4.882 | 3,546,480,640 | 1 | Generated row 6, then deliberately interrupted during the next row for overnight handoff |

Free percentages are the values reported by `memory_pressure`, not unused physical page counts. The first two attempts closed the model and the system subsequently returned to normal. Immediate after-close snapshots can precede asynchronous memory recovery. GPU working memory is not fully represented by process RSS: the runtime separately logged a 3,626.30 MiB Metal model allocation, a 77.75 MiB Metal compute buffer, and a 5.00 MiB CPU compute buffer. Raw logs and metadata are in results with the file stems above. File verification and read-only metadata inspection preceded loading, so file-cache warmth was not controlled.

The authorized third attempt produced one complete, uncapped output for zero-based dataset index 6 in 143.792 seconds. llama.cpp measured 903 generation evaluation tokens in 139.881 seconds, or 6.4555 tokens/s. The process was then deliberately stopped with SIGINT; its metadata records KeyboardInterrupt and a 214.259-second run including unfinished work on the second input. This is a handoff stop, not a memory failure.

| Row | Output words | Reference words | Output SHOT headings | Reference SHOT headings | Unigram F1 | Finish reason |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 6 | 621 | 448 | 6 | 4 | 0.2320 | stop |

The output retains Naisha, Zain, the tea stall, and nervous talking. It repeats similar medium shots and invents Zain's concerned reaction. Its six shots differ from the reference's four. This is limited manual inspection of one example, not a quality verdict on the dataset.

The real output exposed a heading counter bug: `1. **SHOT 1...` was originally counted as zero because of the list prefix. The matcher now accepts that syntax. The original JSONL is preserved unchanged; `baseline-comparison.json` contains regenerated comparison metrics, including the corrected six-heading count. The source matcher was checked against the actual output and reference.

At this one-input wall time, 10 inputs would take roughly 24 minutes and 150 roughly 6 hours, plus load time. This is an extrapolation from one prompt with first-use shader/prompt overhead and uncontrolled thermal/cache conditions. It is not a measured full-run duration. Overnight work will resume completed row indexes, keep one model loaded at a time, and update this estimate with additional evidence.

### 2026-11-11 17:13 UTC: bounded resume probe

The CLI now supports `--resume --max-new-rows 1`. It validates the dataset hash, model manifest, prompt, settings, saved row indexes and source strings before appending only missing selected indexes. The default fixed sample stays unchanged, so the first pending index is 22 and row 6 is reused. Each new row is flushed and `fsync`ed; new rows identify their attempt. Each attempt has a timestamped metadata file, leaving the original metadata untouched. A partial trailing JSONL line is rejected for inspection rather than repaired or silently discarded. Changing `--limit` to 150 can extend the selection with the same inference settings; changing model/settings requires a separate output path. The notebook uses this same resume entry point.

The actual command was `.venv/bin/python src/baseline.py --resume --max-new-rows 1`, with stdout/stderr saved in `results/baseline-resume-probe.runtime.log`. Model bytes were SHA256-verified before loading. The full GPU load started at normal pressure and 70% reported free, took 4.283 seconds including backend initialization (4.004 seconds constructor), and reached peak process RSS 3,386,982,400 bytes. Before input 22, the guard observed warning pressure level 2 and 15% reported free, and stopped before generation. Periodic readings reached 15% free; immediate after-close pressure/free remained 2/15%, before asynchronous recovery. RSS excludes some Metal allocations. No new output completed, no critical pressure was observed, and the model was explicitly closed. Raw evidence is `results/baseline.attempt-20261111T171353192781Z.metadata.json`. The guard failure state is preserved in its error string; subsequent attempts also record the failed before-input snapshot explicitly.

The baseline still contains only one completed row. Neither its raw bytes nor its original metadata were changed. This retry shows that 70% free before loading alone does not guarantee safe full-GPU inference. The next bounded experiment should use the same immutable 7B Q3 file with `--n-gpu-layers 0`, a separate output path, and `--max-new-rows 1`; CPU results must remain separate. No CPU speed or safe fit is established yet. The original one-row full-set estimate remains provisional.

Validation: real CLI reproduction of the existing-file refusal preceded the change. CLI checks reject mismatched settings, duplicate indexes, and incomplete trailing rows without modifying evidence; an already-completed selection exits without model loading. Temporary synthetic fixtures were removed. Notebook schema/code-cell checks, Python compilation, dependency compatibility (`uv pip check`, 106 packages), and diff whitespace checks passed. `results/resume-validation.json` records these checks. Successful appended generation remains to be exercised once a configuration passes the memory guard.

### 2026-11-11 17:15 UTC: CPU-only configuration probe

The same SHA256-verified immutable 7B Q3_K_M file completed a CPU-only probe with `.venv/bin/python src/baseline.py --output results/baseline-cpu-probe.jsonl --n-gpu-layers 0 --max-new-rows 1`. All other settings and the prompt remained unchanged. The separate CPU sample starts at row 6, allowing a configuration comparison with the existing GPU output. This is not an additional unique scene and does not increase the GPU sample count. References remained outside the prompt. The runtime confirms zero of 29 layers offloaded, mmap enabled, a 3,199.93 MiB CPU-mapped model buffer, a 1,675.20 MiB CPU repack buffer, and a 112 MiB CPU KV buffer. Buffer sizes are not additive resident-memory measurements.

| Measurement | CPU probe |
| --- | ---: |
| Load including backend initialization | 4.256 seconds |
| Constructor alone | 3.997 seconds |
| Input wall time | 114.320 seconds |
| Prompt evaluation | 72 tokens / 6.151 seconds |
| Generation evaluation | 1,023 tokens / 107.301 seconds |
| Generation evaluation throughput | 9.534 tokens/s |
| Peak process RSS | 4,330,668,032 bytes |
| Free before / minimum sampled / after close | 68% / 42% / 68% |
| Highest pressure across 60 samples | 1 (normal) |
| Output / reference words | 704 / 448 |
| Output / reference SHOT headings | 8 / 4 |
| Unigram F1 | 0.22396 |
| Finish reason | length |

The CPU output retains the characters, tea stall, and nervous speech, but invents confusion and frustration for Zain. Shots 5-7 substantially repeat shots 2-4, and shot 8 ends mid-description at the generation cap. The CPU and GPU outputs differ despite greedy settings; no numerical parity or quality advantage is claimed. The CPU throughput exceeded the earlier GPU measurement under uncontrolled thermal/cache and operating-system conditions, with different generated text and token counts. One run does not establish a general CPU speed advantage. RSS remains a process measure, not all unified-memory allocations.

Raw output, runtime log, and timestamped metadata use the `results/baseline-cpu-probe` stem; `results/baseline-cpu-comparison.json` records validated metrics and output hashes. The process exited normally and closed the model. The original GPU row and metadata remain unchanged. At this CPU input wall time, 10 inputs would take about 19 minutes and 150 about 4.76 hours plus loading, an extrapolation from one capped input rather than measured dataset duration.

The next bounded step is to resume this separate CPU sample for its nine missing indexes using `--resume --n-gpu-layers 0` and the same output path, with a bounded row count. This successful safe probe provides no reason to download the 3B fallback yet. Cases 2-6 remain unattempted.

A follow-up resume validation mistakenly used `--limit 1`, which draws a new seed-42 sample (index 28), rather than selecting the first member of the ten-row sample. It started inference and was deliberately SIGINT-stopped without completing a row; its separate attempt metadata preserves `KeyboardInterrupt` and memory readings. This was a validation-command mistake, not a memory failure. Keep `--limit 10` (the default) and use `--max-new-rows` to bound continuation. No runtime log was redirected for this interrupted validation attempt.

### 2026-11-11 17:19 UTC: two-row CPU continuation

Resumed the unchanged CPU configuration with `.venv/bin/python src/baseline.py --output results/baseline-cpu-probe.jsonl --resume --n-gpu-layers 0 --max-new-rows 2`. The fixed ten-row selection remained [6, 22, 26, 28, 35, 57, 62, 70, 108, 139]; row 6 was reused byte for byte and only indexes 22 and 26 were appended. Seven sample rows remain. No fallback download was needed.

| Index | Wall seconds | Generation tokens/s | Output/reference words | Output/reference headings | Unigram F1 | Finish |
| ---: | ---: | ---: | --- | --- | ---: | --- |
| 22 | 125.468 | 8.638 | 661 / 440 | 10 / 4 | 0.20890 | length |
| 26 | 134.755 | 7.817 | 673 / 437 | 10 / 4 | 0.19279 | length |

Each evaluated 1,023 generation tokens against the 1,024-token cap. The attempt loaded in 5.669 seconds including backend initialization and lasted 265.761 seconds. Across 135 memory samples, pressure stayed normal and the minimum reported free memory was 42%; before load and after close were 69% and 67%. Peak process RSS was 4,089,675,776 bytes; RSS does not account for all Metal allocations. The process exited normally and closed its model. Raw runtime log: `results/baseline-cpu-continuation-01.runtime.log`; timestamped attempt: `results/baseline-cpu-probe.attempt-20261111T171956487079Z.metadata.json`.

Manual inspection: row 22 asks for exact costume/location continuity with a previous photo that is absent from this text-only input. The model explicitly assumes traditional clothing and a Diwali market, then invents shopping actions. It cannot establish the requested exact match. Row 26 retains Naisha, Zain, an empty morning bridge, and reluctance to pose, but repeats photographing/reaction shots and ends mid-section. These are two limited observations, not systematic human scores.

All three CPU outputs are capped. Mean input wall time is 124.848 seconds, mean unigram F1 is 0.20855, and an extrapolation is 20.81 minutes for ten inputs or 5.20 hours for 150, excluding loading. This replaces the one-row CPU estimate; the full dataset duration remains unmeasured. Generation throughput declined across these observations, but no thermal measurements were collected and no cause is established.

`results/baseline-cpu-sample-summary.json` records the cumulative metrics and remaining indexes. Validation checked unique indexes, dataset hash/source strings, recomputed metrics, and successful attempt completion. Existing GPU output/metadata and the earlier CPU row/attempts remain unchanged. Next: resume the same output/configuration for the remaining seven fixed-sample indexes before deciding how much full-set inference fits alongside cases 2-6.

### 2026-11-11 17:26 UTC: second two-row CPU continuation

The same resume command and unchanged settings appended only indexes 28 and 35. Prior CPU rows remain byte-identical. The fixed sample now contains five completed scenes; indexes 57, 62, 70, 108, and 139 remain.

| Index | Wall seconds | Evaluated generation tokens | Generation tokens/s | Output/reference words | Output/reference headings | Unigram F1 | Finish |
| ---: | ---: | ---: | ---: | --- | --- | ---: | --- |
| 28 | 83.654 | 692 | 9.001 | 466 / 428 | 6 / 4 | 0.24161 | stop |
| 35 | 118.406 | 680 | 5.923 | 458 / 450 | 6 / 4 | 0.25330 | stop |

These are the first naturally stopped CPU outputs; neither hit the generation cap. Load including backend initialization took 9.127 seconds (8.881 seconds constructor), and the attempt lasted 211.060 seconds. Peak process RSS was 3,969,187,840 bytes, which does not include all Metal allocations. Reported free memory was 68% before load, a minimum 36% during sampling, and 62% after closing. Across 108 samples, ten recorded warning pressure level 2 during generation; none recorded critical pressure level 4 or free memory below 8%. Both before-input guards observed normal pressure (42% free for index 28, 60% for index 35). Thus the existing policy permitted completion without being weakened. Pressure returned to normal after each input and after closing. These warnings qualify the earlier CPU observations of consistently normal pressure; safe headroom remains dependent on system conditions.

Manual inspection: index 28 depicts shyness followed by confidence, but never explicitly calls action, introduces possible singing, and invents a bow and exit. Index 35 retains the café power cut but does not allocate the requested 15 seconds, keeps essentially one framing across six headings, and invents phone checking and power restoration. Natural stopping is not evidence of scene fidelity or useful distinct shot coverage.

Across five CPU scenes, three are capped, mean input wall time is 115.321 seconds, and mean unigram F1 is 0.22411. The updated extrapolation is 19.22 minutes for ten inputs or 4.81 hours for 150, excluding load; full-set duration remains unmeasured. Row 35's lower throughput has no established cause because thermals and competing system activity were not measured.

Raw evidence: `results/baseline-cpu-continuation-02.runtime.log`, `results/baseline-cpu-probe.attempt-20261111T172633499026Z.metadata.json`, and appended `results/baseline-cpu-probe.jsonl` rows. The cumulative summary was refreshed after verifying source strings, dataset hash, unique indexes, recomputed metrics, matching attempt IDs, successful attempt completion, and preservation of earlier CPU bytes and original GPU output/metadata. The inference process exited normally and closed its model. No code or dependency changed. Next: continue the five missing fixed-sample scenes in bounded slices, then checkpoint case 1 before cases 2-6.

### 2026-11-11 17:36 UTC: third two-row CPU continuation

The unchanged CPU resume command with `--max-new-rows 2` appended indexes 57 and 62. Seven of the fixed ten sample scenes are now complete; 70, 108, and 139 remain. All prior CPU rows and original GPU output/metadata remain byte-identical.

| Index | Wall seconds | Evaluated generation tokens | Generation tokens/s | Output/reference words | Output/reference headings | Unigram F1 | Finish |
| ---: | ---: | ---: | ---: | --- | --- | ---: | --- |
| 57 | 132.444 | 1023 | 8.177 | 769 / 426 | 6 / 4 | 0.20418 | length |
| 62 | 81.823 | 563 | 7.186 | 383 / 423 | 5 / 4 | 0.23573 | stop |

Load including backend initialization took 7.759 seconds; total attempt time was 221.885 seconds. Peak process RSS was 3,957,096,448 bytes, excluding some Metal allocations. Before-load and after-close pressure were normal with 66% and 62% reported free memory, respectively. Across 113 samples, minimum free memory was 40%; three samples showed warning pressure level 2 during generation. Both before-input checks observed normal pressure. No critical pressure or below-8% reading occurred; guards were unchanged and the model closed normally.

Manual inspection: index 57 retains all four named cats on one rooftop and assigns separate activities, but ends mid-summary at the cap. Text generation cannot demonstrate visual identity preservation or absence of morphing. Index 62 uses a human silhouette and ordinary underpass sounds without explicit supernatural events, but omits the name Ulluman and repeats slow mechanical movement across shots. These observations do not establish systematic quality scores.

Across seven CPU scenes, four are capped, mean input time is 112.981 seconds, and mean unigram F1 is 0.22293. Estimated time is 18.83 minutes for ten or 4.71 hours for 150, excluding loading; these remain extrapolations rather than measured full-run durations. Cases 2-6 remain unattempted.

Raw evidence: `results/baseline-cpu-continuation-03.runtime.log`, `results/baseline-cpu-probe.attempt-20261111T173657628480Z.metadata.json`, and the appended JSONL rows. The cumulative summary was refreshed after checking the dataset hash, all seven source strings and recomputed metrics, unique indexes, successful attempt completion, and byte preservation of previous evidence. No code or dependencies changed. Next: complete the three remaining fixed-sample scenes in a bounded continuation before moving to case 2.
