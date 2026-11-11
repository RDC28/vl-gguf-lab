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
