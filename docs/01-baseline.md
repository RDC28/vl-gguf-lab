# Use case 1: local baseline

## Goal

 generate shot breakdowns from text scene descriptions without training.

## Method and concrete plan

Use `src/baseline.py` or `notebooks/baseline.ipynb`. Both call the same Python implementation. Results contain the original row index, input, reference, output, finish reason, elapsed time, generation throughput, peak process RSS, word counts, SHOT heading counts, and unigram precision/recall/F1.

Prompt: a fixed storyboard assistant instruction asks for numbered SHOT headings, framing, camera movement, action, and sound. Only the input description follows it. The model's embedded chat template is used by llama-cpp-python. References never enter the prompt.

Settings: seed 42; temperature 0; top-p 1; top-k 1; repeat penalty 1; 2,048 context tokens; at most 1,024 new tokens; 128-token batch; 4 CPU threads; all supported layers offloaded to Metal; memory mapping enabled. Context includes prompt plus output. Output caps can cut off a breakdown; the finish reason exposes that.

The sample is a sorted random sample without replacement of 10 of the 150 rows. Results are flushed after each row. The model is explicitly closed even on an error. The metadata file records errors and pressure samples. No parallel inference or image encoding runs.

Unigram F1 uses lowercased word-token multisets and clipped token counts. It measures shared vocabulary, not narrative correctness, camera quality, continuity, or semantic equivalence. SHOT heading counts detect numbered headings at line starts, allowing Markdown bold/headings. Length and structure are only diagnostic. Human review remains necessary.

Load time includes the Python backend import/system-info initialization plus the model constructor. Both components are also recorded separately. Total run time starts at model construction; add backend initialization for full runtime startup. It excludes dataset validation and artifact download.

Generation tokens/s uses llama.cpp evaluation counters, excluding prompt evaluation. Wall time includes prompt processing and Python overhead. Peak process RSS is the macOS process high-water mark, not total system RAM or a complete GPU allocation measure. Pressure samples are system-wide. Metal unified-memory buffers may be accounted differently; do not treat RSS as total application memory.

## Tools and versions

Python 3.12.13; llama-cpp-python 0.3.36; vendored llama.cpp commit `0c1e57098bba43ac29e6e3b677cdceebdd22334f`; gguf 0.19.0; huggingface_hub 2.4.0; JupyterLab 4.6.6; ipykernel 7.4.0. All installed dependencies are pinned in the root requirements file.

## Feasibility on the 8 GB Air

The 3.81 GB language file leaves limited headroom for the operating system, model working buffers, and the KV cache. File size alone does not prove that inference fits. The projector remains unloaded. Use one process, modest context, pre-load checks, and continuous pressure sampling. Stop on critical pressure instead of forcing completion. Actual memory and speed remain measurements to make.

## Risks and open questions

The 10-row sample is exploratory and cannot establish reliability over all scenes. Quantization, prompt wording, truncation, and dataset style affect results. Greedy decoding reduces sampling variation but is not a promise of bit-identical output across software or hardware changes. The model may invent framing or sound not specified by the scene. Text-only results do not establish vision competence.

## Findings

The setup facts below are observed. Inference results, load time, generation speed, peak memory, completion-cap counts, comparison table, and full-dataset time estimate will be added after the baseline runs.

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
