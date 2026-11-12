# VL GGUF lab on Google Colab

Six notebooks run six experiments on 150 scene-description / shot-breakdown pairs. The dataset is committed in this private repository; weights download from pinned Hugging Face revisions. All inputs are text. These experiments test the language side of a vision-language model, not image understanding.

| Case | One notebook | Full experiment |
| --- | --- | --- |
| 1 | [Baseline](notebooks/01_baseline.ipynb) | All 150 scenes, VL GGUF, truncation and structure scores |
| 2 | [Train and re-test](notebooks/02_train_retest.ipynb) | SmolLM2-1.7B response-only LoRA, 120 train / 15 held-out test scenes |
| 3 | [Modify layers](notebooks/03_modify_layers.ipynb) | F16 control: 3 norm scales at 3 layer positions |
| 4 | [Add layers](notebooks/04_add_layers.ipynb) | F16 control: duplicate a block at 3 positions |
| 5 | [Remove layers](notebooks/05_remove_layers.ipynb) | F16 control: remove a block at 3 positions |
| 6 | [Own transformer](notebooks/06_own_transformer.ipynb) | Decode F16 GGUF into our PyTorch decoder, check logits, train and re-test |

Open a notebook and click its **Open in Colab** badge. Select **Runtime > Change runtime type > T4 GPU** or a larger GPU. Add a `GH_TOKEN` with repository read access to Colab Secrets and allow notebook access, since the repo is private. The setup cell clones the repository without storing the token in Git configuration, installs pinned dependencies, checks `nvidia-smi`, and installs the official CUDA llama-cpp-python wheel. If packages were imported before setup, restart the Colab session and run from the top. A missing CUDA backend raises an error instead of silently running the full experiment on CPU.

The badges and setup currently select `fm/vlgguf-colab`, so they work before merge. Once this branch is deleted, select `main` in the Colab URL and set `REPO_REF = "main"` in setup. Use one notebook at a time. Run all cells in order, inspect the config, authorize Drive mounting, and keep the printed run folder name. Full experiments can take hours and may exceed a free Colab session; resumable storage is part of the workflow, not a guarantee of uninterrupted GPU availability.

## Models and GPU tiers

Set `CONFIG["model_tier"]` in case 1. Files and immutable revisions were verified against the Hugging Face API on 2026-11-12. The catalog is authoritative in [src/lab_models.py](src/lab_models.py).

| Tier | Hardware target | Exact file in `unsloth/Qwen2.5-VL-7B-Instruct-GGUF` |
| --- | --- | --- |
| `t4` (default) | T4, 16 GB | `Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf` |
| `l4` | L4, 24 GB | `Qwen2.5-VL-7B-Instruct-Q8_0.gguf` |
| `a100` | A100, 40 GB or larger | `Qwen2.5-VL-7B-Instruct-BF16.gguf` |

The larger tiers increase weight precision. KV cache, context and runtime buffers also consume GPU memory; actual fit and speed require a Colab run. All decoder layers request GPU offload (`gpu_layers=-1`). Vision projectors are unnecessary for these text-only inputs. See the [upstream model files](https://huggingface.co/unsloth/Qwen2.5-VL-7B-Instruct-GGUF/tree/68bb8bc4b7df5289c143aaec0ab477a7d4051aab) and [official CUDA installation instructions](https://github.com/abetlen/llama-cpp-python#installation).

Case 2 uses pinned `HuggingFaceTB/SmolLM2-1.7B-Instruct` with LoRA on query/value projections. Cases 3-6 use `bartowski/SmolLM2-135M-Instruct-GGUF/SmolLM2-135M-Instruct-f16.gguf`, an unquantized text-model control that supports safe tensor decoding and editing. Their results do not establish the effect of modifying a 7B VL model. Case 6 trains all parameters of our independent decoder. Both training paths checkpoint block activations during backward to limit GPU memory use. Increasing the VL baseline tier does not change these controlled models. Model files and checkpoints stay out of Git.

## Drive saves and resumption

The shared helper mounts Drive and saves to `MyDrive/vl-gguf-lab/results/<case>/runs/<UTC timestamp>-<id>/` by default. Set `CONFIG["drive_root"]` to another relative Drive folder before starting. Each run contains immutable config and dataset/split hashes, environment versions and Git revision, raw scene JSONL, parsed structure and overlap scores, metrics, event/runtime logs, charts, and training checkpoints where relevant. Edited model bytes are temporary; saved tensor hashes and block maps make each variant reproducible from its pinned source.

Scene rows and training-step logs are flushed incrementally. Metrics update after every scene. Training checkpoints save optimizer state, trainable weights, RNG state and the completed step, every 10 steps and at completion (configurable with `checkpoint_every`). A disconnect can lose at most that interval of training work; logs after the last checkpoint are replayed correctly. Drive is remote storage: do not assume POSIX atomicity or durability through an active Drive outage. If a final JSONL append was interrupted, resumption discards only that incomplete tail.

To resume, rerun setup, keep exactly the same `CONFIG`, and set `RESUME` to the timestamped folder name. The helper rejects changed provenance/config. Completed scenes are skipped; training resumes from its optimizer checkpoint. To change caps, tiers, epochs or splits, start a new run. Outside Colab, the same helper uses `results/<case>/runs/` without requiring Google packages; `LOCAL_ROOT` can override it.

## Method and interpretation

Case 1 evaluates all 150 inputs with an 8,192-token context and a 4,096-token output cap. Every run reports finish reason and truncation rate. Cases 2-6 share the frozen input-grouped 120/15/15 train/validation/test split in `src/freeze_split.py`; evaluation uses the same 15 test scenes. Default training is three shuffled epochs, response-only supervision, a 2,048-token training cap, and checkpointed AdamW updates. Training reports how many targets were clipped. Validation is reserved; this fixed-budget protocol does not select checkpoints or variants by test score.

Every generated scene reports unigram/bigram F1, shot-count agreement and parsed time/lens/framing/movement recall against the reference. The combined structure score is a proxy, not a judgment of cinematic quality. Cases 3-5 include a no-change rewrite and paired score differences for every predetermined variant, with descriptive scene-bootstrap intervals. Do not pick the best test variant and present it as an independently validated improvement. Related scenes and repeated templates limit inference from this small dataset.

Case 6 re-runs the original strict per-logit criterion (`atol=0.001`, `rtol=0.001`) and saves both logits. The historical NumPy implementation matched argmax but failed this criterion. The new decoder does not assume that gap is fixed: a failing parity check is reported plainly and all subsequent training is labeled exploratory. Its implementation, cache and gradients are locally tested; GPU parity and full training results remain to be measured.

## Local validation

Use Python 3.12. Install portable requirements and a platform-appropriate llama.cpp build:

```sh
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
# Apple Silicon; use the official CPU wheel index on Linux without CUDA instead.
CMAKE_ARGS='-DGGML_METAL=ON' uv pip install --python .venv/bin/python llama-cpp-python==0.3.36
.venv/bin/python -m pytest
VL_LAB_SMOKE=1 .venv/bin/jupyter execute notebooks/01_baseline.ipynb
```

For each notebook, set `VL_LAB_SMOKE=1` before execution for a real, bounded CPU path: small 135M model, one scene, 24 generated tokens, one training step, and one layer position. This deliberately produces capped answers. It checks execution and persistence, not experiment quality. Set `VL_LAB_RESULTS` to an external folder to keep verification outputs separate. Run the same command on each of the six notebooks. Colab-only checks are GPU installation/offload, real Drive mounting and reconnect behavior, VRAM fit and the full runs.

## Repository map

`notebooks/` has exactly six entry points. Shared portable modules are `src/lab_*.py`; the older `src/case*.py` and `src/baseline.py` preserve the Mac workflow for historical investigation with its separate `requirements-macos.txt`. `data/` holds the dataset and provenance; `models/` holds only the historical manifest in Git. `results/01/` through `06/` each contain untouched historical evidence in `local-2026-11-11/`, with new runtime folders ignored by Git. `docs/01-06` explain each method and `docs/07-paper.md` is the paper draft. Colab result tables are intentionally pending, not projected numbers.
