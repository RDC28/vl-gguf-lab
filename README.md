# Local VL GGUF lab

Test whether a local Qwen vision-language model can turn scene descriptions into shot breakdowns. The supplied dataset is text only, so this baseline tests the language side of a VL model. It does not test image understanding.

| Folder | Contents |
| --- | --- |
| docs | Six case studies and the draft paper; each case keeps its findings |
| data | Ignored local dataset copy and provenance note |
| models | Ignored model files; committed download manifest |
| notebooks | Jupyter baseline entry point |
| src | Download, inference, and comparison code |
| results | Raw sample outputs, timings, and run metadata |

Python 3.12 and an Apple Silicon Mac with Metal are used. The local environment is `.venv`.

```sh
uv venv --python 3.12.13 .venv
CMAKE_ARGS='-DGGML_METAL=ON' uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python -m ipykernel install --prefix .venv --name vl-gguf-lab --display-name 'VL GGUF Lab'
cp /Users/rohit/Downloads/dataset.json data/dataset.json
.venv/bin/python src/download_model.py
.venv/bin/python src/baseline.py --output results/new-baseline.jsonl
.venv/bin/jupyter lab notebooks/
```

The download command checks the manifest hashes and reuses already verified files. It downloads one language model and one vision projector. The projector stays unloaded for text-only inference. Do not download another quant for this baseline.

Close other memory-heavy work before inference. Run one inference process at a time. The script refuses to load under memory warnings or below 15% system memory headroom and stops generation on critical memory pressure. These checks reduce risk; they cannot reserve RAM against other applications. Exit the notebook kernel after inference. Stop Jupyter with Ctrl-C when finished.

The script refuses to overwrite result files. Resume the saved sample with `.venv/bin/python src/baseline.py --resume --max-new-rows 1`; remove `--max-new-rows 1` after a successful probe to finish the sample. Resume validates dataset/model/settings provenance, skips completed indexes, syncs each appended row to disk, and keeps separate timestamped attempt metadata. Use a separate output path for CPU-only runs (`--n-gpu-layers 0`). Default sampling is 10 sorted, randomly selected row indexes using seed 42. Generation is greedy, with a 1,024-token output cap. A context overflow raises an error rather than silently truncating the input. See docs/01-baseline.md for the actual measured run and its limits.

Read docs/01-baseline.md through docs/06-transformer-plan.md in order. docs/07-paper.md ties the research together. Case 2 has a frozen scene-group split in results/case2-split.json. The pinned SmolLM2-135M-Instruct text model is downloaded locally for case 2; file hashes and architecture inspection are in results/case2-model-provenance.json. Its execution and training remain pending. Tiny local training, tensor edits, and custom architecture experiments are authorized next steps; none has run yet. See each case for measured status.
