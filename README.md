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

Read docs/01-baseline.md through docs/06-transformer-plan.md in order. docs/07-paper.md ties the research together. Case 2 has a frozen scene-group split in results/case2-split.json. The pinned SmolLM2-135M-Instruct text model is downloaded locally for case 2; file hashes and architecture inspection are in results/case2-model-provenance.json. A guarded 128-token LoRA forward/backward preflight completed; its untrained 15-scene held-out baseline is recorded in results/case2-before*, and 20 guarded LoRA updates have saved a small adapter under `models/case2-lora`; the identical-settings 15-scene held-out re-test is recorded in `results/case2-after*` (mean unigram F1 0.14008 versus 0.12496 before, all after-outputs capped). Tiny local training, tensor edits, and custom architecture experiments are authorized next steps; the first tiny-model tensor edit is verified and loads with unchanged fixed-prompt text; custom architecture experiments remain pending. See each case for measured status.

Case 3 preparation acquired one pinned small F16 GGUF with a read-only tensor inventory in `results/case3-source.json`; see docs/03-modify-layers.md. The no-change GGUFWriter control is byte-identical to the source, verified in `results/case3-rewrite.json`; sequential original/control CPU inference produced identical capped smoke outputs in `results/case3-control.json`; the predeclared norm edit is verified in `results/case3-perturb.json`, and `results/case3-edited.json` confirms safe loading with unchanged fixed-prompt text.

Case 4 has a verified 31-block tiny GGUF created by copying the original final block; `results/case4-expand.json` preserves its map and hashes. The guarded CPU comparison in `results/case4-expanded.json` confirms loading and changed repetitive text, without useful shot structure or established quality gain. See docs/04-add-layers.md.

Case 5 has a verified 29-block tiny F16 artifact with predetermined interior block 15 removed and later blocks renumbered; `results/case5-remove.json` preserves maps, hashes and rewrite measurements. The guarded comparison in `results/case5-pruned.json` confirms loading and changed repetitive text with no SHOT headings; one capped scene establishes no quality or speed advantage. See docs/05-remove-layers.md.

Case 6 saved five-position reference logits from the unchanged tiny F16 GGUF in `results/case6-reference*`, with a predeclared numerical criterion. The independent full tiny-model NumPy forward comparison in `results/case6-forward*` matched five argmax tokens but failed the predeclared numerical tolerance (maximum absolute error 0.14151). A separate F16-input-rounding diagnostic also failed (maximum error 0.15122); operand rounding alone does not resolve the mismatch. The reference model was closed; further numerical diagnosis and training remain pending. See docs/06-transformer-plan.md.
