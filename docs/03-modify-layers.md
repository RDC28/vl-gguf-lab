# Case 3: scored GGUF norm-edit sweep

Run [03_modify_layers.ipynb](../notebooks/03_modify_layers.ipynb). This controlled edit experiment uses the pinned `bartowski/SmolLM2-135M-Instruct-GGUF/SmolLM2-135M-Instruct-f16.gguf`, not the quantized VL model from case 1. `src/lab_edits.py` accepts unquantized F16/F32 Llama-family tensors only. General editing of packed quantization or arbitrary VL architectures is outside this method.

First evaluate the unchanged model on the frozen 15 test scenes. Then perform a no-change rewrite and require identical greedy text on the same scenes before proceeding. The predeclared sweep multiplies `blk.<layer>.attn_norm.weight` by 0.9, 1.01 and 1.1, separately at layers 0, 15 and 29. These nine independent variants always start from the original source. They do not accumulate edits.

The writer preserves unrelated tensor bytes and metadata, opens each output again, and verifies every tensor hash and block count. The run records source/output hashes, tensor hashes and the exact block map. Temporary edited model files are discarded after evaluation; they can be rebuilt from the pinned source and saved edit description.

Every variant uses identical held-out scene indexes and decoding settings, with per-scene text overlap, parsed shot count/fields, finish reason, truncation and timings. Each receives a paired comparison to the original baseline and a descriptive scene-bootstrap interval. Test scores are reported for all variants; do not choose the winner on test and claim an independently validated improvement. Norm scaling can damage quality, and text changing alone is not evidence of improvement.

Drive storage is incremental under case `03`, with the shared resume contract in the README. CPU smoke mode runs one held-out scene, one scale and one layer, still performing a real rewrite and inference.

| Full Colab variant | Overlap / structure | Truncation | Paired change |
| --- | --- | --- | --- |
| Original | PENDING COLAB RUN | PENDING COLAB RUN | Reference |
| No-change rewrite | PENDING COLAB RUN | PENDING COLAB RUN | PENDING COLAB RUN |
| Each of nine norm variants | PENDING COLAB RUN | PENDING COLAB RUN | PENDING COLAB RUN |

Historical Mac source inventory, rewrite and one-prompt edit evidence remains in `results/03/local-2026-11-11/`. It established loading behavior only, not a quality gain.
