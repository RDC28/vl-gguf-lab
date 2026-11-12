# Case 4: add blocks and score every variant

Run [04_add_layers.ipynb](../notebooks/04_add_layers.ipynb). Use the same pinned F16 SmolLM2-135M control, frozen test indexes, scores and generation settings as case 3. This tests adding decoder blocks in GGUF; it does not modify the 7B VL baseline or its vision tower.

For each predeclared layer 0, 15 and 29, insert one exact copy immediately after that original block, renumber later blocks and update `llama.block_count`. Each variant has one extra block and starts from the unchanged source. Duplication is not identity initialization: residual updates occur twice, so behavior preservation and quality improvement are not assumed. No adaptation is performed in this case, isolating the effect of the edit.

The shared writer verifies every output tensor against its source mapping and saves source/output hashes, the complete block map and tensor inventory. The unchanged model and a no-change rewrite are evaluated first; control text must match before the sweep continues. Every variant runs all 15 test inputs and saves overlap, shot structure, truncation, latency and paired score changes against the unedited baseline. Report all variants rather than selecting a test-set winner.

All scene outputs and metrics save incrementally to the case `04` Drive run; reconnect with the same config and saved folder name. Edited weights are temporary and reproducible from the audit map. The CPU smoke path inserts a real block at one position and generates one short scene answer.

| Full Colab variant | Overlap / structure | Truncation / latency | Paired change |
| --- | --- | --- | --- |
| Original / no-change control | PENDING COLAB RUN | PENDING COLAB RUN | PENDING COLAB RUN |
| Duplicate layer 0 | PENDING COLAB RUN | PENDING COLAB RUN | PENDING COLAB RUN |
| Duplicate layer 15 | PENDING COLAB RUN | PENDING COLAB RUN | PENDING COLAB RUN |
| Duplicate layer 29 | PENDING COLAB RUN | PENDING COLAB RUN | PENDING COLAB RUN |

Historical Mac block-copy and single-prompt evidence is preserved under `results/04/local-2026-11-11/`. Changed repetitive output was not a measured quality improvement.
