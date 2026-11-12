# Case 5: remove blocks and measure the tradeoff

Run [05_remove_layers.ipynb](../notebooks/05_remove_layers.ipynb). This uses the pinned F16 SmolLM2-135M control and the same frozen held-out scenes as cases 2-4 and 6. It does not establish pruning behavior for the larger VL baseline.

Create three independent variants by removing original block 0, 15 or 29. Renumber all later blocks and reduce `llama.block_count` by one. Selection is predeclared, not based on held-out quality or activation inspection. The shared writer verifies retained tensor hashes and saves the original-to-output block map and file hashes. Source weights remain unchanged.

Evaluate the original and no-change rewrite before testing removals. Require matching greedy control output, then score every variant on all 15 test scenes with identical generation settings. Save word overlap, shot count, field recall, combined structure, truncation and elapsed time. Paired deltas and descriptive bootstrap intervals accompany every variant. Single-scene timing, a text change or a smaller file cannot establish a useful quality/speed tradeoff; compare repeated full runs on the same GPU before claiming a speed gain.

Case `05` run folders support incremental per-scene saves and provenance-checked Drive resume. CPU smoke mode still rewrites and loads a real pruned model, but only evaluates one short answer.

| Full Colab variant | Overlap / structure | Truncation / latency | Paired change |
| --- | --- | --- | --- |
| Original / no-change control | PENDING COLAB RUN | PENDING COLAB RUN | PENDING COLAB RUN |
| Remove layer 0 | PENDING COLAB RUN | PENDING COLAB RUN | PENDING COLAB RUN |
| Remove layer 15 | PENDING COLAB RUN | PENDING COLAB RUN | PENDING COLAB RUN |
| Remove layer 29 | PENDING COLAB RUN | PENDING COLAB RUN | PENDING COLAB RUN |

Historical Mac removal and one-prompt evidence remains in `results/05/local-2026-11-11/`. It did not establish quality or performance benefits.
