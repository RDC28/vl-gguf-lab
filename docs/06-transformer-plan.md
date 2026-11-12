# Case 6: independent decoded transformer and training

Run [06_own_transformer.ipynb](../notebooks/06_own_transformer.ipynb). `src/lab_transformer.py` implements a trainable PyTorch decoder directly from pinned F16 SmolLM2-135M GGUF tensors. It implements embedding, RMSNorm, grouped-query causal attention, adjacent-pair rotary positions, SwiGLU, residual connections and tied output projection. Transformers supplies only the tokenizer for this case, not the decoder. Packed quantization and vision encoders are not implemented.

The implementation keeps decoded parameters in float32 and uses its own KV cache for generation. Training uses full-sequence causal attention, while cached inference attends to all prior keys. Local behavioral tests compare full-sequence math against the existing independent NumPy forward pass on a small architecture, compare cached/full logits and require finite gradients. These checks establish implementation properties, not exact agreement with llama.cpp's kernels on the production weights.

Before training, run the unchanged model through CPU llama.cpp and the independent decoder on the same short token sequence. Save both logit arrays, maximum and RMS error, violating-logit count and argmax agreement. Keep the original acceptance rule: every logit must satisfy `abs(custom-reference) <= 0.001 + 0.001*abs(reference)`. Argmax agreement alone does not pass. The historical NumPy run failed tolerance despite matching argmax, and rounding F16 inputs alone did not close the gap. Float32 PyTorch and CPU GGML use different operand rounding, reductions and kernel ordering; the new run measures the remaining discrepancy without asserting an unproven root cause or loosening the criterion.

If parity still fails, the notebook records that failure and proceeds with training explicitly labeled **exploratory, numerical equivalence unestablished**. This is a documented gap, not a validated exact decoder claim. Review saved logits and kernel semantics before treating it as an equivalent replacement for llama.cpp.

Evaluate all 15 frozen test scenes before training. Train all decoded parameters on the 120 training scenes for three seeded shuffled epochs using response-only cross entropy, AdamW at 0.0001, gradient clipping and a 2,048-token training cap. Then evaluate the same held-out scenes at identical greedy settings. Record supervised token counts, clipped-target rates, finite loss/gradients, per-step losses, before/after overlap and structure, truncation and paired changes. Full checkpoints include all trained parameters and optimizer/RNG state, about 1.6 GB. They stay on Colab local disk so repeated replacement does not fill a free Drive through Trash.

Every step is logged and every test answer is saved directly under the case `06` Drive run. Every 10 steps and at completion a full checkpoint is written locally; once per epoch and at completion a fixed-name float16 `weights-snapshot.pt` is written to Drive. Resume restores weights and optimizer from the local checkpoint when it survives, otherwise float16 weights from the Drive snapshot with a restarted optimizer and recorded restart step, never from an arbitrary later loss log. The CPU smoke path performs a real logit comparison, one optimizer step and one short before/after generation.

| Full Colab measure | Result |
| --- | --- |
| Strict parity / max error / RMS / violations | PENDING COLAB RUN |
| Argmax agreement | PENDING COLAB RUN |
| Completed training steps / loss / clipped targets | PENDING COLAB RUN |
| Before and after overlap / structure | PENDING COLAB RUN |
| Before and after truncation / paired changes | PENDING COLAB RUN |

Historical source excerpts, reference logits, NumPy outputs and rounding diagnostics remain untouched in `results/06/local-2026-11-11/`. Full Colab parity, training feasibility, GPU memory and held-out outcomes remain unmeasured until the notebook runs.
