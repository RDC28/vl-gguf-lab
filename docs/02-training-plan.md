# Case 2: train and re-test on a frozen split

Run [02_train_retest.ipynb](../notebooks/02_train_retest.ipynb). This is a text-model training comparison, separate from the VL GGUF baseline. The default model is pinned `HuggingFaceTB/SmolLM2-1.7B-Instruct`, loaded in FP16 on CUDA with FP32 LoRA parameters. Query/value projections receive rank-16 LoRA adapters with alpha 32 and no dropout. The small CPU path uses the pinned 135M model and one optimizer step.

`src/freeze_split.py:build()` freezes the committed dataset hash and 120/15/15 train/validation/test partition using input-only manual story groups, exact duplicate detection and lexical similarity. Prior exploratory inputs are forced into training. The same test indexes are used in cases 2-6. Characters and story worlds still cross splits, so this is a small held-out test with limited semantic independence. The validation set remains reserved; the fixed-budget default does not tune on validation or test.

The notebook evaluates the untrained adapter model on all 15 test scenes, trains on the 120 training inputs for three seeded shuffled epochs, and re-evaluates on exactly the same 15 scenes. No test reference appears in the training schedule. The answer is appended to the chat-formatted prompt, with prompt positions masked from next-token cross entropy. AdamW uses learning rate 0.0001 and gradient clipping at 1.0. The training sequence cap is 2,048; clipped target rates and supervised token counts are saved so a partial-response training run cannot be mistaken for full-target training.

Evaluation uses the same greedy decoding, system instruction, 8,192-token context and 4,096-token output cap before and after training. Each phase reports overlap, shot structure and truncation. Paired per-scene score changes and descriptive bootstrap intervals are saved. Improvements in these proxies do not establish better cinematic quality, and repeated templates restrict generalization claims.

Step losses and gradient norms save incrementally. Every 10 steps and at completion, `checkpoint.pt` stores trainable weights, optimizer state, CPU/CUDA RNG states and the completed step directly in the Drive run. Resume restores the checkpoint and replays any later uncheckpointed steps, discarding their stale log tail. A completed run can be resumed without retraining or regenerating finished scenes. Do not change configuration to extend a resumed run; create a new run instead.

| Full Colab measure | Before | After |
| --- | --- | --- |
| Held-out scenes | PENDING COLAB RUN | PENDING COLAB RUN |
| Unigram / bigram F1 | PENDING COLAB RUN | PENDING COLAB RUN |
| Shot count / field recall / structure | PENDING COLAB RUN | PENDING COLAB RUN |
| Truncation rate | PENDING COLAB RUN | PENDING COLAB RUN |

Training steps, target clipping, loss trajectory and paired changes: **PENDING COLAB RUN**. Historical Mac MLX smoke evidence remains in `results/02/local-2026-11-11/`; it is a different tiny training protocol and is not the result of this notebook.
