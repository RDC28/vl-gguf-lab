# Case 1: complete VL baseline on Colab

Run [01_baseline.ipynb](../notebooks/01_baseline.ipynb). Setup clones the public repository anonymously over HTTPS, installs the pinned portable requirements and official CUDA llama.cpp wheel, and checks CUDA. Configuration selects the pinned model tier, generation cap and Drive root. Shared implementation: `src/lab_experiments.py`, `src/lab_models.py`, `src/lab_metrics.py` and `src/lab_storage.py`.

The full run uses all 150 committed text-only inputs, greedy generation, an 8,192-token context and a 4,096-token completion cap. Default weights are Qwen2.5-VL-7B Q4_K_M; Q8_0 and BF16 tiers target L4 and A100. The vision projector is not loaded because there are no images. The baseline estimates descriptive performance over the full dataset, not held-out training generalization.

Each saved JSONL row includes dataset index, input, reference, generated text, finish reason, token usage, elapsed time, overlap scores and parsed shot fields. References enter scoring only. A `length` finish reason counts as truncation. Output caps are explicit; increasing a cap requires a new run, not mixing settings during resume.

Unigram and bigram F1 use Unicode word tokens with multiplicity. Shot-count agreement is `max(0, 1 - abs(predicted-reference)/max(1, reference))`. Field recall is the mean capped predicted/reference count for time, lens, framing and movement. Structure score averages count agreement and field recall. Regex field matches are transparent proxies and can be gamed by repeated phrases; they are not semantic continuity or artistic quality measures. Raw parses are saved for inspection.

Drive rows and metrics update after every scene under `MyDrive/vl-gguf-lab/results/01/runs/<timestamp-id>/`. Set `RESUME` to the printed folder name and keep the same config to continue. Outside Colab, storage falls back to local `results/01/runs/`. The final notebook cell prints a measured summary and saves three plain charts.

| Full Colab measure | Result |
| --- | --- |
| Completed scenes / 150 | PENDING COLAB RUN |
| Model tier and GPU | PENDING COLAB RUN |
| Unigram / bigram F1 | PENDING COLAB RUN |
| Shot count / field recall / structure | PENDING COLAB RUN |
| Truncation rate / elapsed time | PENDING COLAB RUN |

Historical 8 GB Mac evidence is preserved unchanged in `results/01/local-2026-11-11/`. It covered only a small sample and suffered capped outputs. It is not evidence that the full Colab method has already run. CPU smoke mode deliberately uses one scene, 24 completion tokens and the small text-model control; its numbers must not fill this table.
