# Dataset

Source: `/Users/rohit/Downloads/dataset.json`.
Local ignored copy: `data/dataset.json` (454,187 bytes, 150 objects).
Each object has exactly two string fields: `user's scene description` and `shot breakdown`.
The first is the input; the second is used only for comparison after generation, never in the model prompt.
The committed run metadata records the dataset SHA256. Raw sample results include the sampled inputs and references for auditability.

No training split has been made. Before any training, freeze disjoint train/validation/test sets and check scene duplication. The present sample is exploratory, not a held-out benchmark for a future trained model.

SHA256: `073bd194bdb79e3ecfcfad51e09f23510ca697431475221b785c369619d7b8cf`.
All 150 inputs and all 150 references are unique as exact strings. This does not rule out semantic duplicates. Input word counts range from 13 to 34 (mean 17.87); reference word counts range from 413 to 497 (mean 434.57). All references have four numbered SHOT headings. Word tokenization uses the baseline script's Unicode word regex. See `results/dataset-summary.json` for the raw statistics.
