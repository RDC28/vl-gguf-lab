# Dataset

`dataset.json` is committed in this public repository. It contains 150 text-only pairs copied unchanged from the supplied dataset: `user's scene description` is the input and `shot breakdown` is the reference. References are never included in evaluation prompts. There are no images, so no experiment establishes visual understanding.

SHA256: `073bd194bdb79e3ecfcfad51e09f23510ca697431475221b785c369619d7b8cf`.

`src/freeze_split.py:build()` is the authoritative input-only grouping and split procedure. It freezes 120 training, 15 validation and 15 test scenes, conservatively grouping related scenes and forcing previously explored inputs into training. Exact strings are unique, but characters and story worlds cross splits. This reduces leakage without certifying semantic independence. The validation set is reserved; the default fixed training budget does not tune against either validation or test references.

All references have four numbered SHOT headings. Structure scoring uses transparent regex proxies for time, lens, framing and movement, alongside shot count. It does not assess dramatic intent, continuity fidelity or cinematic quality. Inspect generated outputs and reference parses before interpreting scores.
