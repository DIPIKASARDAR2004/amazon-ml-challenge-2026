# Pass 3 verification record

**Completed on 2026-09-26.** Pass 3 connects retrieval, supervised pair construction, features, LightGBM training, saved-model inference, official scoring, and both output files on invented data. Begin with the [inspection walkthrough](pass3_walkthrough.md) for a beginner explanation.

## Commands and results

From the repository root:

```bash
venv/bin/python -m pytest -q
venv/bin/python -m src.pipeline --output-dir artifacts/pipeline/pass3_v1
venv/bin/python utils/validate_submission.py \
    --matching artifacts/pipeline/pass3_v1/matching_results.tsv \
    --candidate artifacts/pipeline/pass3_v1/candidate_pairs.tsv \
    --test-dir tests/fixtures/pipeline/test --check-ids
```

The accumulated regression suite passed **165 cases in 8.85 seconds**. The standalone pipeline run is recorded at `2026-09-26T14:26:30.482945+00:00`. Its measured internal workflow took approximately **0.158 seconds**, excluding interpreter/module startup. These tiny-fixture timings are not scaling estimates.

The recorded environment is Linux, Python 3.14.4, with the already pinned packages: Pandas 3.0.6, NumPy 2.5.3, scikit-learn 1.9.1, RapidFuzz 3.14.6, and LightGBM 4.7.0. No dependency additions were needed.

The version directory already exists; use a new name to repeat the pipeline command. Regression tests create their own temporary directories. Generated artifacts are ignored by Git, so teammates recreate them from the trackable fixture and commands.

## Measured run

| Item | Result |
|---|---:|
| Invented training Source 1 businesses / target records | 10 / 14 |
| Retrieved training pairs | 17 |
| Positive / negative labels | 9 / 8 |
| Training true links | 9, all retrieved in this fixture |
| Separate invented prediction businesses / targets | 8 / 10 |
| Prediction candidate pairs, all model-scored | 15 |
| Matching/candidate output rows | 8 each |
| Empty match lists / empty candidate lists | 3 / 2 |
| Accepted links | 6 |
| Model score difference after reload | 0.0; tolerance 1e-12, zero relative tolerance |
| Official fixture macro F0.5 | 0.875 |
| TP / FP / FN | 6 / 0 / 1 |
| Candidate recall | 6/7 = 0.8571428571428571 |
| Project checks / supplied validator with ID checks | Passed / passed, no warnings |

`S1-demo-missed` deliberately has a true target absent from retrieval. It remains in both output files with an empty list and receives score 0; the other seven fixture businesses score 1. No truth was removed to improve the score. Fixture labels are read for grading only after predictions and validation complete.

## Implementation and contracts

- `src/normalization.py` retains raw fields and adds NFKC/casefold/whitespace-normalized copies. It retains accents and scripts, does not transliterate or expand abbreviations, and preserves `NA` as text.
- `src/candidates.py` fits separate name/address character TF-IDF representations on the appropriate target pool. The larger cosine similarity determines eligibility. Defaults are minimum 0.25, at most four candidates, ID ordering for ties, and no country filter. A pair found through both fields is emitted once. Querying is independent of batch/order on the tested cases.
- `src/pair_labels.py` assigns 1 exactly when a retrieved target is in that training business's truth; other retrieved pairs receive 0. Duplicate pairs fail, and missing positives are not inserted from labels.
- `src/features.py` provides one ordered set of ten features for training and inference: three text similarities, two nonblank exact-match flags, four blank flags, and country equality. Similarity/exactness is zero when a compared text is blank.
- `src/model_io.py` trains 40 LightGBM rounds with seed 42, one thread, and small-fixture settings. It saves a native text model and JSON metadata, then verifies feature/preprocessing contracts and the model file fingerprint on reload. These settings are demonstration settings, not a real-data baseline.
- `src/submission.py` requires complete query coverage, existing unique target IDs, the exact scored candidate pairs, finite scores, and final matches contained in candidates. Acceptance is inclusive: `score >= threshold`.
- `src/pipeline.py` runs the fixture with separate training/prediction IDs and target pools. It uses the reloaded model/settings for inference, saves inspectable training/scoring traces, reloads the outputs, runs both validators, and grades afterward. It rejects oversized fixtures and existing output directories; failed runs do not publish partial result directories.

The seven planned Pass 3 test files now exist. Tests cover raw/derived text, Unicode, blanks, retrieval ties/limits, more than five candidates/matches, pair-label correctness, independent feature values, model reload/schema rejection, threshold boundaries, output coverage/subsets, invalid files, all-empty prediction batches, and failure cleanup. Entrypoint tests cover imports without work and commands outside the repository. A regression changes only evaluation labels and confirms identical predictions, candidate lists, pair traces, and native model; the grade changes. No learned accuracy target is asserted.

## Artifacts

The completed directory contains `matching_results.tsv`, `candidate_pairs.tsv`, `training_pairs.tsv`, `scored_pairs.tsv`, `model/model.txt`, `model/metadata.json`, `validator.txt`, and `report.json`.

The report fingerprints every input and the other output files. For this recorded run:

| Output | SHA-256 |
|---|---|
| `matching_results.tsv` | `2c207c49c3011029609306d28997647f6cefa7f132d1daefb41be609b181017e` |
| `candidate_pairs.tsv` | `977b0d09ca65dc5c599199c6f67d95b92239144803232426d89c76d41c88f5a1` |
| `model/model.txt` | `e5a1e8597d223e513381e05ba632f9027831c1b3b0600dbbc10caf39a1955a60` |

Exact learned scores/hashes are evidence for this pinned local run; they are not promised across arbitrary library/platform changes. Reload equivalence is tested within the same environment with a stated tolerance.

## Boundaries and next work

No original dataset, real split, Pass 2 experiment manifest, or reserved evaluation business was used or modified. The old root demos remain historical examples; their random-negative training behavior is not used by the connected runner. The supplied validator was not changed. Nothing was uploaded or committed.

This runner holds small records/features in memory and scans the target vectors per query. It is capped at 1,000 rows per input file and does not solve full-pool indexing, resource scaling, threshold selection, or resumable inference. Its saved metadata records retrieval settings; its tiny target index is rebuilt from the appropriate targets during prediction. The fixture score provides no reliable accuracy estimate for the real data.

At the end of this historical run, Pass 4 had not started. Its later [verification record](pass4_verification.md) now records full-pool retrieval on the saved development queries. Pass 5 introduces realistic supervised training and honest evaluation; see [brain.md](../brain.md) for current status.
