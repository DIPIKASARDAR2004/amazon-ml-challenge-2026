# Synthetic connected-pipeline fixture

All records, IDs, and labels here are invented. They are not competition data.

- `train/` contains 10 Source 1 businesses, 14 target records, and training labels.
- `test/` contains 8 different Source 1 businesses and 10 different targets.
- `expected_test_matches.tsv` is the invented prediction answer key, used only for grading after inference. Real competition test data has no public equivalent.

Cases include zero/one/many matches, a same-name distractor at another address, a shared-address nonmatch, `NA` with both positive and negative training examples, a changed name, blank fields, accents, Telugu, and a synthetic country label. `S1-demo-missed` intentionally cannot retrieve `S3-demo-hidden` through textual overlap: the missed true link must still count against the score.

See the [walkthrough](../../../docs/pass3_walkthrough.md) and run `venv/bin/python -m src.pipeline --output-dir artifacts/pipeline/my_first_run` from the repository root. Do not upload these invented predictions to the challenge portal.
