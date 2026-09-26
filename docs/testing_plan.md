# Testing Plan

**Status: Pass 1–4 tests implemented; later-pass additions remain planned.** The thirteen current test files contain 190 passing cases, including parameterized cases. The installed runner is `pytest`. The connected synthetic model pipeline now has unit, regression, integration, and command-line checks. See the [Pass 3 walkthrough](pass3_walkthrough.md) to understand their outputs.

From the repository root, run `venv/bin/python -m pytest -q`. Setup and fixture inspection commands are in the [README](../README.md).

## What are we testing?

A **software test** checks whether our code behaves as intended. An **ML evaluation** checks how accurately our approach matches unseen businesses. We need both.

- **Unit test:** checks one small behavior, such as reading a literal `NA` without changing it to a missing value.
- **Regression test:** protects a behavior from breaking again after future edits. It can be a unit or integration test. The `NA` case is our first concrete example.
- **Integration test:** checks that several parts work together, such as retrieval, feature generation, and output writing.
- **Smoke test:** runs a small example to catch obvious execution failures; it does not prove accuracy or scalability.
- **ML evaluation:** scores predictions against held-out ground truth. Candidate recall, macro F0.5, runtime, and memory belong in experiment reports.

We will not create tests just to check that a function was called or that a documentation sentence exists. Tests should protect data, mathematics, boundaries between components, or externally required output behavior.

## Fixtures: tiny examples with known answers

A **fixture** is a small, controlled input used repeatedly in tests. The first fixtures now live under `tests/fixtures/dataset/` and contain invented records and IDs, not the full challenge dataset. They cover text loading, blank fields, schemas, and inspector execution. Fixture TSVs contain real tabs and UTF-8 text.

The following list covers the overall test design. Extend the first fixtures with the later retrieval/scoring/model cases as their passes begin:

- Names containing literal `NA`, ordinary names, empty names, and whitespace-only names. Keep separate expected raw and derived values.
- Empty addresses, commas inside addresses, French accents, and Indian scripts.
- A short name `NA` and an invented expanded name sharing a useful address, plus an unrelated `NA` elsewhere. This tests information preservation and candidate behavior, not a rule that all `NA` names match.
- Zero, one, and multiple true matches; include an entity with more than five true matches to expose accidental fixed-five assumptions.
- Similar names at different addresses, and different names sharing an address. A shared address is evidence, not a guaranteed match.
- Candidate duplicates, empty candidate lists, and a true match deliberately omitted from retrieval.
- Malformed headers, missing required fields, duplicate IDs, unknown target IDs, and incorrect source prefixes in separate invalid fixtures.
- Country labels including France and a synthetic unseen label, so code cannot silently restrict itself to US and India.

Fixtures are for software correctness. Real training/development samples are for learning and accuracy measurement. Test data has no public ground truth, so a test-file smoke run cannot provide an accuracy score.

## Current and planned files

All rows except `test_resume.py` exist now. Resume testing is planned for Pass 6. Model tests will expand when real supervised training is connected.

| Planned file | First pass | Important behavior to protect |
|---|---|---|
| `tests/test_data_loading.py` — implemented | 1 | Exact schemas, string IDs, literal `NA`, blank-field handling, UTF-8, TSV read/write preservation, malformed-input errors, chunk limits, inspection counts |
| `tests/test_entrypoints.py` — implemented | 1 | Importing a module does not scan data, train, or write outputs; configured paths work in a temporary workspace; report/sample scope is explicit |
| `tests/test_splits.py` — implemented | 2 | Duplicate/overlapping IDs, complete coverage, aligned unchanged labels, target existence/ownership, reproducible stratification, rare-group handling, output immutability |
| `tests/test_scoring.py` — implemented | 2 | Hand-calculated scores, singleton behavior, macro averaging, retrieval misses, complete entity coverage, malformed lists, candidate subsets |
| `tests/test_normalization.py` — implemented | 3 | Raw values retained, Unicode preserved, blanks explicit, abbreviations not erased; repeat normalization gives the same result |
| `tests/test_candidates.py` — implemented | 3–4 | Designed fixture matches retrievable, duplicate removal, valid target IDs, deterministic ties/limits, empty results, country support |
| `tests/test_pair_labels.py` — implemented | 3 | Retrieved pairs agree with ground truth; known positives cannot become negatives; no duplicated contradictory pairs |
| `tests/test_features.py` — implemented | 3 | Same feature values and column order in training/inference; declared missing-value behavior; no accidental text `nan` |
| `tests/test_model_io.py` — implemented | 3 | Saved/reloaded model and settings reproduce scores within a documented tolerance; incompatible feature schema fails clearly |
| `tests/test_submission.py` — implemented | 3 | Exact headers, one row per Source 1 entity, empty lists, unique existing S2/S3 IDs, matches contained in scored candidates |
| `tests/test_pipeline_smoke.py` — implemented | 3 | Tiny input completes the connected workflow and produces both valid outputs in a temporary directory |
| `tests/test_retrieval_index.py` — implemented | 4 | Full source ingestion, SQLite reuse, raw text/Unicode, deterministic ties and caps, empty pools, fingerprint checks, restricted pilot labels |
| `tests/test_retrieval_evaluation.py` — implemented | 4 | Frozen development selection, full target coverage, no answer-key access during retrieval, exact exported candidates, cap losses, input mismatch rejection, failure cleanup |
| `tests/test_resume.py` | 6 | Interrupted/resumed processing gives the same records as uninterrupted processing, with no duplicate or missing rows |

Extend existing tests when a pass changes behavior. Avoid creating an additional file for every minor function.

## The first regression: literal `NA`

The [dataset findings](dataset_insights.md) document raw `NA` names and their labeled counterparts. In the installed Pandas version, default missing-value parsing changes `NA` to a missing value. That is a reproducible parser behavior, but it is not a dataset definition.

The implemented regression distinguishes:

| Input field | Expected raw value after loading | Derived blank indicator |
|---|---|---|
| `NA` | `NA` | False |
| Empty field | Empty string | True |
| Spaces only | Original spaces | True |
| `National Alliance` | `National Alliance` | False |

The reader validates raw TSV fields with Python's standard CSV parser, then constructs Pandas tables from the preserved strings. It computes blank indicators separately. Pass 3 adds separate normalized copies (Unicode NFKC, casefolding, collapsed whitespace), while raw values remain available. Do not automatically expand `NA`, discard its row, or force a match from the token.

## Scorer examples: answers calculated independently

These cases are now implemented. The CLI fixture under `tests/fixtures/scoring/` produces a macro score of 2/3 from three business scores of 1, 0, and 1. See the [Pass 2 record](pass2_verification.md) and README for the runnable example.

For a business with true matches, use `1.25 × TP / (1.25 × TP + FP + 0.25 × FN)`. Here `TP` is a correct predicted link, `FP` is a wrong predicted link, and `FN` is a missed true link. `A`, `B`, and `C` below represent distinct target IDs.

| True targets | Predicted targets | Expected business F0.5 | What it protects |
|---|---|---|---|
| Empty | Empty | 1 | Correct no-match prediction gets credit |
| Empty | A | 0 | A false match on a singleton gets no credit |
| A | Empty | 0 | A missed match is counted |
| A, B | A, B | 1 | Exact result |
| A, B | A | 5/6 ≈ 0.833333 | Partial recall with no false matches |
| A, B | A, B, C | 5/7 ≈ 0.714286 | Extra wrong target reduces precision |
| A, B | A, C | 1/2 | Both a false positive and a false negative |

Additional requirements:

- Business scores of 1, 0, and 1 must average to **2/3**, regardless of their different numbers of true links. This protects macro averaging.
- If truth is `{A, B}` but retrieval finds only `{A}`, a model that accepts `A` scores **5/6**, not 1. The missing candidate `B` remains a false negative.
- An explicitly empty prediction row is valid. A missing Source 1 row is a pipeline contract error; the scorer must not silently shrink its denominator.
- Reject duplicate/malformed prediction rows or validate them before scoring. Set mathematics must not hide an invalid submission.
- Compute expected results independently; do not copy the scorer's implementation into the test to produce expected answers.

## Connected fixture checks from Pass 3

The new fixture lives under `tests/fixtures/pipeline/`; it is separate from the original reader fixture and the hand-calculated scoring fixture. The runner uses 10 invented training businesses and 8 separate prediction businesses. An answer key is available only because we invented these records.

- A pair's label must agree with its training business's answer key. Known positives cannot be labeled negative by a random sampler, and retrieval misses cannot be inserted from truth.
- Features are shared by training/inference. Independent simple examples assert similarity values; comparing two blank fields produces zero similarity, with blank flags set.
- Native model save/reload must preserve scores within absolute tolerance 1e-12, with zero relative tolerance. A reordered feature schema, incompatible normalization/retrieval settings, or changed model file fails.
- Controlled scores test inclusive `>= 0.5` thresholding, including exactly 0.5 and just below it. These tests support zero, one, and more than five accepted targets without demanding that a learned fixture model produce those cases.
- Every exported candidate pair must be exactly a pair that was scored. Missing query rows, duplicate IDs/pairs, nonexistent targets, invalid scores, and predictions outside candidates fail. The supplied validator also receives valid and invalid files with target-ID checking enabled.
- The connected run preserves all query rows, including an entirely empty candidate batch. An intentional retrieval miss remains a false negative in the official score.
- Changing only the evaluation answer key must change the grade without changing model training or predictions. Saved-model inference is also tested with truth-reading and training functions forbidden.
- Existing outputs/input directories are protected; failed runs leave no published partial results. Oversized fixture inputs fail instead of silently becoming misleading truncated experiments.

These tests use synthetic files and temporary output directories. They do not use the reserved real evaluation set or assert a high learned fixture score. [Recorded verification](pass3_verification.md) includes the separate runnable demonstration.

## Full-pool retrieval checks from Pass 4

Software tests create tiny SQLite indexes in temporary directories. They verify that index construction needs only Source 2/3, retains every supplied target, rejects duplicate IDs, and preserves raw `NA`/Unicode. Retrieval tests cover differing names/scripts with useful address evidence, reordered name words, arbitrary country labels, empty queries/pools, frequent-word filtering, deterministic ties and caps above five, and independence from input/query order and chunk size.

The evaluation tests use a small synthetic experiment manifest with the same contracts as Pass 2. They reject changed source/sample hashes, incomplete target coverage, and restricted pilot indexes. They check that ground truth is first loaded after retrieval, changing only answers cannot change candidates, retrieval misses survive scoring, and no partial results are published on failure. Production selection is fixed to the saved development sample; the command does not offer reserved evaluation queries as a selector.

The separate real run searches 1,000 development businesses against 10,320,219 targets. It measures recall, candidate counts, resource costs, and errors. This job is not part of `pytest`, and its accuracy numbers are not hard-coded software assertions. See the [Pass 4 verification record](pass4_verification.md).

## What should not become a brittle assertion

- Do not assert a high F0.5 from a model trained and scored on the same tiny data.
- Do not require a learned model to accept every illustrative fixture pair. Use controlled scores to test threshold/output logic; evaluate learned decisions on development data.
- Do not hard-code a particular candidate ranking where equally scored results have no specified tie rule. Define the tie behavior first.
- Do not require identical training results across every library version or machine. Pin the environment and use tolerances where numerical variation is expected.
- Do not run all 24 million source records after every small code edit. Large-data integrity/resource checks are separate, explicit runs.

## When to run which checks

1. During a task, run the tests for the changed behavior and its dependent components.
2. Before completing a pass, run the accumulated fast regression suite and the small integration test when available.
3. When changing features, retrieval, or model settings, also run a fixed development experiment. Compare accuracy and cost with the saved baseline.
4. At the planned evaluation checkpoint, score untouched entities after settings are frozen; do not repeatedly use that set for tuning.
5. Before submission, run complete output checks and the supplied validator. Its default mode skips target-ID existence, and candidate-subset violations are warnings. Our project checks must still enforce these requirements.

Passing the tests means the defined behaviors hold for their cases. It does not mean the model is competitive or that the complete dataset has been exhaustively audited.
