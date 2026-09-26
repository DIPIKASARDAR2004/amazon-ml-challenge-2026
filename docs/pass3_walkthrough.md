# Pass 3 — What you should inspect

**You can now follow one business through the complete program.** It reads records, finds candidates, calculates numerical clues, uses a trained model, and writes final matches. Everything in this walkthrough uses invented, tiny data. The real challenge data and reserved evaluation businesses were not used.

The completed run is already under `artifacts/pipeline/pass3_v1/`. You can inspect it without running anything. Start with steps 1–3; the model code can wait until the outputs make sense.

## 1. Read the input and its answer key

Open [test_source1.tsv](../tests/fixtures/pipeline/test/test_source1.tsv). It contains **8 invented businesses**. Then open [test_source2.tsv](../tests/fixtures/pipeline/test/test_source2.tsv) and [test_source3.tsv](../tests/fixtures/pipeline/test/test_source3.tsv). Together they contain **10 possible target records**.

Find this example:

| Record ID | Name | Address |
|---|---|---|
| `S1-demo-many` | Silver Bakery | 40 Elm Lane |
| `S2-demo-many` | Silver Bakery | 40 Elm Lane |
| `S3-demo-many` | SILVER BAKERY, with extra spaces | 40 Elm Lane |
| `S2-demo-distractor` | Silver Bakery | 700 Faraway Street |

The first two target records are the intended matches. The distractor is a different invented business with the same name. This is why name equality alone cannot decide identity.

Open [expected_test_matches.tsv](../tests/fixtures/pipeline/expected_test_matches.tsv). Its row for `S1-demo-many` lists `S2-demo-many,S3-demo-many`. This is our **answer key**, written when designing the fixture. The model does not read it during prediction. The runner reads it afterward to grade the saved predictions. The real competition test data does not provide this file.

**Checkpoint:** You understand what one Source 1 row is asking and why its answer can contain several target IDs.

## 2. Compare candidates with accepted matches

Open these generated files side by side:

- [candidate_pairs.tsv](../artifacts/pipeline/pass3_v1/candidate_pairs.tsv): records selected for model scoring.
- [matching_results.tsv](../artifacts/pipeline/pass3_v1/matching_results.tsv): records accepted after model scoring.

In the completed run:

| Source 1 ID | Candidates | Final matches | What to notice |
|---|---|---|---|
| `S1-demo-many` | 3 | 2 | The same-name distractor was considered, then rejected |
| `S1-demo-one` | 2 | 1 | One target was accepted |
| `S1-demo-reject` | 4 | 0 | The model can reject every candidate |
| `S1-demo-empty` | 0 | 0 | The row remains present even without candidates |
| `S1-demo-missed` | 0 | 0 | An empty result can also be an error; inspect this in step 5 |

Both files have **8 data rows plus one header**. Count Source 1 rows, not comma-separated target IDs. Across all candidate lists there are **15 pairs**; across the final match lists there are **6 accepted pairs**.

Each row has two columns separated by a tab. Several IDs within the second column are separated by commas. An empty answer still has its Source 1 ID and the separating tab. These files share the competition's format, but their invented IDs are only for local checks.

**Checkpoint:** A candidate means “examine this pair.” A match means “the model's score passed the cutoff.” Every accepted match must have been a scored candidate.

## 3. Follow a decision in the pair trace

Open [scored_pairs.tsv](../artifacts/pipeline/pass3_v1/scored_pairs.tsv). It has one row per candidate pair: **15 rows**, not 8. Businesses with no candidates have no pair-trace rows; they still have rows in both output files.

Search for `S1-demo-many`. Focus on these columns first:

| Column | Meaning |
|---|---|
| `source1_entity_id`, `target_entity_id` | The two records being compared |
| `query_business_name`, `target_business_name` | Their original, unchanged names |
| `*_normalized` | Separate comparison copies with case/spacing/Unicode normalization |
| `name_ratio`, `address_ratio` | RapidFuzz similarity, scaled from 0 to 1 |
| `model_score` | LightGBM's learned match score |
| `accepted` | `1` when the score is at least 0.5, otherwise `0` |

The completed run contains:

| Pair | Name similarity | Address similarity | Model score, rounded | Accepted |
|---|---:|---:|---:|---:|
| `S1-demo-many` → `S2-demo-many` | 1.000 | 1.000 | 0.992 | 1 |
| `S1-demo-many` → `S3-demo-many` | 1.000 | 1.000 | 0.992 | 1 |
| `S1-demo-many` → `S2-demo-distractor` | 1.000 | 0.276 | 0.009 | 0 |

The model learned from the **separate training fixture**. We did not write a rule saying Silver Bakery must match these IDs. Feature generation does not use ID spelling or the evaluation answer key as clues.

The model score is distinct from either similarity value. Do not read 0.992 as proven “99.2% certainty”: this tiny demonstration has not been calibrated. The fixed cutoff of 0.5 was not tuned for competition performance.

Now search for `S1-demo-na` and `S3-demo-na`. The raw target name remains `NA`; its normalized copy is `na`; `target_name_blank` is 0. Name similarity is only about 0.222, while address similarity is 1. The candidate was retrieved through address evidence and accepted in this run. This demonstrates preservation of the text and the use of multiple clues. It does not establish a universal meaning or match rule for `NA`.

Two blank addresses produce similarity 0 and explicit blank indicators. They do not get similarity 1 simply because both are empty. Accents and Indian scripts remain in the raw fields; normalization does not translate names between scripts.

**Checkpoint:** You can explain where each value came from: text → numerical features → learned score → threshold decision → output row.

## 4. See what the model learned from

Open [training_pairs.tsv](../artifacts/pipeline/pass3_v1/training_pairs.tsv). It contains **17 retrieved pairs** from 10 different invented training businesses. Its `label` column comes from [train_ground_truth.tsv](../tests/fixtures/pipeline/train/train_ground_truth.tsv):

- `1`: this target appears in that business's known match list.
- `0`: this retrieved target does not appear in its known match list.

There are **9 positive labels and 8 negative labels**. Search for `S1-tr01`: a same-name/same-address pair has label 1, while its same-name/different-address distractor has label 0. Also inspect `S1-tr03`: one retrieved `NA` is positive and another is negative. `NA` alone does not determine the label.

The same feature function prepares both training and prediction pairs. During training, LightGBM sees those features and the labels. During prediction, it sees only features. We do not randomly assign negatives, insert missing positive candidates from the answer key, or fit the model on the eight prediction businesses.

Open [model/metadata.json](../artifacts/pipeline/pass3_v1/model/metadata.json) next. It records the ordered feature names, normalization settings, retrieval settings, threshold, seed, library versions, and fingerprint of the saved model. The model itself is in [model.txt](../artifacts/pipeline/pass3_v1/model/model.txt); you do not need to read its tree details yet.

There are two different cutoffs in the metadata:

- `minimum_similarity: 0.25`: candidate retrieval requires sufficient name **or** address TF-IDF similarity. TF-IDF represents character fragments as numerical vectors. This similarity is separate from RapidFuzz's feature values.
- `threshold: 0.5`: after feature calculation, the model score must reach this value for acceptance.

`top_k: 4` limits this demonstration to four candidates per business. It is configurable, and it is not a competition limit. Tests explicitly cover more than five candidates/matches. Retrieval sorts ties by target ID and does not require country equality.

**Checkpoint:** Training uses known answers to learn a model; prediction uses the saved model without answers. Settings are saved alongside the model so the two paths agree.

## 5. Inspect the deliberate failure and the report

Open [report.json](../artifacts/pipeline/pass3_v1/report.json). Read `scope`, `training`, `inference`, `reload`, `validation`, and `evaluation` first. You can skip the long file-hash sections initially.

| Report value | Recorded result | Meaning |
|---|---:|---|
| `scope` | `synthetic_fixture_only` | This run says nothing about real-data accuracy |
| `reload.maximum_absolute_difference` | 0.0 | Saving/reloading preserved scores; allowed difference was 1e-12 |
| `validation.check_ids` | true | Target IDs were checked against the fixture's actual target files |
| `evaluation.macro_f0_5` | 0.875 | Average of the eight fixture business scores |
| `evaluation.pair_diagnostics` | TP 6, FP 0, FN 1 | Six correct links accepted; one true link missed |
| `evaluation.candidate_diagnostics.pair_recall` | 6/7 ≈ 0.857 | Retrieval found six of the seven known true links |

Look up `S1-demo-missed` in the answer key. Its true target is `S3-demo-hidden`, but retrieval never found it. Their invented names use different scripts and have no useful textual overlap; both addresses are blank. Our character-based retriever cannot recognize their identity from this information.

The model never gets a pair to score for this business. Its empty prediction scores 0. The other seven business predictions score 1 in this run, so the average is **7/8 = 0.875**. The missed link remains a false negative; it is not removed from the answer key to improve the score.

The report's `oracle_macro_f0_5` is also 0.875: even a perfect classifier could not recover the missing candidate. This shows why Pass 4 focuses on retrieval before real model tuning. Fixture results are designed examples, not an estimate of leaderboard performance.

Open [validator.txt](../artifacts/pipeline/pass3_v1/validator.txt) to see the output checks. A validator checks IDs, rows, and formatting. It does not decide whether two businesses truly match.

## 6. Run it yourself when ready

From the repository root, run the complete fixture into a new directory:

```bash
venv/bin/python -m src.pipeline --output-dir artifacts/pipeline/my_first_run
```

If that directory already exists, use a new name such as `my_second_run`. The command finishes quickly on this workspace and prints a short result. Existing runs are protected so you can compare them. The program rejects files over 1,000 rows: this command is intentionally a small teaching runner.

For one readable test about zero/one/many matches and the threshold boundary:

```bash
venv/bin/python -m pytest -vv tests/test_submission.py::test_zero_one_many_and_threshold_boundary_survive_both_validators
```

That test uses controlled scores, including exactly 0.5 and a value just below it. It checks output behavior without relying on a learned model to produce a particular score.

Then run the integration checks, or the complete regression suite:

```bash
venv/bin/python -m pytest -v tests/test_pipeline_smoke.py
venv/bin/python -m pytest -q
```

Pass 3 finished with **165 passing cases**; after Pass 4 additions the complete suite has **190**. That is a count of software checks, not a percentage accuracy or a count of correctly matched real businesses. Tests include changing only the evaluation answer key and proving that predictions remain unchanged while the score changes.

No real-data model training or leaderboard upload is part of these commands. The older root `train_model.py` and `candidate_generation.py` are historical demos; use `src.pipeline` for this connected example.

## When you are ready to read code

Follow the data in this order: [normalization.py](../src/normalization.py) → [candidates.py](../src/candidates.py) → [pair_labels.py](../src/pair_labels.py) and [features.py](../src/features.py) → [model_io.py](../src/model_io.py) → [submission.py](../src/submission.py). [pipeline.py](../src/pipeline.py) connects them; [scoring.py](../src/scoring.py) grades the final files.

Pass 4 has now measured retrieval against the complete training target pool using fixed development queries; continue with its [inspection walkthrough](pass4_walkthrough.md). The next implementation pass is Pass 5, realistic classifier training and evaluation. The fixture's ease and score cannot establish real-data accuracy.
