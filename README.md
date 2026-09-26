# Amazon ML Challenge 2026 — Business Entity Resolution

For each Source 1 business, find its matching records in Sources 2 and 3. Start with the [beginner team guide](docs/team_guide.md), then the [implementation plan](implementation_plan.md).

**Passes 1–4 are complete, with 190 passing regression cases.** We have reliable data reading, audited experiment splits, the official scorer, a connected **tiny synthetic pipeline**, and disk-backed retrieval over the complete real training target pool with a fixed development benchmark. Real classifier accuracy and full test inference remain later work.

**Start here:** [project memory](brain.md), [Pass 4 inspection walkthrough](docs/pass4_walkthrough.md), and [Pass 4 verification](docs/pass4_verification.md). The [Pass 3 walkthrough](docs/pass3_walkthrough.md) still explains the complete synthetic model pipeline.

## Setup

Run the commands below from the repository root, the directory containing this README. The verified environment is Linux with **CPython 3.14.4**. The requirement files pin the installed runtime and test dependencies. Other Python/platform combinations have not been verified.

On a new machine, with Python 3.14 installed:

```bash
python3.14 -m venv venv
venv/bin/python -m pip install -r requirements-dev.txt
```

On this workspace the environment and pytest are already installed. You can start with the test command. No environment activation is needed when using `venv/bin/python` explicitly. Runtime-only installs can use `requirements.txt`; tests also require `requirements-dev.txt`.

## 1. Run the regression tests

```bash
venv/bin/python -m pytest -q
```

All tests should pass. They use tiny synthetic files and temporary directories, so the challenge dataset is not required. Some train a tiny model on invented records; none train on the challenge data or contact external services.

Run one regression when investigating a specific issue:

```bash
venv/bin/python -m pytest -q tests/test_data_loading.py::test_na_is_text_and_blanks_are_separate
```

See the [testing plan](docs/testing_plan.md) for what is covered and what belongs to later passes.

## 2. Inspect the tiny fixture first

```bash
venv/bin/python -m src.inspect_data \
    --dataset-dir tests/fixtures/dataset \
    --full --chunk-size 2 \
    --report artifacts/inspection/fixture.json
```

Here `--full` means all rows of the tiny fixture, not the challenge dataset. Check that `train_source1.tsv` reports **5 records, 2 blank names, and 1 literal `NA` name**. The fixture ground truth has **5 rows and 3 businesses with no matches**. The JSON report also records countries, timestamps, environment, and memory measurements.

This fixture deliberately includes blank fields, literal tokens, commas, a quoted tab, accented text, an Indian script, and an unfamiliar country label. These are software checks, not realistic country distributions or ML accuracy measurements.

## 3. Inspect a small sample of the real data

Keep the supplied files under this layout, or point `--dataset-dir` at an equivalent directory:

```text
dataset/
  train/
    train_source1.tsv
    train_source2.tsv
    train_source3.tsv
    train_ground_truth.tsv
  test/
    test_source1.tsv
    test_source2.tsv
    test_source3.tsv
```

```bash
venv/bin/python -m src.inspect_data \
    --sample-rows 1000 \
    --report artifacts/inspection/sample.json
```

The default is also 1,000 rows per file. This is a first-row sample, not a random or representative sample. The report marks it as `sample` and records the limit. It might contain no literal `NA` records even though the complete files contain 120.

## 4. Reproduce the full inspection

```bash
venv/bin/python -m src.inspect_data \
    --full --chunk-size 50000 \
    --report artifacts/inspection/full.json
```

This explicitly reads all seven original files. It does not train a model, rewrite the dataset, or inspect derived split files. Source-file counts should match [dataset findings](docs/dataset_insights.md): **24,229,173 source records**, **0 blank names**, **120 literal `NA` names**, and **610,389 blank addresses**. Training ground truth contains **2,206,821 rows** and **123,247 empty match lists**.

`--chunk-size` limits rows in each Pandas table; it is not an exact RAM limit. Smaller chunks can reduce memory use at some runtime cost. Memory snapshots describe the host environment and may not reflect a container limit. Unsupported memory measurements are recorded as unavailable instead of causing the inspection to fail.

Reports are optional; omitting `--report` prints the summary only. Reports must be written outside the selected dataset directory. `artifacts/` is ignored by Git. The [inspection verification record](inspection_report.md) records the completed checks for this pass.

## 5. Run the Pass 2 scoring example

```bash
venv/bin/python -m src.scoring \
    --truth tests/fixtures/scoring/truth.tsv \
    --predictions tests/fixtures/scoring/predictions.tsv \
    --candidates tests/fixtures/scoring/candidates.tsv \
    --source1 tests/fixtures/scoring/source1.tsv \
    --report artifacts/evaluation/hand_checked_score.json
```

Expected **`macro_f0_5`: 0.6666666666666666**, or 2/3. The three invented businesses score 1, 0, and 1. Their pooled pair recall is 0.8, which is a different diagnostic. This verifies the mathematics; it is not a model or leaderboard score.

The scorer requires exactly one prediction row per selected business, including empty predictions. If candidates are supplied, their row coverage must also be exact and every prediction must appear in its candidate list. Missed retrieval links remain false negatives. Duplicate rows/targets and malformed IDs fail before set conversion can hide them.

JSON `null` for a diagnostic means its denominator is zero, for example precision when no links are predicted. The official macro score still uses the specified singleton rules. `oracle_macro_f0_5` is the best possible score if a perfect classifier selected only the true links available in those candidates; it is not achieved model performance.

## 6. Understand the saved experiment IDs

The current workspace has an audited experiment under `artifacts/evaluation/pass2_v1/`. It preserves the existing training partition and divides the old validation partition into development and evaluation, using seed 42 and country/no-match strata. A stratum is a group such as “India, no true matches.”

| File | Use |
|---|---|
| `train_ids.tsv` | Complete existing training businesses |
| `development_ids.tsv` | Businesses available for choosing retrieval/model settings |
| `evaluation_ids.tsv` | Reserved businesses for the final estimate after settings are frozen |
| `sample_train_ids.tsv` | Up to 5,000 training businesses for small experiments |
| `sample_development_ids.tsv` | Up to 1,000 development businesses |
| `sample_evaluation_ids.tsv` | Up to 1,000 reserved evaluation businesses |
| `manifest.json` | Seed, selection rules, counts, strata, and input/output file hashes |
| `audit.json` | Full audit checks, counts, timing, memory, and limitations |

The ID files contain only the `source1_entity_id` column. They select queries; they do not select target candidates or alter the supplied labels. All training Source 2/3 records remain the target pool. Do not feed evaluation businesses into fitting or setting selection, even if their answers are locally available.

To reproduce the audit and IDs, choose a **new** output directory:

```bash
venv/bin/python -m src.splits \
    --output-dir artifacts/evaluation/pass2_recheck \
    --seed 42 \
    --development-fraction 0.5
```

This is a full-data job, unlike the quick regression suite. It reads the original training files and the existing `train/*_sampled.tsv` / `val/val_*.tsv` files. It checks uniqueness, every training label's target existence, disjoint/complete partitions, unchanged Source 1 fields, and unchanged label sets. A target linked to multiple reference businesses is also rejected to prevent label overlap between partitions.

Temporary SQLite indexes use disk space beside the output directory and are deleted afterward. Source files and the existing split are not rewritten. Existing output directories are refused to protect frozen IDs. If a check fails, the new directory contains a failure report and no usable ID manifests; fix the cause and use another version directory.

Use `--dataset-dir` for a relocated original dataset and `--split-dir` if its existing derived split is elsewhere. Sample sizes and chunk size are configurable; `venv/bin/python -m src.splits --help` lists the flags. Samples include every available country/no-match stratum; an undersized request fails rather than silently omitting a group. A one-entity validation stratum cannot appear in both development and evaluation, so the manifest records a warning. Deterministic ranking is independent of row order and read chunk size.

For future real prediction scoring, pass `--entities` with the saved development/evaluation ID file, `--truth` with the original training ground truth, and prediction/candidate files covering exactly those IDs. `--source1` optionally enables country breakdowns. The scorer streams ground truth but retains the selected query results in memory; use saved small samples for initial experiments. It validates ID syntax, not target existence; use the audited target pool and output checks for that guarantee.

See the [Pass 2 verification record](docs/pass2_verification.md) for the actual run results. Full-data audits are separate from `pytest` and do not need to run after every small edit.

## 7. Run the connected Pass 3 fixture

The completed run is already in `artifacts/pipeline/pass3_v1/`. Open its two output TSVs and `scored_pairs.tsv`, following the [detailed inspection walkthrough](docs/pass3_walkthrough.md).

To produce your own run, choose a new directory:

```bash
venv/bin/python -m src.pipeline --output-dir artifacts/pipeline/my_first_run
```

The default input is **`tests/fixtures/pipeline/`**, not `dataset/`. It has 10 training businesses and 8 separate prediction businesses, with invented answer keys. The command retrieves training pairs, labels them, builds features, trains LightGBM, saves/reloads it, predicts on the separate fixture records, writes both files, and validates them. Only then does it read `expected_test_matches.tsv` to calculate the fixture score. The actual competition test data has no such answer key.

Expected with the pinned environment and default settings: 17 training pairs (9 positive / 8 negative), 15 scored prediction pairs, 8 rows in each output, and a **fixture-only macro F0.5 of 0.875**. One deliberately missed true link remains a false negative. This score is not an estimate of challenge performance. The software tests check contracts rather than requiring this learned score on every machine.

The command refuses existing output directories, publishes results only after successful checks, and rejects source/truth files over 1,000 rows instead of silently truncating. `--fixture-dir` selects another compatible tiny fixture. `--top-k`, `--minimum-similarity`, `--threshold`, and `--seed` are available in `--help`. This runner uses small in-memory retrieval; the real Pass 2 experiment IDs are reserved for later full-pool retrieval and training work.

Recheck the saved outputs with the supplied validator:

```bash
venv/bin/python utils/validate_submission.py \
    --matching artifacts/pipeline/pass3_v1/matching_results.tsv \
    --candidate artifacts/pipeline/pass3_v1/candidate_pairs.tsv \
    --test-dir tests/fixtures/pipeline/test --check-ids
```

The validator's generic “Safe to submit” message means these files satisfy the format against the selected fixture. **These invented IDs are for local inspection only; do not upload the fixture outputs.** Project checks additionally require the candidate file and enforce matches being contained in the exact candidates scored.

## 8. Reuse the full target index for Pass 4

The compact run recovered **90.90% of true links** with **92.19 candidates per query** on average; the wider comparison recovered **92.50%** with **143.25 candidates**. The provisional 95% overall recall improvement target is not yet met. The compact configuration remains the initial classifier baseline default.

The completed full index is `artifacts/retrieval/train_full_v1/`. It contains all **10,320,219 training Source 2/3 records**. The benchmark selects only `sample_development_ids.tsv` from the saved Pass 2 experiment, verifies input/index fingerprints, and compares three retrieval variants on those 1,000 queries.

```bash
venv/bin/python -m src.retrieval_evaluation \
    --index-dir artifacts/retrieval/train_full_v1 \
    --output-dir artifacts/retrieval/my_development_run
```

Use a new output directory each time. The default experiment is `artifacts/evaluation/pass2_v1/`; use `--experiment-dir` or `--dataset-dir` for relocated inputs. Read the [measured comparison](docs/pass4_verification.md) and [what to inspect](docs/pass4_walkthrough.md). Candidates are saved before the answer key is loaded for grading. The report contains candidate recall, list-size distributions, a theoretical score ceiling, country breakdowns, resource use, and missed-link diagnostics. These are retrieval measurements, not classifier or leaderboard scores.

On a machine without the index, build it once:

```bash
venv/bin/python -m src.retrieval_index \
    --output-dir artifacts/retrieval/train_full_v1
```

The recorded build took 338.8 seconds, peaked at 232.13 MiB process RSS, and wrote a 3.98 GiB database. These measurements are from this machine, not guarantees. The builder reads raw targets in chunks, preserves original text, and stores exact-name/address indexes plus SQLite FTS5 word indexes. The standard-library SQLite build must support FTS5; this workspace uses SQLite 3.46.1. No new pip dependency was needed.

`--limit-per-source` is an optional restricted build pilot; its index is explicitly marked restricted and rejected by the full-pool benchmark. Both builder and evaluator protect existing output directories and publish results only after checks succeed. The main retrieval limits (`--per-channel`, `--max-candidates`, `--max-terms`, `--max-token-documents`) are recorded in each report. The default final cap is 100 candidates, not five or eleven.

Generated `candidate_pairs.tsv` contains the candidates available for later ML scoring. `retrieval_trace.tsv` records each candidate's rank, contributing routes, and string-based ordering score. Pass 4 does not emit accepted `matching_results.tsv`; realistic model fitting and threshold selection belong to Pass 5.

## Paths and compatibility commands

The default dataset path is anchored to this repository, even when a script is invoked from another working directory. An explicitly supplied relative `--dataset-dir` or `--report` path is relative to your current working directory. Use absolute paths when that is clearer.

Both older inspector commands call the same implementation and accept the same options:

```bash
venv/bin/python inspect_dataset.py --help
venv/bin/python src/inspect_data.py --help
```

The other scripts also have help and safe imports:

```bash
venv/bin/python candidate_generation.py --help
venv/bin/python split_validation.py --help
venv/bin/python train_model.py --help
```

The existing candidate demo can be run against the already-created validation split:

```bash
venv/bin/python candidate_generation.py --query-rows 2 --targets-per-source 1000
```

It prints candidates from a restricted pool and does not establish matching accuracy. It requires `dataset/val/val_source1.tsv` in addition to the original training source files.

The older `split_validation.py` command materializes Source 1 and ground truth and regenerates derived split files. By default it replaces the existing derived files under the selected dataset root; use `--output-dir` to put a separate split elsewhere. It is not the Pass 2 audit command. Regenerating the split changes experiment inputs and requires a new audit/version. The root candidate/training demos are historical, independent examples. Use `python -m src.pipeline` for the connected Pass 3 fixture. Pass 4 uses the saved development IDs with the full target index; real supervised training and final evaluation belong to Pass 5.

The training demo still uses random negatives and scores its training pairs. Its labels can conflict with ground truth, and its pair-level F0.5 is not the competition metric. The new `src.pipeline` path replaces these practices with labels from retrieved pairs and saved-model inference on separate invented records. Honest real-data evaluation still belongs to Pass 5.

## Why the reader preserves `NA`

The shared loader first uses Python's standard TSV/CSV parser to validate the exact header and the number of fields in every consumed row, then builds a Pandas table from the strings. This avoids both automatic null-token conversion and silent filling of missing columns. It supports standard quoted TSV fields, including embedded tabs and newlines.

`NA`, `NULL`, and `nan` remain literal strings. Empty and whitespace-only fields remain unchanged too; a separate blank indicator identifies them. We never infer what a short token means merely from a parser default. See the [verified `NA` examples](docs/dataset_insights.md).

The strict reader rejects malformed headers/rows instead of skipping them. A sample validates only its consumed rows. The inspector counts ID prefixes and match-list lengths; the separate Pass 2 audit checks training ID uniqueness, relationships, and split integrity. It does not establish the semantic truth of the supplied labels or perform the equivalent full audit of test data.

## Repository contents and next pass

| Location | Purpose |
|---|---|
| `src/config.py` | Shared dataset path and command-line helpers |
| `src/data_loading.py` | Text-preserving reader, blank indicators, TSV writer |
| `src/inspect_data.py` | Shared sample/full inspector |
| `src/splits.py` | Disk-backed training/split audit and frozen experiment ID generation |
| `src/scoring.py` | Official macro F0.5 and separate pair/candidate diagnostics |
| `src/evaluation_io.py` | Shared ID-list validation and file fingerprinting |
| `src/pipeline.py` | Connected synthetic fixture runner and inspectable pair traces |
| `src/normalization.py`, `src/features.py` | Raw-preserving normalized copies and shared numerical features |
| `src/candidates.py`, `src/pair_labels.py` | Tiny TF-IDF retrieval and ground-truth pair labels |
| `src/model_io.py`, `src/submission.py` | Saved-model contracts, thresholding, and strict output validation |
| `src/retrieval_index.py` | Chunked full-target SQLite index and reusable retrieval |
| `src/retrieval_evaluation.py` | Fixed development candidate benchmark and error reports |
| `brain.md` | Current project state, decisions, artifacts, and next work |
| `tests/` | Regression tests and tiny trackable synthetic TSVs |
| `requirements*.txt` | Pinned environment |
| `docs/` | Beginner guidance, verified findings, and test design |
| `artifacts/inspection/` | Local generated JSON reports, ignored by Git |
| `artifacts/evaluation/` | Local audit reports, experiment IDs, and scoring reports, ignored by Git |
| `artifacts/pipeline/` | Fixture models, pair traces, output TSVs, and reports, ignored by Git |
| `artifacts/retrieval/` | Real target index, development candidate files, and reports, ignored by Git |

Git ignores the real dataset, virtual environments, generated outputs, databases, serialized models, archives, caches, and machine-local settings. Synthetic fixtures and the supplied challenge README/template remain eligible for version control. Teammates can run the regression suite after cloning and installing dependencies, without downloading the real dataset. Review `git status --short` before staging; changes are not automatically committed or pushed.

Review [Pass 4](docs/pass4_verification.md) before authorizing **Pass 5: an honest real-data ML baseline**. Complete competition predictions and uploading remain later passes.
