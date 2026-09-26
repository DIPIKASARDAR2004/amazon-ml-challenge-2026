# Implementation Plan — Work in Passes

**Current status: Pass 4 complete, with a measured full-pool retrieval baseline and comparison.** Start with [brain.md](brain.md), the [Pass 4 walkthrough](docs/pass4_walkthrough.md), and [verification record](docs/pass4_verification.md). The provisional 95% overall / 92% India retrieval improvement targets remain unmet; measured recall and cost are documented honestly. Pass 5 has not started.

Start with the [team guide](docs/team_guide.md), then read the [dataset findings](docs/dataset_insights.md). The [testing plan](docs/testing_plan.md) explains the future test files and their expected behavior. Competition requirements come from the supplied [challenge README](<Data Set ML Challenge/student_resource/README.md>).

## What already exists

- [x] Original datasets and a working local Python environment.
- [x] Full source-file counts and training match statistics inspected; the `NA` parsing issue independently rechecked against raw records and labeled matches.
- [x] A seeded 90%/10% Source 1 split script and its output files; integrity verified in Pass 2.
- [x] A runnable candidate demonstration over 200,000 target records.
- [x] Training demonstration code using 5,000 Source 1 records, name/address similarity, and LightGBM.
- [x] A supplied submission validator, checked with small valid/invalid examples.
- [x] Shared text-preserving TSV loading, portable paths, safe script imports, pinned dependencies, and 39 passing Pass 1 regression cases.
- [x] The maintained inspector reproduces the complete documented counts in a bounded-memory scan.
- [x] Full training/split integrity audit, fixed training/development/evaluation IDs, and the official scorer. Pass 2 finished with 96 passing cases.
- [x] A connected, tested synthetic pipeline with reloadable model, pair traces, and both validated output files.
- [x] A reusable full training target index, fixed development retrieval comparison, resource measurements, and missed-link analysis; 190 accumulated regression cases pass.
- [ ] A trustworthy held-out classifier score on real challenge businesses.
- [ ] A saved final model, complete test predictions, and final package.

The training demonstration scores its own training pairs with a pair-level metric. That is not held-out evaluation or the official per-business metric. The candidate demonstration's target pool contains only 14,944 of 763,411 validation true links (1.96%), before retrieval. These demos demonstrate mechanics, not competitive performance.

## How we will work

1. Pick one task within the current pass and agree on the behavior it must provide.
2. During implementation, add a small test for important correctness requirements, then implement the behavior. When fixing a bug, first capture it as a regression test where practical.
3. Run the relevant checks, examine a small output, and explain the result in plain language.
4. Record changes, evidence, limitations, and the next action. Review the exit checkpoint before beginning another pass.

Checked boxes indicate completed evidence, not intentions. Pass 1–4 tests now exist; later-pass additions remain planned. No full-data model run belongs in the default test suite. A passing software test does not establish ML accuracy.

## Pass overview

| Pass | Purpose | Result we review |
|---|---|---|
| 0 | Finalize facts and the plan | Shared understanding, documented assumptions, and planned tests |
| 1 — complete | Make setup and data reading reliable | Reproducible environment, safe text loading, and first regression tests |
| 2 — complete | Make evaluation trustworthy | Verified splits and an independently checked official scorer |
| 3 — complete | Connect a tiny complete pipeline | Candidate → features → model → two output files works on a small fixture |
| 4 — complete | Improve retrieval at realistic scale | Small query set searched against the full training target pool, with measured recall/cost |
| 5 | Establish an honest ML baseline | Realistic training pairs, development tuning, and one untouched evaluation |
| 6 | Scale the frozen approach | Resumable full test inference and both complete output files |
| 7 | Validate, submit, and package | Reviewed leaderboard file and reproducible final ZIP |

## Pass 0 — Finalize facts and decisions

**Why:** Incorrect assumptions about the data or score can invalidate all later work.

- [x] Distinguish raw blank fields, literal text such as `NA`, and missing values introduced by a parser.
- [x] Trace all 15 training `NA` records through ground truth; document evidence for abbreviated names without claiming to know the generator's intent.
- [x] Document what the current demos can and cannot establish.
- [x] Draft beginner explanations, pass checkpoints, and meaningful regression cases.
- [x] Plan owner reviews the plan and authorizes Pass 1 using the data-handling policy below.

**Data-handling policy:** Preserve raw text, including `NA`, Unicode, and IDs. Derive normalized text in separate fields. Treat empty/whitespace-only name or address fields as blank for missingness indicators. Do not infer that a nonempty token is missing merely because Pandas recognizes it as a null token. An empty ground-truth match list means zero matches, not a broken record. Short names such as `NA` remain weak evidence and never establish identity on their own.

**Exit checkpoint:** Scope, evidence, data policy, and test strategy are documented, and the plan owner has authorized the first implementation pass.

## Pass 1 — Reliable setup and data reading

**Why:** Every downstream result depends on loading the original information correctly.

- [x] Replace absolute machine paths with a shared configurable dataset path; document commands run from the repository root.
- [x] Put script execution behind `main()` guards so imports do not scan data or start training.
- [x] Record working runtime dependencies and the `pytest` development dependency; add a reproducible root setup guide.
- [x] Correct ignore rules for `dataset/`, `venv/`, generated outputs/artifacts, extracted data copies, and OS metadata. Review source/document tracking separately from generated data; new project files remain uncommitted for review.
- [x] Create tiny synthetic TSV fixtures and the first tests described in the testing plan. Preserve `NA` and distinguish it from empty and whitespace-only fields.
- [x] Consolidate inspection into a chunked reader that validates schemas and reports raw blanks separately from tracked literal tokens. Replace Windows-only memory inspection with a supported local check.
- [x] First inspect the fixture, then reproduce the documented full-file counts with bounded memory. Record the measurement environment and time.

**Tests introduced:** `test_data_loading.py`, `test_entrypoints.py`.

**Deliverables:** Setup instructions, explicit loading policy, dependency records, small fixtures, first regression tests, and a reproducible inspection report.

**Exit checkpoint:** Literal `NA`, IDs, commas, tabs, and Unicode survive reading/writing as specified; blank fields are counted correctly; malformed schemas fail clearly; imports do not launch work. The full inspection agrees with the documented statistics.

**Verification:** 39 tests passed. The full scan on 2026-09-26 reproduced all documented counts in 49.918 seconds, with a measured peak process RSS of approximately 113.85 MiB at 50,000 rows per chunk. These are measurements from this run, not runtime/memory guarantees. Details and commands are in the [verification record](inspection_report.md).

## Pass 2 — Correct splits and scoring

**Why:** A misleading score can make a weak model appear successful.

- [x] Audit the existing Source 1 split for duplicates, overlap, coverage, and exact ground-truth alignment. Verify target references before relying on labels.
- [x] Save fixed training/development/evaluation ID lists and seeds. Split by Source 1 business, never by pairs belonging to the same business.
- [x] Use the existing 90% portion for training. Divide the remaining 10% into development data for choosing settings and reserved evaluation data for the final estimate; include both training countries and no-match businesses in small samples.
- [x] Implement the official per-business F0.5 scorer, then average over every selected Source 1 business, including those with no candidates.
- [x] Keep true links missed by retrieval in the false-negative count. Require complete prediction rows instead of silently dropping businesses.
- [x] Add candidate recall and clearly labeled diagnostic precision/recall. Compare them with hand-calculated examples in the testing plan.

For each business, `TP` means correct predicted links, `FP` means incorrect predicted links, and `FN` means true links not predicted. With nonempty truth, `F0.5 = 1.25 × TP / (1.25 × TP + FP + 0.25 × FN)`. With empty truth, an empty prediction scores 1 and a nonempty prediction scores 0. The final score is the arithmetic mean of all business scores.

**Tests introduced:** `test_splits.py`, `test_scoring.py`.

**Deliverables:** Split audit, fixed experiment ID lists, tested scorer, and a hand-checked example report.

**Exit checkpoint:** Entity sets are disjoint and complete, labels align, and every hand-calculated score passes. Model performance is still unmeasured at this stage.

**Verification:** 96 accumulated tests passed. The full audit verified 7,638,365 target links and the existing 1,986,139/220,682 split. Seed 42 produced 110,340 development and 110,342 evaluation businesses, with small 5,000/1,000/1,000 training/development/evaluation samples. Files are under `artifacts/evaluation/pass2_v1/`; see the [Pass 2 verification record](docs/pass2_verification.md). No model accuracy has been measured.

## Pass 3 — A tiny complete pipeline

**Why:** Prove that components connect correctly before investing in expensive retrieval or training.

- [x] Build a shared normalization/feature path that preserves raw text, handles blanks explicitly, and keeps multilingual text available.
- [x] Connect a simple candidate generator, pair labeling, RapidFuzz features, a small LightGBM training run, scoring, thresholding, and output writing on a tiny fixture.
- [x] Generate negatives from incorrect retrieved pairs; never label a known positive as negative. Prevent duplicate pairs.
- [x] Save and reload the model together with feature order and preprocessing settings; check equivalent predictions within a stated tolerance.
- [x] Produce both output files, preserving zero/one/many matches, empty candidates, and exactly one row per fixture Source 1 entity.
- [x] Pass a tiny fixture through the supplied validator, including ID-existence checks. Test invalid outputs too.

**Tests introduced:** `test_normalization.py`, `test_candidates.py`, `test_pair_labels.py`, `test_features.py`, `test_model_io.py`, `test_submission.py`, `test_pipeline_smoke.py`.

**Deliverables:** A runnable small pipeline, reloadable model artifact, and two valid fixture outputs.

**Exit checkpoint:** The components agree on IDs, features, candidates, and output formats. Fixture scores are plumbing checks, not estimates of competition performance. No need for full-data training here.

**Verification:** The default run trained on 17 retrieved synthetic pairs (9 matches, 8 nonmatches), scored 15 separate prediction pairs, and wrote 8 rows in each output. Reloaded scores differed by 0 within a 1e-12 tolerance. Project checks and the supplied validator passed with ID existence enabled. The deliberately missed link remains a false negative. See the [run record](docs/pass3_verification.md) and [inspection walkthrough](docs/pass3_walkthrough.md).

**Scope clarification:** The new `src.pipeline` command connects reusable components on its own synthetic fixture. It does not modify the historical root demos or use the real Pass 2 ID lists. Full-pool retrieval on those IDs begins in Pass 4, and real supervised training/evaluation in Pass 5.

## Pass 4 — Useful retrieval with measured cost

**Why:** The model cannot recover true matches missing from its candidate list.

- [x] Establish the simple retrieval baseline on fixed development queries; clearly label any experiment using a restricted target pool.
- [x] Test complementary name and address retrieval rules. Include cases involving short abbreviations, different scripts, and names that differ despite strong address evidence.
- [x] Do not require exact country/name/address agreement without checking its effect on training/development true links. Support arbitrary country labels.
- [x] Choose an index design using measured memory, runtime, and disk cost. Use chunked ingestion and disk-backed indexing as needed; avoid all-to-all comparisons.
- [x] Build access to the complete training Source 2/3 pool, then query a small fixed development sample. Never use ground truth to insert missing validation candidates or preselect the target pool.
- [x] Deduplicate and persist the actual candidates to be scored. Make candidate limits configurable; the observed maximum of 11 true matches is not a safe universal cap.
- [x] Measure candidate recall, empty lists, average/upper-percentile/maximum list sizes, runtime, memory, and failure examples. Set a documented recall/cost target using this baseline before expanding work.

**Tests introduced:** `test_retrieval_index.py`, `test_retrieval_evaluation.py`; command/import coverage extended in `test_entrypoints.py`.

**Behaviors verified:** Candidate coverage on designed fixtures, abbreviation/address retrieval, duplicate removal, deterministic limits/ties, empty results, arbitrary country labels, and query-batch consistency.

**Deliverables:** Reusable retrieval configuration/index and a development report against the complete training target pool.

**Exit checkpoint:** The team reviews the measured recall/cost tradeoff and remaining misses. No unmeasured claim that the shortlist is good enough.

**Verification:** Indexed all 10,320,219 training targets in 338.8 seconds, with 232.13 MiB peak process RSS and a 3.98 GiB database. On the frozen 1,000-query development sample, compact combined retrieval recovered 3,067/3,374 links (90.90%) versus 49.59% for the name-only baseline, averaging 92.19 candidates. A wider comparison recovered 92.50%, averaging 143.25 candidates. All 190 regression cases passed. See the [record](docs/pass4_verification.md) for country breakdowns, misses, timings, fingerprints, and limits.

**Decision:** Keep the compact 50-per-route / 100-final configuration as the initial Pass 5 default; retain the wider 100/150 run as a measured alternative. The wider setting adds 54 recovered links overall at 55.39% more candidate pairs. Before scaling, aim for at least 95% overall / 92% India recall within the documented cost limits. Both configurations miss these recall targets, so retrieval remains an improvement area. Completing this measurement pass does not establish sufficient final model quality.

## Pass 5 — An honest ML baseline

**Why:** Learn from the difficult pairs retrieval actually produces, then measure generalization to unseen businesses.

- [ ] Retrieve pairs for the selected training businesses and label them using training ground truth. Keep full truth separately for evaluating retrieval misses.
- [ ] Start with name/address similarity and missingness features; add richer features only when development errors justify them. Preserve the same feature computation for training and inference.
- [ ] Train LightGBM with recorded seeds/settings. Keep development/evaluation businesses out of supervised training and target-derived tuning.
- [ ] Use development macro F0.5 to choose the acceptance threshold. Allow zero, one, or many matches; do not force a match.
- [ ] Freeze the approach and score the untouched evaluation businesses once using the complete retrieval-to-output workflow.
- [ ] Report macro F0.5, candidate recall, diagnostic precision/recall, country/singleton breakdowns, error examples, resource use, and exact query/target scope.
- [ ] Save model, feature order, normalization/retrieval settings, threshold, split identifiers, environment versions, and results. If evaluation errors drive tuning, it is no longer untouched; reserve a fresh evaluation subset for the next final estimate.

**Tests extended:** No contradictory labels, training/inference feature consistency, model reload tolerance, threshold boundary behavior, and output candidate-subset consistency.

**Deliverables:** A reproducible held-out report and saved baseline configuration/model.

**Exit checkpoint:** A trustworthy score and inspected errors support a specific decision to improve or scale. No training-set score is described as validation.

## Pass 6 — Scale and generate test predictions

**Why:** A correct small pipeline must fit the available resources and cover every test business.

- [ ] Increase training/query sizes in measured stages, recording memory, runtime, disk needs, and development quality before the next increase.
- [ ] Add checkpoint/resume behavior and verify it on a deliberately interrupted small job. Changing configuration must not silently reuse incompatible artifacts.
- [ ] Freeze the final approach and threshold, then retrain on the chosen full-training configuration after reviewing the baseline. Preserve the earlier held-out score as evidence about that earlier experiment.
- [ ] Build the test target index from test Source 2/3 only. Run a small execution/format check including France; this cannot measure accuracy without test labels.
- [ ] Process all 1,732,544 test Source 1 records in bounded-memory batches.
- [ ] Write `output/candidate_pairs.tsv` with exactly the last candidate set passed to model inference, and `output/matching_results.tsv` with accepted matches.
- [ ] Ensure both files contain one row per test Source 1 entity, including empty lists, unique existing test S2-/S3- targets, and final matches contained in candidates.

**Tests introduced/extended:** `test_resume.py`, batch-size equivalence, complete output coverage, and a small mixed-country integration run.

**Deliverables:** Complete test outputs, final artifacts/settings, and a resource/resume report.

**Exit checkpoint:** All required rows exist and incremental processing agrees with uninterrupted processing on the small reference job. Full-output data checks pass.

## Pass 7 — Validate, submit, and package

**Why:** Correct formatting and reproducibility are separate requirements from a good model score.

- [ ] Run the supplied validator on both final files. Verify target-ID existence with `--check-ids` when memory permits, or a separately tested bounded-memory equivalent.
- [ ] Treat missing candidate rows or matches outside candidates as project failures even where the supplied validator only warns. Understand every remaining warning before submission.
- [ ] Review the output summary, country coverage, and representative rows.
- [ ] After review, upload only `matching_results.tsv` to the leaderboard portal. Record the submitted version, status, score, and submission time; retain the exact uploaded file.
- [ ] Prepare the separate final ZIP with both output files, runnable code under `code/business_entity_resolution/src/`, reproduction `README.md`, pinned dependencies, and completed `Documentation_template.md`.
- [ ] Confirm the final model satisfies the supplied MIT/Apache 2.0 and size requirements. Keep entity resolution limited to the provided data; do not query external business databases, geocoding services, or identity APIs.
- [ ] Run a clean-environment reproduction check on small data first; verify the documented full regeneration process and final archive layout.

**Tests extended:** Validator acceptance/rejection cases and package/command integration checks.

**Deliverables:** Reviewed leaderboard upload, experiment/submission record, and reproducible final ZIP.

**Exit checkpoint:** Submission and packaging requirements are met, and another teammate can follow the instructions to regenerate outputs.

## Next course of action

Review the [Pass 4 comparison and misses](docs/pass4_verification.md), then use [brain.md](brain.md) to resume work. The next planned scope is **Pass 5: an honest real-data classifier baseline**, using fixed training/development IDs and the reusable target index. Begin with the documented compact retrieval configuration, preserve the wider comparison, and keep retrieval improvement targets visible. Reserved evaluation remains untouched until settings are frozen.
