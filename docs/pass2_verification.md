# Pass 2 Verification Record

Pass 2 completed on **2026-09-26**. It establishes trustworthy training partitions and score calculation. It does not establish model accuracy. See the [README](../README.md) for commands and the [implementation plan](../implementation_plan.md) for the next pass.

## Results

| Check | Result |
|---|---|
| Accumulated fast regression suite | **96 passed** in 7.24 seconds |
| Official hand-calculated scoring cases | Exact matches, partial matches, false positives, retrieval misses, singleton rules, and macro averaging agree |
| Scoring command fixture | Three business scores of 1, 0, 1 give **macro F0.5 = 2/3**; pair/candidate recall = 0.8 |
| Original training Source 1 IDs | **2,206,821**, unique with valid prefixes |
| Original training Source 2/3 IDs | **10,320,219**, unique with correct source prefixes |
| Original ground truth | Exactly one row per Source 1 entity, including empty lists |
| Ground-truth target references | **7,638,365** links; every target exists and belongs to at most one reference business |
| Existing 90%/10% partition | Disjoint, duplicate-free, and covers all original Source 1 IDs |
| Derived source/label contents | Source fields match the originals; label sets match, independent of list order |
| Saved experiment files | Counts and hashes checked; full partition coverage and sample containment independently rechecked |
| Temporary audit database | Deleted after completion |

The corruption tests deliberately introduce missing/duplicate rows, wrong IDs, nonexistent targets, changed names/labels, and shared target ownership. Each fails instead of producing usable manifests. Reordering input rows or changing chunk size preserves the selected IDs; changing the seed changes the tested assignment.

## Saved partitions and samples

The original validation set contains 220,682 businesses. It was divided within each country/no-match stratum using seed 42 and a development fraction of 0.5. Rounding each stratum down for development accounts for the two-row difference between the final development/evaluation totals.

| Role | Complete partition | Small sample | Purpose |
|---|---:|---:|---|
| Training | 1,986,139 | 5,000 | Fit the model |
| Development | 110,340 | 1,000 | Choose retrieval rules, features, and thresholds |
| Evaluation | 110,342 | 1,000 | Measure once after settings are frozen |

All three complete partitions and all three samples contain US and India businesses with and without true matches. There were **no rare-stratum warnings** in this run.

| Small-sample stratum | Training | Development | Evaluation |
|---|---:|---:|---:|
| India, has matches | 1,889 | 376 | 376 |
| India, no matches | 113 | 23 | 23 |
| US, has matches | 2,830 | 566 | 566 |
| US, no matches | 168 | 35 | 35 |

Generated files are under `artifacts/evaluation/pass2_v1/`, which is ignored by Git. Each ID TSV has one column, `source1_entity_id`. `manifest.json` records all six file counts/hashes, strata, exact seed/selection rules, and input file hashes. `audit.json` records the complete checks and measurements. Preserve this directory; reproductions should use a new version directory.

The manifest fixes Source 1 query sets only. The target pool remains **all 10,320,219 original training Source 2/3 records**. No target candidates were selected using validation labels.

## Run measurement

Executed command:

```bash
venv/bin/python -m src.splits --output-dir artifacts/evaluation/pass2_v1
```

- Started: **2026-09-26 13:49:02 UTC**.
- Finished: **2026-09-26 13:52:06 UTC**.
- Audit plus manifest generation: **184.008 seconds**, approximately 3 minutes 4 seconds.
- Peak process resident memory reported by the OS: **251,461,632 bytes**, approximately **239.81 MiB**.
- Environment: Linux/WSL2, CPython **3.14.4**, SQLite **3.46.1**.
- Input chunk size: **50,000 rows**; temporary SQLite page-cache target: **64 MiB**.

These are measurements from this run, not guaranteed bounds. SQLite uses additional temporary disk space. The process memory measure is separate from host filesystem caches and container limits. The independent post-generation check ran separately and is not included in these measurements.

## Scorer behavior

The scorer averages F0.5 across every selected Source 1 business, including empty candidate lists. It does not filter truth to the candidate universe. Missing prediction rows are errors; explicitly empty rows are valid. Optional candidate files must cover the same businesses and contain every predicted target.

The report separately labels pooled pair precision/recall, singleton accuracy, country breakdowns, candidate recall, and the best possible score using those candidates. These diagnostics are not interchangeable with the official macro score. Undefined diagnostic ratios are JSON `null`, not fabricated 0 or 1 values.

The hand-checked report is saved at `artifacts/evaluation/hand_checked_score.json`. It uses invented records and is a correctness demonstration, not a result from model inference.

## Limits and next step

- The audit validates the original **training** data and its existing split. It does not perform an equivalent uniqueness/reference audit on test data.
- It establishes label consistency and reference existence, not the real-world correctness of supplied business matches.
- It checks the current legacy split contents, not which historical random seed created that split.
- The scorer validates ID syntax and coverage; target existence is established by the audit and later output validation.
- Evaluation IDs are reserved for later use. No evaluation predictions, threshold selection, or model tuning were performed in this pass.
- The existing candidate/training demonstrations have not been connected to these manifests. That is Pass 3 work.

Review the tests and scoring fixture, then proceed to **Pass 3: a tiny complete pipeline**.
