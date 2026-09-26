# Pass 4 — Inspect retrieval on real data

Pass 3 demonstrated a complete pipeline on invented records. Pass 4 measures its first important stage on real data: **can we find the true targets in a manageable candidate list?** A later model can only accept or reject pairs that reach it.

The benchmark uses the fixed **1,000 development businesses** saved in Pass 2 and searches all **10,320,219 training Source 2/3 records**. It does not use the reserved evaluation queries or competition test data. These development results can guide improvements, but they are not a final model or leaderboard score.

## What changed in retrieval?

The small Pass 3 retriever scans a tiny in-memory TF-IDF target matrix. The real target pool now has a reusable **SQLite database on disk**, built in chunks. SQLite is accessed through Python's standard library; no new package installation is required in the verified environment.

Four routes propose candidates:

| Route | What it searches | Example of why it helps |
|---|---|---|
| `exact_name` | Equal normalized names | Case/spacing variations of a business name |
| `exact_address` | Equal normalized addresses | A changed or abbreviated name at the same address |
| `name_words` | Selected name words | Extra or reordered words in a name |
| `address_words` | Selected address words | Partial address overlap |

The word routes use indexed token frequencies to select up to four relatively rare words per field. They skip tokens appearing in more than 20,000 target records and tokens shorter than two characters. They search with OR, so sharing one selected word can produce a candidate. This helps keep search cost manageable; it can also lose real matches. Exact lookup remains available even for common names.

Each route returns at most 50 records. The combined pool is deduplicated, ordered using name/address string similarity, and capped at 100 candidates per business. These are recorded, configurable retrieval limits, not limits on how many matches a business is allowed to have. No route requires equal countries. Blank fields do not become shared evidence. Raw text, including `NA` and non-Latin scripts, remains stored alongside normalized copies.

Word overlap does not translate scripts or recognize every spelling error. A target with a very different name can still be found through its address; a target with no usable overlap may remain invisible to these routes. The benchmark measures that limitation rather than assuming it away.

SQLite's own [FTS5 documentation](https://www.sqlite.org/fts5.html) describes its full-text index, Unicode tokenizer, vocabulary tables, and BM25 ranking. Our channel selection, candidate caps, final similarity ordering, and measurements are project-specific.

## 1. Read the comparison report

Open [the development report](../artifacts/retrieval/development_v1/report.json), then the [verification record](pass4_verification.md) for a compact comparison table.

The completed compact run found **3,067 of 3,374 true links (90.90%)**, with **92.19 candidates per business** on average. A [wider comparison](../artifacts/retrieval/development_wider_v1/report.json) found **92.50%**, averaging **143.25 candidates**. The compact setting remains the default for initial classifier work; neither reaches the provisional 95% recall target.

Look at `variants`:

- `name_only`: exact-name and name-word routes. This is the simple baseline.
- `exact_name_address`: only the two exact routes. This shows what exact lookup can recover.
- `hybrid`: all four routes, followed by the final candidate limit.

Read `true_links`, `recovered_true_links`, and `pair_recall` together. If the answer key contains ten real links and our shortlist contains nine of them, candidate recall is 9/10. Incorrect candidates do not directly reduce this number, which is why we also report candidate-list sizes. The later classifier must reject incorrect candidates.

`mean_candidates`, `p95_candidates`, and `maximum_candidates` describe how much later scoring work we create. For example, a 95th percentile of 100 means at least 95% of the observed list sizes are at most 100. `oracle_macro_f0_5` assumes perfect accept/reject decisions within these candidates; it is an upper bound, not an achieved model score.

`channels_before_final_cap` and `union_before_final_cap` help distinguish a missing retrieval route from a match lost during final truncation. `by_country` covers the countries in this development sample. France remains absent from real training development data; the multilingual synthetic tests do not establish France accuracy.

## 2. Follow one development business

Open [query_details.tsv](../artifacts/retrieval/development_v1/query_details.tsv). Each row is one of the 1,000 selected businesses.

| Column | What to inspect |
|---|---|
| `true_links` | Number of known matches for this business |
| `name_only_recovered` | True links available through the name baseline |
| `hybrid_recovered` | True links available after all four routes and the final cap |
| `hybrid_candidates` | Total pairs the later classifier would examine |
| `missed_ids` | Correct target IDs still absent from the shortlist |
| `name_terms`, `address_terms` | Selected words used in the indexed searches |
| `query_seconds` | Measured time for all channels and final ordering |

Find a row where `hybrid_recovered` exceeds `name_only_recovered`. Address search helped that business. Then find one with nonempty `missed_ids`: there is still a limitation to understand.

Zero true links is valid. Such a business may still get candidates, and a later classifier needs to reject them. An empty candidate list also remains an explicit row.

## 3. Inspect a missed true pair

Open [missed_links.tsv](../artifacts/retrieval/development_v1/missed_links.tsv). It contains every missed true link in this sample, with the raw names, addresses, countries, and two string-similarity clues.

- `not_returned_by_channels` means none of the returned channel lists supplied that target. Missing words, very common words, different scripts, spelling changes, or a per-channel limit may be involved; that label alone does not prove which cause applies.
- `final_candidate_cap` means a route found the target, but final ordering and truncation removed it.

The file is produced **after retrieval**, by comparing the saved candidates with ground truth. It is for development error analysis. Its contents were not used to insert missing targets into the candidate lists.

## 4. Understand the candidate files and trace

Open [candidate_pairs.tsv](../artifacts/retrieval/development_v1/candidate_pairs.tsv) and [retrieval_trace.tsv](../artifacts/retrieval/development_v1/retrieval_trace.tsv).

The first file has exactly one row per development query and unique target IDs in its second column. The trace has one row per candidate pair, its rank, the routes that found it, and a `retrieval_score` used for final ordering. That score is a string-similarity formula with a possible maximum of 1.1. It is **not a LightGBM score or a probability**.

These are the final retrieval candidates available for a future classifier. Pass 4 does not create `matching_results.tsv` or claim these pairs are accepted matches. When the real model is connected in Pass 5, it must score this exact final candidate set and preserve any later filtering in the exported candidate file.

The report fingerprints the files so a future experiment can identify precisely which candidate version it consumed. Generated artifacts remain local and ignored by Git.

## 5. Run checks or reproduce the benchmark

From the repository root, run the normal software tests:

```bash
venv/bin/python -m pytest -q
```

To focus on Pass 4:

```bash
venv/bin/python -m pytest -v tests/test_retrieval_index.py tests/test_retrieval_evaluation.py
```

The tests use tiny invented datasets and temporary SQLite files. They do not build a full real-data index.

The completed real index is already in `artifacts/retrieval/train_full_v1/`. Reuse it to repeat the development benchmark into a **new** directory:

```bash
venv/bin/python -m src.retrieval_evaluation \
    --index-dir artifacts/retrieval/train_full_v1 \
    --output-dir artifacts/retrieval/my_development_run
```

The default experiment directory is `artifacts/evaluation/pass2_v1/`. The command verifies its saved development selection, source fingerprints, complete target coverage, and index fingerprint. It rejects changed inputs or a restricted pilot index instead of silently reporting them as the full benchmark.

On another machine, after reproducing Pass 2, build a full index once:

```bash
venv/bin/python -m src.retrieval_index \
    --output-dir artifacts/retrieval/train_full_v1
```

This explicitly reads all training targets and uses several GiB of disk; see the [measured build cost](pass4_verification.md). Existing output directories are protected. Use `--dataset-dir` and `--experiment-dir` for relocated data and experiment files. `--help` lists the retrieval limits. Changing limits creates a new development experiment, not an untouched evaluation.

`--limit-per-source` on the index builder is only for a measured restricted pilot. Its index is labeled as restricted and cannot be used by the full-pool development benchmark. Do not point the original Pass 3 fixture command at the real dataset.

## What to review before real model training

Review the [measured recall/cost decision](pass4_verification.md), inspect several misses, and read [brain.md](../brain.md) for the current state and next work. Pass 5 will build realistic labeled training pairs, train the classifier, choose its threshold on development data, and only then use reserved evaluation businesses after settings are frozen.
