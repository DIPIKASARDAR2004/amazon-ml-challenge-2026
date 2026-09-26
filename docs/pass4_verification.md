# Pass 4 verification — Full-pool retrieval

**Completed on 2026-09-26.** We built a reusable index over every training Source 2/3 record, compared retrieval methods on the frozen development sample, inspected missed links, and measured resource costs. **190 regression cases passed in 14.39 seconds.** Start with the [inspection walkthrough](pass4_walkthrough.md) for definitions and files to open.

Pass 4 establishes a measured retrieval baseline. It does not train a real-data classifier or meet every future quality target. Pass 5 has not started.

## Scope and controls

- Query set: the exact 1,000 IDs in `artifacts/evaluation/pass2_v1/sample_development_ids.tsv`, fingerprint checked against its manifest.
- Query composition: 399 India / 601 US businesses; 942 with matches and 58 without matches. Ground truth contains **3,374 true links** for these queries.
- Target pool: **5,034,616 Source 2 + 5,285,603 Source 3 = 10,320,219 records**, with source fingerprints and total coverage checked against the Pass 2 audit.
- No target selection from labels, no ground-truth insertion of missing candidates, and no reserved evaluation queries or competition test records used.
- Candidate files are written before the answer key is loaded for grading. The index builder does not accept or read Source 1 labels.
- The SQLite database is opened read-only for retrieval, and its full file fingerprint is checked at benchmark startup.

The held-out development sample can be used to choose settings. These results are not an untouched final evaluation and not a model/leaderboard score.

## Retrieval comparison

The compact/default configuration uses **50 results per route and 100 final candidates**. Routes are exact name, exact address, name words, and address words. Word searches use up to four relatively rare tokens per field, ignoring tokens shorter than two characters or found in more than 20,000 targets. Names and addresses are normalized separately from preserved raw fields. There is no country filter.

The wider comparison changes only the limits to **100 per route and 150 final candidates**. Both use the same full index, query IDs, token rules, and answer key.

| Configuration | True links recovered / 3,374 | Candidate recall | Mean candidates | p95 / maximum | Empty queries | Perfect-classifier macro ceiling |
|---|---:|---:|---:|---:|---:|---:|
| Compact: exact name/address only | 754 | 22.35% | 4.84 | 31 / 52 | 315 | 0.431772 |
| Compact: name-only baseline | 1,673 | 49.59% | 46.20 | 61 / 100 | 24 | 0.652157 |
| Compact: combined name/address | **3,067** | **90.90%** | **92.19** | **100 / 100** | **0** | **0.954003** |
| Wider: name-only baseline | 1,811 | 53.68% | 87.35 | 111 / 150 | 24 | 0.682392 |
| Wider: combined name/address | **3,121** | **92.50%** | **143.25** | **150 / 150** | **0** | **0.962619** |

The compact combined method improves recovered-link counts for **627 businesses**, with no losses relative to its name-only baseline in this sample. The address-word route alone recovers 2,721 links (80.65%), demonstrating why name-only retrieval is insufficient here. Exact equality alone also misses most true pairs.

Widening adds **54 recovered links overall** for **51,065 additional candidate pairs**: 55.39% more later classifier work. It gains 56 true links and loses two because the larger combined pool is reordered and capped; bigger route limits do not guarantee a strict superset after final truncation.

These candidate lists contain many incorrect pairs. A later classifier must reject them, including all candidates for true no-match businesses. The perfect-classifier ceiling assumes all such decisions are correct; it is not achieved ML performance.

## Country results and hard-filter evidence

| Country | Queries | True links | Compact recall | Wider recall |
|---|---:|---:|---:|---:|
| India | 399 | 1,343 | 87.71% | 89.58% |
| US | 601 | 2,031 | 93.01% | 94.44% |

Across these 3,374 labeled links, **2,822 lack exact nonblank normalized name agreement** and **3,137 lack exact nonblank normalized address agreement**. Making either a mandatory identity filter would discard those links. No true country mismatch was observed in this sample; that does not establish a universal guarantee or France performance. The implementation therefore keeps arbitrary country labels and avoids a mandatory country filter.

There are seven links with target names of at most three normalized characters and 447 links involving a non-ASCII name. These counts describe the sample; they are not claims about every abbreviation or script. Synthetic regressions separately cover literal `NA`, Unicode marks, different scripts with shared address evidence, and unseen country labels.

## What the misses show

The compact run misses **307 links**: 305 are absent from all returned route lists; two are found by a route but removed by the final cap. Its uncapped returned-route union recovers 3,069 links (90.96%). Increasing only the final cap cannot recover the other 305.

The wider run misses **253 links**: 247 are absent from the returned routes, and six are lost at its final cap. The corresponding union recall is 92.68%. Further improvement needs better retrieval, not just exporting every current union member.

Inspection examples from the compact run:

- **`S1-110403115` → `S3-659065828`, Beth Zion:** names agree exactly, but the index contains 74 equal normalized names. This true target is number 61 in the exact-name route's ID order, outside its limit of 50. The wider configuration recovers it. This confirms a concrete per-route truncation failure.
- **`S1-104573195`, Family Partners of Little Rock:** several useful words are too common for the current word-frequency cutoff. Its address search retains only `taylor`; terms such as `little`, `rock`, `park`, and `loop` exceed the configured address frequency limit. The current selection can lose distinguishing combinations of otherwise common words. This is a development hypothesis for improving token combinations, not proof that one change will fix every miss.
- **`S1-101500008` → `S2-332386636`:** the target changes the name to `schóenborndelta.com` and introduces address spelling/number noise. Whole-word overlap is brittle even though the two address strings have high RapidFuzz similarity. Fuzzy string scoring cannot help a target that never reaches the shortlist.
- **`S1-126223960` → `S3-628968939`:** Latin-script `Shivam Producer` and Gujarati `શિવમ પ્રોડ્યુસર` have little character overlap; the compact run finds this target through a route but drops it at final ordering/capping. Cross-script pairs with partial addresses need further attention.

Among the 307 compact misses, 31 targets have blank addresses and 106 pairs involve a non-ASCII name. These groups overlap and describe symptoms, not exhaustive causal labels. Every missed link is retained in `missed_links.tsv`; no error was removed from the score denominator.

## Resource measurements

Environment: Linux/WSL2, Python 3.14.4, SQLite 3.46.1 with FTS5, and the existing pinned Python dependencies. The host reports approximately 7.6 GiB RAM. SQLite uses a 64 MiB page-cache setting and file-backed temporary work; peak process RSS includes Python, batch tables, and other allocations too.

| Work | Targets / queries | Recorded time | Peak process RSS | Database size |
|---|---:|---:|---:|---:|
| Restricted build pilot | 100,000 targets | 4.502 s | 149.35 MiB | 42.35 MiB |
| Full build | 10,320,219 targets | 338.778 s | 232.13 MiB | 3.981 GiB |
| Compact development benchmark | 1,000 queries | 129.014 s total; 115.395 s search | 225.85 MiB | Reuses index |
| Wider development benchmark | 1,000 queries | 113.927 s total; 96.047 s search | 264.98 MiB | Reuses index |

Compact search latency: mean 0.115 s, p95 0.254 s, maximum 0.413 s per query. Wider search latency: mean 0.096 s, p95 0.218 s, maximum 0.327 s. These are single-run observations with different OS cache conditions; the later wider run being faster does **not** establish that wider searches inherently cost less. Search timings include all four routes and final ordering, not isolated timing of each ablation.

Build time is the internal measurement before the final database fingerprint/metadata publication. Benchmark time includes query selection, index verification, retrieval, answer-key grading, and most report preparation; interpreter startup is excluded. Independent post-checks ran separately and are not part of the recorded benchmark RSS/time. No full-test runtime guarantee follows from these 1,000 queries.

The database contains raw records, normalized fields, exact lookup indexes, an external-content FTS5 index, and 2,195,047 token/field frequency entries. It avoids an in-memory matrix over all targets or a comparison of every query with every target. Querying visits relevant index postings and reorders only the bounded returned candidate pool. The benchmark still holds its selected query results/traces in memory; full inference batching/resumption remains Pass 6 work.

## Decision and provisional target

**Retain the compact configuration as the default for the initial Pass 5 classifier baseline.** It gives a measured starting point with 92,185 pairs for these 1,000 queries. Preserve the wider run as a recall/cost comparison rather than silently replacing the default: its 54 extra true links require 55.39% more candidate pairs.

After the compact result and before reading the wider result, we recorded the following provisional improvement target in `artifacts/retrieval/recall_cost_target.json`:

- At least **95% overall candidate recall**, with at least **92% for India**, on this fixed development sample.
- Mean and maximum candidate count at most **150**, and observed query p95 at most **0.5 seconds** on this machine.
- Index build peak process RSS at most **512 MiB** and persistent index size at most **6 GiB**.

**The recall targets remain unmet by both configurations.** The measured resource limits are satisfied. These are project targets for further development before scaling, not competition requirements or evidence about held-out evaluation quality. An initial Pass 5 classifier experiment is still useful with this documented baseline; scaling or claiming retrieval is solved is premature.

Future retrieval work should test common-word combinations, typo/concatenation tolerance, and ranking of cross-script pairs with address evidence. The measured cap losses show why increasing the final cap alone is insufficient. Continue to choose settings on development data and preserve the reserved evaluation set.

## Files, commands, and reproducibility

```bash
venv/bin/python -m src.retrieval_index \
    --output-dir artifacts/retrieval/pilot_100k --limit-per-source 50000
venv/bin/python -m src.retrieval_index \
    --output-dir artifacts/retrieval/train_full_v1
venv/bin/python -m src.retrieval_evaluation \
    --index-dir artifacts/retrieval/train_full_v1 \
    --output-dir artifacts/retrieval/development_v1
venv/bin/python -m src.retrieval_evaluation \
    --index-dir artifacts/retrieval/train_full_v1 \
    --output-dir artifacts/retrieval/development_wider_v1 \
    --per-channel 100 --max-candidates 150
```

These completed directories are protected; choose new output names when reproducing. The first command is a resource pilot only. It is explicitly marked restricted and rejected by the full benchmark. Teammates can skip it and build the full index once.

Each benchmark contains `candidate_pairs.tsv`, `name_only_candidates.tsv`, `exact_name_address_candidates.tsv`, `retrieval_trace.tsv`, `query_details.tsv`, `missed_links.tsv`, and `report.json`. These are development retrieval artifacts, not leaderboard files. There is no accepted-match output or real model artifact in Pass 4.

| Artifact | SHA-256 |
|---|---|
| Full target SQLite database | `80dac5f901a180153a908732a0573dacab3a080baf05e4ab195fd0117cb31c00` |
| Compact candidate file | `a4af8835c9679bd50fe5f95b74920a2a5ccff68bbb2b2b9bfdbf8e592d3f2b19` |
| Wider candidate file | `266950bae4b189a6c6366fcb681d6e3a5a41aeb9391ddee3455cb132d1927c72` |

Independent post-checks reloaded both candidate files, confirmed exact 1,000-query coverage, verified the existence of all 87,207 / 135,258 unique referenced targets, recomputed the recovered-link totals, and checked every recorded output hash. Their local record is `artifacts/retrieval/verification_checks.json`.

The fast tests cover input preservation, full indexing, duplicate rejection, deterministic limits/ties/order/chunk behavior, Unicode, empty results, restricted-index rejection, manifest/input fingerprint contracts, truth-independent candidate generation, final-cap losses, exact trace/output agreement, and failed-run cleanup. No real-data accuracy number is a hard-coded regression assertion.

Original sources and Pass 2 artifacts were not rewritten. Generated indexes/results stay under ignored `artifacts/`. No commit, push, model deployment, or competition upload was performed during this pass. See [brain.md](../brain.md) for the current project state.
