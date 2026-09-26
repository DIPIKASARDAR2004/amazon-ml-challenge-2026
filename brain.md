# Project brain

This is the short project memory for teammates returning to the repository. Detailed evidence belongs in the linked verification records. Updated: **2026-09-26**.

## Objective and current state

For each Source 1 business, find zero, one, or many matching Source 2/3 records using only the provided challenge data. Preserve every required Source 1 row. The competition rewards accurate matches and penalizes false matches through macro F0.5.

**Passes 1–4 are complete, with 190 passing regression cases.** Pass 4 measured real development retrieval against the full training target pool. Pass 5 has not started. There is no trustworthy real-data classifier score or complete competition submission yet.

## Read first

1. [README](README.md): environment and runnable commands.
2. [Team guide](docs/team_guide.md): beginner terminology and workflow.
3. [Implementation plan](implementation_plan.md): pass boundaries and checkpoints.
4. [Pass 3 walkthrough](docs/pass3_walkthrough.md): one invented business through a complete model pipeline.
5. [Pass 4 walkthrough](docs/pass4_walkthrough.md) and [verification record](docs/pass4_verification.md): real-data retrieval comparison, resource measurements, and error inspection.
6. [Hardware plan](docs/hardware_plan.md): local capacity, pair-count projections, and when cloud compute would help.

## Decisions to preserve

- Keep original strings, IDs, Unicode, and empty fields. The 120 literal `NA` names are not blank fields. Training evidence supports abbreviation/noise in some cases; it does not establish one universal meaning. Normalize into separate columns.
- Split by Source 1 business, not by pairs. Pass 2 preserved the old training split and saved development/evaluation selections with seed 42. Use development to choose settings; reserve evaluation until settings are frozen.
- Query sampling selects Source 1 businesses only. Real retrieval must still search the complete training Source 2/3 pool. Never select targets or insert missed candidates using the answer key.
- Build training negatives from incorrect retrieved pairs; never randomly assign a known positive a negative label.
- Use the same ordered features during training and inference. Save preprocessing/retrieval settings and feature order alongside the model; verify reload equivalence.
- An empty match list is a valid answer. A missing Source 1 row is an error. Predictions must be contained in the exact final candidate set scored by the model.
- Candidate recall, a theoretical perfect-classifier score, a synthetic fixture score, and real model accuracy are different measurements. Label them explicitly.
- No external business lookups, geocoding, or identity augmentation. Technical documentation may be consulted; entity resolution uses the supplied data.
- Work one pass at a time, run meaningful regressions, and explain inspectable evidence before expanding scope.

## Data and evaluation facts

| Item | Count / meaning |
|---|---|
| Original source records across train/test | 24,229,173 |
| Training Source 1 businesses | 2,206,821 |
| Training Source 2/3 target pool | 10,320,219 |
| Training true links | 7,638,365 |
| No-match training businesses | 123,247 |
| Real train / development / evaluation businesses | 1,986,139 / 110,340 / 110,342 |
| Saved small query samples | 5,000 / 1,000 / 1,000 |
| Competition test Source 1 businesses | 1,732,544 |
| Training countries | India and US; France is additionally present in test |

For nonempty truth, each business scores `5 TP / (5 TP + 4 FP + FN)`. Empty truth plus empty prediction scores 1; empty truth plus any prediction scores 0. Average over every selected business. True matches absent from retrieval remain false negatives.

## Code map and technology

- Python runs the workflow; Pandas holds bounded text batches, NumPy holds numerical arrays, and RapidFuzz computes string clues.
- `src/data_loading.py`, `normalization.py`: strict string reading and separate normalized fields.
- `src/splits.py`, `evaluation_io.py`, `scoring.py`: audited splits, identity/file contracts, official scoring.
- `src/candidates.py`: tiny Pass 3 TF-IDF retrieval using scikit-learn; not a full-pool index.
- `src/features.py`, `pair_labels.py`, `model_io.py`: shared pair features, labels, LightGBM training and native-model persistence.
- `src/submission.py`, `pipeline.py`: strict output behavior and the synthetic connected runner.
- `src/retrieval_index.py`: chunked SQLite target storage, exact indexes, and FTS5 word retrieval.
- `src/retrieval_evaluation.py`: fixed development sample, full-index checks, candidate comparison, resource measurements, and missed-link reports.
- Root `candidate_generation.py`, `train_model.py`, and `split_validation.py` are historical demonstrations. The old trainer still has unsafe random-negative methodology; use the maintained `src/` workflow. The old split generator can replace derived files, so it is not the audit command.

## Local artifacts and reproducibility

`dataset/`, `venv/`, and `artifacts/` are ignored by Git. The verified environment is Linux/Python 3.14.4 with pinned packages in `requirements*.txt`. Small synthetic TSV fixtures under `tests/fixtures/` are versioned inputs; teammates regenerate the artifacts locally.

- `artifacts/evaluation/pass2_v1/`: immutable experiment ID lists, audit, manifest, and hashes.
- `artifacts/pipeline/pass3_v1/`: invented model, pair traces, both valid fixture output files, and report. Its recorded macro F0.5 is 0.875; this is not real-data accuracy.
- `artifacts/retrieval/pilot_100k/`: explicitly restricted index used to measure build cost; prohibited in the full benchmark.
- `artifacts/retrieval/train_full_v1/`: reusable complete training target index.
- `artifacts/retrieval/development_v1/`: candidate-only development benchmark and error analysis.

Use new version directories when rerunning; completed artifacts are protected against replacement. Fingerprints tie runs to the exact sources, index, settings, and query sample. The retrieval benchmark generates candidate files for later scoring; it does not generate final accepted matches.

## Latest measured result and decision

The fixed development sample has 3,374 true links. The complete target index took 338.8 seconds to build, used 232.13 MiB peak process RSS, and occupies 3.98 GiB on disk.

| Retrieval setting | Candidate recall | Mean candidates | Recorded query time for 1,000 businesses |
|---|---:|---:|---:|
| Compact name-only baseline | 49.59% | 46.20 | Not timed separately |
| Compact combined name/address: 50 per route, 100 final | **90.90%** | **92.19** | 115.40 s across all routes/order |
| Wider combined name/address: 100 per route, 150 final | **92.50%** | **143.25** | 96.05 s across all routes/order |

The later run had different cache conditions; its lower time is not evidence that wider search is inherently faster. Both index construction and benchmark costs are measured independently of the fast regression suite.

**Keep the compact configuration as the initial Pass 5 default.** Widening gains 54 true links overall while adding 55.39% more candidate pairs. Preserve the wider run at `artifacts/retrieval/development_wider_v1/` as a comparison. The compact run misses 307 links (305 not returned by routes, two lost at the final cap); wider still misses 253.

Provisional improvement targets before scaling: at least **95% overall / 92% India candidate recall**, mean/max candidates at most 150, observed query p95 at most 0.5 seconds, index build RSS at most 512 MiB, and disk at most 6 GiB. **Both recall targets remain unmet.** Compact India recall is 87.71%, wider 89.58%. Resource limits were met. These targets were recorded after the compact baseline and before reading the wider comparison; they are project goals, not competition rules.

Next retrieval hypotheses include common-word combinations, typo/concatenation tolerance, and cross-script pairs with partial address evidence. Raising only the final cap is insufficient. A controlled Pass 5 classifier baseline can proceed with these limitations documented; no claim that retrieval or real matching quality is solved is justified.

**Hardware decision:** Begin the small Pass 5 baseline locally. The user's Windows screenshot confirms **16 GB physical RAM** and approximately **190 GB free physical storage** (477 GB total minus 287 GB used). WSL currently exposes about 7.6 GiB RAM and eight logical CPUs, consistent with its default half-RAM limit; Linux's reported 923 GiB free virtual filesystem space is not the physical disk budget. Real training memory is unmeasured. At the current candidate density, 5,000 queries imply about 461,000 pairs and 35 MiB of numeric feature payload, plus substantial object/model overhead. A rough sequential extrapolation gives 55.5 hours for full-test retrieval alone, so batching, checkpointing, and throughput matter alongside RAM. Kaggle CPU notebooks are an optional larger-memory environment, not a necessary next step. See the [hardware plan](docs/hardware_plan.md) for assumptions and current provider documentation.

## Commands to remember

Run from the repository root:

```bash
venv/bin/python -m pytest -q
venv/bin/python -m src.pipeline --output-dir artifacts/pipeline/my_fixture_run
venv/bin/python -m src.retrieval_index --output-dir artifacts/retrieval/my_full_index
venv/bin/python -m src.retrieval_evaluation \
    --index-dir artifacts/retrieval/train_full_v1 \
    --output-dir artifacts/retrieval/my_development_run
```

The first two commands use tiny synthetic data. The third builds an index over all training targets. The fourth reuses an index and the saved development query IDs. Read the measured resource costs and error analysis before increasing sizes or changing limits.

## Submission and next work

The leaderboard receives only the real `matching_results.tsv`. The separate final ZIP includes that file, `candidate_pairs.tsv`, code, requirements, and documentation. The supplied validator's default skips target existence; project checks also enforce requirements that it merely warns about. The fixture files must never be uploaded.

Next, review Pass 4's recall/cost tradeoff and missed links. Pass 5 will retrieve realistic training pairs, label them from training truth, train LightGBM, tune acceptance on development, freeze settings, and evaluate once on the reserved businesses. Full test inference, resumability, final packaging, and upload are later passes.

Git includes source code, documentation, dependency pins, the supplied validator, and small synthetic fixtures. Real data, environments, indexes, generated outputs, archives, and machine-local settings remain ignored. Review the staged files before committing; teammates regenerate ignored artifacts locally.
