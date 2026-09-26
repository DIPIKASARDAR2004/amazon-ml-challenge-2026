# Dataset Insights and Findings

This is the shared record of what we have measured, what we infer, and what remains unknown. Start with the [team guide](team_guide.md) if entity resolution is new to you. The [implementation plan](../implementation_plan.md) describes what we will build next.

**Evidence:** The inspection on 2026-09-26 streamed all six original source files and the training ground truth in chunks of 50,000 rows. A follow-up check independently searched the raw TSVs for `NA`, reproduced Pandas parsing on a tiny in-memory example, and traced all 15 training `NA` targets through ground truth. No dataset contents were changed.

## 1. Our goal and file meanings

For every Source 1 business, find all records in Sources 2 and 3 that represent the same business. Zero, one, or multiple matches are possible. Names and addresses can differ even for a correct match.

Each source TSV has four columns: `entity_id`, `business_name`, `business_address`, and `country`. The training ground truth has `source1_entity_id` and `matched_entity_ids`; an empty match list means no matches. Source prefixes identify record origin, not shared business identity.

## 2. Verified record counts

| Original source file | Records |
|---|---:|
| `train_source1.tsv` | 2,206,821 |
| `train_source2.tsv` | 5,034,616 |
| `train_source3.tsv` | 5,285,603 |
| `test_source1.tsv` | 1,732,544 |
| `test_source2.tsv` | 4,887,273 |
| `test_source3.tsv` | 5,082,316 |
| **Total source records** | **24,229,173** |

The total excludes ground-truth rows and derived train/validation split files. Counting those again would inflate the dataset size.

Each final output file must contain 1,732,544 data rows, covering every test Source 1 ID exactly once, plus a header.

## 3. Countries and unseen formats

| File | US | India | France |
|---|---:|---:|---:|
| `train_source1.tsv` | 1,323,633 | 883,188 | 0 |
| `train_source2.tsv` | 3,016,817 | 2,017,799 | 0 |
| `train_source3.tsv` | 3,170,056 | 2,115,547 | 0 |
| `test_source1.tsv` | 663,106 | 809,986 | 259,452 |
| `test_source2.tsv` | 1,871,330 | 2,312,565 | 703,378 |
| `test_source3.tsv` | 1,945,701 | 2,405,000 | 731,615 |

**Observed:** France occurs in testing and not in training. Other examples contain Indian scripts, accented text, and mixed-language addresses.

**Implication:** Preserve Unicode and handle arbitrary country labels. Validation on US and India cannot establish French matching accuracy. A French test smoke run can check execution and formatting only.

## 4. Correction: blank names, literal `NA`, and parser behavior

These are different concepts:

- **Raw blank field:** an empty field, or a field containing only whitespace. We counted these explicitly.
- **Literal text:** characters actually present in the file, such as `NA`.
- **Parsed missing value:** a special value a reader may create according to its settings, even when the original field was nonempty.

| File | Blank name fields | Names exactly `NA` | Blank address fields |
|---|---:|---:|---:|
| `train_source1.tsv` | 0 | 0 | 0 |
| `train_source2.tsv` | 0 | 2 | 168,967 |
| `train_source3.tsv` | 0 | 13 | 175,916 |
| `test_source1.tsv` | 0 | 0 | 0 |
| `test_source2.tsv` | 0 | 46 | 129,408 |
| `test_source3.tsv` | 0 | 59 | 136,098 |
| **Total** | **0** | **120** | **610,389** |

The independent raw-text search confirmed all 120 exact `NA` names. In installed Pandas 3.0.6, a tiny TSV containing a name `NA` produced a missing value with default missing-value parsing, even with `dtype=str`. With `keep_default_na=False`, it remained the literal string `NA`. Thus string dtype alone does not prevent the conversion.

This explains why the older report counted 2, 13, 46, and 59 missing names while the raw-field inspection found no blanks. Neither count establishes the intended meaning of the text.

### Is `NA` intentional?

**Evidence:** All 15 training target records named `NA` have a ground-truth link to a Source 1 business. Several linked names fit an abbreviation pattern:

| Source 1 ID and name | Labeled target ID | Raw target name |
|---|---|---|
| `S1-15232735` — National Alliance Inc. | `S3-263823274` | `NA` |
| `S1-757103645` — Nayana Air Private Limited | `S3-761073618` | `NA` |
| `S1-401007482` — Novex Aerospace LLC | `S3-749220699` | `NA` |
| `S1-172444211` — Novara Armour L.L.C. | `S2-982925237` | `NA` |
| `S1-679088438` — New Agro Private Limited | `S3-42389827` | `NA` |
| `S1-684348466` — Niva | `S2-599473918` | `NA` |

For example, National Alliance Inc. and its `NA` target both have the exact address `519 Odeneal Street, Dallas, TX`. The target occurs at line 3,155,295 of the current `dataset/train/train_source3.tsv`, counting the header as line 1. IDs are the stable way to locate it if file ordering changes.

**Interpretation:** Abbreviation/noise is a well-supported explanation for several training examples. Treating every `NA` as “not available” would discard observed text that can carry matching information. The Niva example also shows why we should not claim every case is simply the first letters of two words.

**Unknown:** The supplied README describes abbreviation and spelling noise but does not define the exact token `NA`. We have not inspected a data-generation specification. Test data has no public ground truth, so we cannot establish the meaning of all 105 test occurrences through labeled links.

**Correction to earlier wording:** Calling all these values “placeholders” was unsupported. The defensible claim is that 120 raw name fields contain `NA`, and default Pandas parsing would convert them to missing values. We should preserve them as observed name text.

### Handling implemented in Pass 1

The shared loader preserves the original fields and computes blank indicators separately. It validates each TSV row with Python's standard CSV reader before creating a Pandas table, avoiding implicit null conversion and silent column padding. Normalized feature fields belong to a later pass. Do not erase `NA`, automatically expand it, or infer identity solely from this short token.

The first regression tests now verify that raw `NA` survives loading and differs from an empty field. Other literal tokens remain observable too. Embedded words inside an address are a separate issue: a blank-field count is not an audit of every possibly noisy address component.

To verify the National Alliance example locally, these read-only searches show the reference, the target, and the supplied matching relationship:

```bash
rg -n '^S1-15232735\t' dataset/train/train_source1.tsv
rg -n '^S3-263823274\t' dataset/train/train_source3.tsv
rg -n '^S1-15232735\t' dataset/train/train_ground_truth.tsv
```

## 5. Other labeled examples and their implications

These are supplied training labels, not guesses based on names:

| Source 1 | Labeled target | What this illustrates |
|---|---|---|
| `S1-773889195` — Prime Money | `S2-970528089` — @primemoney | Handle-like names, removed spaces, address reordering |
| `S1-564729135` — Dream Construction Limited | `S2-327309238` — డ్రీమ్ కన్‌స్ట్రక్షన్ లిమిటెడ్ | Different writing systems for a labeled match |
| `S1-82053739` — Smart Healthcare Private Limited | `S3-250737963` — Kelonyla | Very different names with an identical address |

**Implication:** Name-only character similarity can miss correct links. Retrieval should consider multiple sources of evidence, including addresses. A matching address or an abbreviation still does not prove identity by itself; unrelated businesses can share them. These few examples illustrate failure modes, not their frequency across the dataset.

## 6. Training ground truth

| Measurement | Verified value |
|---|---:|
| Source 1 ground-truth rows | 2,206,821 |
| True links to Source 2/3 | 7,638,365 |
| Source 1 businesses with zero matches | 123,247, about 5.58% |
| Mean links per business | 3.4613 |
| Maximum links per business | 11 |
| Businesses with exactly 11 links | 37 |

Eleven is an observed maximum, not a guaranteed test-set limit or candidate-list budget. We need enough candidates to include correct matches as well as plausible incorrect alternatives. Correctly returning no matches matters too.

## 7. Resources and limits of the inspection

The original source TSVs occupy several gigabytes on disk. In-memory tables and retrieval indexes can require substantially more space. Chunked reading bounds ingestion memory; it does not make exhaustive pair comparisons affordable.

Earlier RAM figures came from inconsistent or unrecorded environments and are not a fixed project budget. The read-only Linux inspection session reported about 7.59 GiB total and 6.14 GiB available at that moment. Available memory changes, and the earlier 5.85 GiB total cannot be reconciled from the historical notes alone. Remeasure on the actual run environment and record the time before scaling.

The full scan checked schemas, record/country counts, blank fields, exact parser-sensitive name tokens, ID blanks/prefixes, and match-list lengths. No blank IDs or unexpected source prefixes were observed. The follow-up trace checked all 15 training `NA` targets specifically.

**Pass 2 follow-up:** All original training source IDs are unique, all 7,638,365 ground-truth target links exist, and each target belongs to at most one reference business. The existing train/validation split is disjoint and complete; its Source 1 fields and label sets agree with the originals. See the [Pass 2 verification record](pass2_verification.md).

**Still unverified:** equivalent full test-ID integrity checks, address/name truthfulness, and model generalization. The training audit establishes consistency of supplied labels, not their real-world correctness. No trustworthy ML accuracy evaluation has been completed.

Pass 1's maintained inspector reproduced these statistics using `venv/bin/python -m src.inspect_data --full --report artifacts/inspection/full.json`. The [verification record](../inspection_report.md) contains the run environment, time, and memory measurement. The original temporary JSON is not required by the project. See the [README](../README.md) for a tiny fixture and a sample command to run first.

## Pass 4 development retrieval measurement

The complete training target index now contains all 10,320,219 Source 2/3 records. A benchmark on the exact 1,000 development queries saved in Pass 2 found 3,374 true links. Name-only retrieval recovered 49.59%; combining name/address routes recovered 90.90% with 92.19 candidates per query on average. A wider comparison reached 92.50% with 143.25 candidates. These are development candidate-recall measurements, not model accuracy or statistics over all Source 1 businesses.

The [Pass 4 record](pass4_verification.md) documents country breakdowns, missed links, resource cost, the wider-list tradeoff, and provisional recall targets that remain unmet. Neither exact name nor exact address equality can be a mandatory filter for this sample: 2,822 / 3,137 of the 3,374 true links respectively lack that nonblank normalized agreement. No country mismatch was observed in the sample; that is not a universal test-data guarantee.
