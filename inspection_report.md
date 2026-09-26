# Pass 1 Verification Record

Pass 1 completed on **2026-09-26**. The maintained [dataset findings](docs/dataset_insights.md) explain the statistics and the `NA` correction. The [README](README.md) provides setup and run commands for teammates.

This is the historical Pass 1 record. The later full training/split audit and scorer are documented in the [Pass 2 verification record](docs/pass2_verification.md).

## What was verified

| Check | Result |
|---|---|
| Fast regression suite | **39 passed** in 4.07 seconds in the final run |
| Tiny synthetic dataset inspection | Expected blanks, literal tokens, countries, and empty match lists; JSON output produced |
| Real-data sample inspection | 1,000 rows per original file; explicitly labeled as a sample |
| Full source and ground-truth inspection | All documented row/country/blank/token counts and ground-truth statistics reproduced |
| Runtime/development dependency pins | Match the installed environment; pip's dependency check reports no broken requirements |
| Candidate demo smoke check | Two queries against 1,000 targets per source; executed successfully, without an accuracy claim |
| Split and training demo smoke checks | Executed successfully against a temporary copy of tiny fixtures, outside the real dataset |
| Git ignore review | Real data, environments, generated reports, and OS metadata excluded; code, docs, challenge instructions, and synthetic fixtures remain visible |

No real source files or existing real split files were written by these checks. The supplied submission validator was not changed. This is not a completed split audit, official scorer, ML validation, or submission pipeline.

## Full inspection measurement

Command, run from the repository root:

```bash
venv/bin/python -m src.inspect_data \
    --full --chunk-size 50000 \
    --report artifacts/inspection/full.json
```

- Started: **2026-09-26 13:30:49 UTC**.
- Finished: **2026-09-26 13:31:39 UTC**.
- Elapsed inspection time: **49.918 seconds**; excludes interpreter startup and report serialization.
- Environment: Linux/WSL2, CPython **3.14.4**, Pandas **3.0.6**.
- Chunk size: **50,000 data rows**.
- Peak process resident memory reported by the OS: **119,377,920 bytes**, approximately **113.85 MiB**.
- Host memory snapshot before scanning: **8,144,941,056 bytes total**, **6,471,262,208 bytes available**.

These measurements describe this run, not a guaranteed budget or speed on another machine. Host memory and process/container limits can differ. The inspector records unavailable measurements explicitly on unsupported systems.

## Reproduced totals

| Measurement | Result |
|---|---:|
| Original source records, train and test combined | 24,229,173 |
| Empty/whitespace-only business names | 0 |
| Business names exactly equal to literal `NA` | 120 |
| Empty/whitespace-only addresses | 610,389 |
| Training ground-truth rows | 2,206,821 |
| Empty training match lists | 123,247 |
| Training true links | 7,638,365 |
| Mean true links per Source 1 business | 3.4612526344 |
| Maximum observed true links | 11 |

File-level counts and countries are in [dataset findings](docs/dataset_insights.md) and the generated JSON. The JSON is an ignored local artifact and can be regenerated with the command above.

## What this establishes

The tests and inspection protect text preservation, explicit blank handling, schema/row validation, chunk limits, safe imports, configurable paths, and sample/full reporting. The inspector now has one shared implementation; both original command paths delegate to it.

The counts of 2, 13, 46, and 59 previously described as missing names are literal `NA` strings in the four target source files. They remain text in the new loader. Training labels link several to names such as National Alliance Inc.; their intended meaning is not inferred from a Pandas default.

## What to review next

Follow the README's test and fixture commands. Pass 2 has since verified training ID/label/split integrity and implemented the official scorer; see its separate record above. The candidate and model demonstrations still have the limitations recorded in the [implementation plan](implementation_plan.md).
