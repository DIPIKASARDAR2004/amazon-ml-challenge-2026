"""Inspect the seven original TSVs without materializing complete source tables."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import sys
from time import monotonic

# Retain the original direct-script entry point as well as python -m src.inspect_data.
if __name__ == "__main__" and not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.config import add_dataset_argument, positive_int
from src.data_loading import (
    GROUND_TRUTH_COLUMNS, SOURCE_COLUMNS, DataFormatError, blank_mask, iter_tsv,
)

# Count these literal strings as a diagnostic, never as missing values.
# This explicit list is not a claim about every parser's current null vocabulary.
TRACKED_LITERAL_TOKENS = ("NA", "N/A", "NULL", "null", "NaN", "nan", "None", "<NA>")


def memory_snapshot():
    """Read a Linux memory snapshot; report unavailable on unsupported systems."""
    try:
        values = {}
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            key, value = line.split(":", 1)
            if key in {"MemTotal", "MemAvailable", "SwapTotal", "SwapFree"}:
                values[key] = int(value.split()[0]) * 1024
        return {"source": "/proc/meminfo", "bytes": values}
    except (OSError, ValueError):
        return {"source": "unavailable", "bytes": {}}


def peak_process_memory_bytes():
    try:
        import resource
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return int(peak if sys.platform == "darwin" else peak * 1024)
    except (ImportError, OSError, AttributeError):
        return None


def inspect_source(path, *, source_number, chunk_size=50_000, limit=None):
    summary = {
        "records": 0, "countries": Counter(), "blank_names": 0,
        "blank_addresses": 0, "literal_name_tokens": Counter(),
        "literal_address_tokens": Counter(), "non_ascii_names": 0,
        "blank_ids": 0, "unexpected_id_prefix": 0,
    }
    for chunk in iter_tsv(path, SOURCE_COLUMNS, chunk_size=chunk_size, limit=limit):
        summary["records"] += len(chunk)
        summary["countries"].update(chunk["country"].value_counts().to_dict())
        summary["blank_names"] += int(blank_mask(chunk["business_name"]).sum())
        summary["blank_addresses"] += int(blank_mask(chunk["business_address"]).sum())
        summary["blank_ids"] += int(blank_mask(chunk["entity_id"]).sum())
        summary["unexpected_id_prefix"] += int((~chunk["entity_id"].str.startswith(f"S{source_number}-")).sum())
        summary["non_ascii_names"] += int(chunk["business_name"].str.contains(r"[^\x00-\x7f]", regex=True).sum())
        for column, key in [("business_name", "literal_name_tokens"), ("business_address", "literal_address_tokens")]:
            tokens = chunk.loc[chunk[column].isin(TRACKED_LITERAL_TOKENS), column]
            summary[key].update(tokens.value_counts().to_dict())
    return summary


def inspect_ground_truth(path, *, chunk_size=50_000, limit=None):
    distribution = Counter()
    for chunk in iter_tsv(path, GROUND_TRUTH_COLUMNS, chunk_size=chunk_size, limit=limit):
        text = chunk["matched_entity_ids"]
        counts = text.str.count(",").add(1).where(~blank_mask(text), 0)
        distribution.update({int(count): int(frequency) for count, frequency in counts.value_counts().items()})
    records = sum(distribution.values())
    links = sum(count * frequency for count, frequency in distribution.items())
    return {
        "records": records, "zero_matches": distribution.get(0, 0),
        "total_true_links": links, "mean_matches": links / records if records else 0,
        "maximum_matches": max(distribution, default=0),
        "match_count_distribution": dict(sorted(distribution.items())),
    }


def inspect_dataset(dataset_dir, *, chunk_size=50_000, limit=None):
    dataset_dir = Path(dataset_dir).resolve()
    sources = [(split, number, dataset_dir / split / f"{split}_source{number}.tsv")
               for split in ("train", "test") for number in (1, 2, 3)]
    truth_path = dataset_dir / "train" / "train_ground_truth.tsv"
    for path in [*(path for _, _, path in sources), truth_path]:
        if not path.is_file():
            raise FileNotFoundError(f"Required file not found: {path}")
    started = monotonic()
    report = {
        "schema_version": 1,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_dir": str(dataset_dir),
        "mode": "full" if limit is None else "sample",
        "rows_per_file_limit": limit,
        "chunk_size": chunk_size,
        "environment": {"python": platform.python_version(), "pandas": pd.__version__, "platform": platform.platform()},
        "memory_before": memory_snapshot(),
        "tracked_literal_tokens": list(TRACKED_LITERAL_TOKENS),
        "sources": {},
    }
    print(f"Inspection mode: {report['mode']} (row limit per file: {limit})", flush=True)
    for _, number, path in sources:
        summary = inspect_source(path, source_number=number, chunk_size=chunk_size, limit=limit)
        report["sources"][path.name] = summary
        print(
            f"{path.name}: {summary['records']:,} rows; "
            f"blank names={summary['blank_names']:,}; "
            f"literal NA names={summary['literal_name_tokens'].get('NA', 0):,}; "
            f"blank addresses={summary['blank_addresses']:,}", flush=True,
        )
    truth = inspect_ground_truth(truth_path, chunk_size=chunk_size, limit=limit)
    report["ground_truth"] = truth
    print(f"Ground truth: {truth['records']:,} rows; {truth['zero_matches']:,} with no matches; mean={truth['mean_matches']:.4f}", flush=True)
    report["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    report["elapsed_seconds"] = round(monotonic() - started, 3)
    report["memory_after"] = memory_snapshot()
    report["peak_process_memory_bytes"] = peak_process_memory_bytes()
    report["limitations"] = [
        "Counts describe only the consumed rows in sample mode.",
        "No full ID uniqueness, ground-truth referential integrity, or split audit.",
        "Ground-truth link counts assume nonblank lists contain comma-separated IDs; semantic list validation is a later pass.",
        "Literal token counts do not establish the tokens' intended meaning.",
        "System memory snapshots may differ from process/container memory limits.",
    ]
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    add_dataset_argument(parser)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--sample-rows", type=positive_int, default=1000, help="Inspect the first N data rows per file (default: 1000).")
    mode.add_argument("--full", action="store_true", help="Explicitly scan all rows in the seven original files.")
    parser.add_argument("--chunk-size", type=positive_int, default=50_000, help="Maximum rows per in-memory chunk (default: 50000).")
    parser.add_argument("--report", type=Path, help="Optional JSON report path, outside the dataset directory.")
    args = parser.parse_args(argv)
    dataset_dir = args.dataset_dir.resolve()
    if args.report and (args.report.resolve() == dataset_dir or dataset_dir in args.report.resolve().parents):
        parser.error("Write reports outside the dataset directory to keep source files unchanged.")
    try:
        report = inspect_dataset(dataset_dir, chunk_size=args.chunk_size, limit=None if args.full else args.sample_rows)
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            print(f"Saved report: {args.report.resolve()}")
    except (OSError, UnicodeError, DataFormatError) as exc:
        parser.exit(1, f"Inspection failed: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
