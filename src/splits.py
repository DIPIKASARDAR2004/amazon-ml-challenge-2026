"""Audit existing training splits and freeze reproducible experiment query IDs."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import csv
import hashlib
import json
from pathlib import Path
import platform
import sqlite3
import tempfile
from time import monotonic

from src.config import PROJECT_ROOT, add_dataset_argument, positive_int
from src.data_loading import GROUND_TRUTH_COLUMNS, SOURCE_COLUMNS, DataFormatError, iter_tsv
from src.evaluation_io import ID_COLUMNS, fingerprint_file, parse_targets, validate_id
from src.inspect_data import memory_snapshot, peak_process_memory_bytes


SCHEMA = """
CREATE TABLE reference (
    id TEXT PRIMARY KEY, country TEXT NOT NULL, fingerprint BLOB NOT NULL,
    rank BLOB NOT NULL, truth TEXT, link_count INTEGER, legacy TEXT,
    split_truth_seen INTEGER NOT NULL DEFAULT 0, role TEXT
) WITHOUT ROWID;
CREATE TABLE targets (id TEXT PRIMARY KEY) WITHOUT ROWID;
CREATE TABLE links (target_id TEXT PRIMARY KEY, source1_id TEXT NOT NULL) WITHOUT ROWID;
"""


def _write_json(path, payload):
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _record_hash(row):
    return hashlib.sha256(json.dumps(row, ensure_ascii=False, separators=(",", ":")).encode("utf-8")).digest()


def _insert_unique(connection, statement, rows, context):
    try:
        connection.executemany(statement, rows)
    except sqlite3.IntegrityError as exc:
        raise DataFormatError(f"{context}: duplicate ID or conflicting ground-truth target ownership") from exc


def _load_reference(connection, path, chunk_size, seed):
    count = 0
    for chunk in iter_tsv(path, SOURCE_COLUMNS, chunk_size=chunk_size):
        rows = []
        for row in chunk.itertuples(index=False, name=None):
            entity_id = validate_id(row[0], context=str(path))
            rank = hashlib.sha256(f"{seed}\0{entity_id}".encode("utf-8")).digest()[:16]
            rows.append((entity_id, row[3], _record_hash(row), rank))
        _insert_unique(connection, "INSERT INTO reference(id,country,fingerprint,rank) VALUES (?,?,?,?)", rows, str(path))
        count += len(rows)
        connection.commit()
    return count


def _load_targets(connection, path, number, chunk_size):
    count = 0
    for chunk in iter_tsv(path, SOURCE_COLUMNS, chunk_size=chunk_size):
        rows = [(validate_id(value, (f"S{number}-",), context=str(path)),) for value in chunk["entity_id"]]
        _insert_unique(connection, "INSERT INTO targets VALUES (?)", rows, str(path))
        count += len(rows)
        connection.commit()
        if count % 1_000_000 == 0:
            print(f"  {path.name}: indexed {count:,} IDs", flush=True)
    return count


def _load_truth(connection, path, chunk_size):
    count = 0
    for chunk in iter_tsv(path, GROUND_TRUTH_COLUMNS, chunk_size=chunk_size):
        rows, links = [], []
        for entity_id, text in chunk.itertuples(index=False, name=None):
            validate_id(entity_id, context=str(path))
            targets = parse_targets(text, context=f"{path}: {entity_id}")
            rows.append((",".join(sorted(targets)), len(targets), entity_id))
            links.extend((target, entity_id) for target in targets)
        cursor = connection.executemany(
            "UPDATE reference SET truth=?, link_count=? WHERE id=? AND truth IS NULL", rows,
        )
        if cursor.rowcount != len(rows):
            raise DataFormatError(f"{path}: missing or duplicate ground-truth Source 1 ID")
        _insert_unique(connection, "INSERT INTO links VALUES (?,?)", links, "ground-truth target ownership")
        count += len(rows)
        connection.commit()
    missing = connection.execute("SELECT id FROM reference WHERE truth IS NULL LIMIT 5").fetchall()
    if missing:
        raise DataFormatError(f"ground-truth coverage: missing Source 1 IDs, e.g. {missing}")
    unknown = connection.execute(
        "SELECT l.target_id FROM links l LEFT JOIN targets t ON t.id=l.target_id WHERE t.id IS NULL LIMIT 5"
    ).fetchall()
    if unknown:
        raise DataFormatError(f"ground-truth target IDs do not exist in training Source 2/3: {unknown}")
    return count


def _check_source_split(connection, path, label, chunk_size):
    count = 0
    for chunk in iter_tsv(path, SOURCE_COLUMNS, chunk_size=chunk_size):
        rows = []
        for row in chunk.itertuples(index=False, name=None):
            validate_id(row[0], context=str(path))
            rows.append((label, row[0], _record_hash(row)))
        cursor = connection.executemany(
            "UPDATE reference SET legacy=? WHERE id=? AND fingerprint=? AND legacy IS NULL", rows,
        )
        if cursor.rowcount != len(rows):
            raise DataFormatError(f"{path}: split has duplicate, overlapping, unknown, or changed Source 1 records")
        count += len(rows)
        connection.commit()
    return count


def _check_truth_split(connection, path, label, chunk_size):
    count = 0
    for chunk in iter_tsv(path, GROUND_TRUTH_COLUMNS, chunk_size=chunk_size):
        rows = []
        for entity_id, text in chunk.itertuples(index=False, name=None):
            validate_id(entity_id, context=str(path))
            targets = parse_targets(text, context=f"{path}: {entity_id}")
            rows.append((entity_id, ",".join(sorted(targets)), label))
        cursor = connection.executemany(
            "UPDATE reference SET split_truth_seen=1 WHERE id=? AND truth=? AND legacy=? AND split_truth_seen=0", rows,
        )
        if cursor.rowcount != len(rows):
            raise DataFormatError(f"{path}: duplicate, extra, or changed split ground-truth records")
        count += len(rows)
        connection.commit()
    return count


def _audit(connection, dataset_dir, split_dir, chunk_size, seed):
    paths = {
        "source1": dataset_dir / "train/train_source1.tsv",
        "source2": dataset_dir / "train/train_source2.tsv",
        "source3": dataset_dir / "train/train_source3.tsv",
        "ground_truth": dataset_dir / "train/train_ground_truth.tsv",
        "legacy_train": split_dir / "train/train_source1_sampled.tsv",
        "legacy_validation": split_dir / "val/val_source1.tsv",
        "legacy_train_truth": split_dir / "train/train_ground_truth_sampled.tsv",
        "legacy_validation_truth": split_dir / "val/val_ground_truth.tsv",
    }
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(f"Required file not found: {path}")
    counts = {}
    counts["source1"] = _load_reference(connection, paths["source1"], chunk_size, seed)
    print(f"Reference: {counts['source1']:,} unique Source 1 IDs", flush=True)
    for number in (2, 3):
        counts[f"source{number}"] = _load_targets(connection, paths[f"source{number}"], number, chunk_size)
        print(f"Source {number}: {counts[f'source{number}']:,} unique target IDs", flush=True)
    counts["ground_truth"] = _load_truth(connection, paths["ground_truth"], chunk_size)
    counts["true_links"] = connection.execute("SELECT COUNT(*) FROM links").fetchone()[0]
    print(f"Ground truth: {counts['true_links']:,} unique, existing target links", flush=True)
    for role in ("train", "validation"):
        counts[f"legacy_{role}"] = _check_source_split(connection, paths[f"legacy_{role}"], role, chunk_size)
    missing = connection.execute("SELECT id FROM reference WHERE legacy IS NULL LIMIT 5").fetchall()
    if missing:
        raise DataFormatError(f"split coverage: original Source 1 records absent from both partitions: {missing}")
    for role in ("train", "validation"):
        counts[f"legacy_{role}_truth"] = _check_truth_split(connection, paths[f"legacy_{role}_truth"], role, chunk_size)
    missing = connection.execute("SELECT id FROM reference WHERE split_truth_seen=0 LIMIT 5").fetchall()
    if missing:
        raise DataFormatError(f"split ground-truth coverage: missing rows for {missing}")
    if not counts["legacy_train"] or not counts["legacy_validation"]:
        raise DataFormatError("Both legacy split partitions must contain businesses")
    inputs = {name: {"path": str(path), **fingerprint_file(path)} for name, path in paths.items()}
    return {
        "status": "pass", "counts": counts, "inputs": inputs,
        "legacy_validation_fraction": counts["legacy_validation"] / counts["source1"],
        "checks": {
            "source_id_uniqueness_and_prefixes": True, "complete_original_ground_truth": True,
            "target_references_exist": True, "target_has_at_most_one_reference_business": True,
            "legacy_source_partitions_disjoint_and_complete": True,
            "legacy_source_fields_unchanged": True,
            "legacy_truth_aligned_and_unchanged_as_sets": True,
        },
        "scope": "Original training sources/labels and existing derived split; no test dataset read.",
        "limitations": ["Does not verify the semantic correctness of the supplied labels.",
                        "Audits existing split contents; does not prove which historical seed generated them."],
    }


def _assign_roles(connection, development_fraction, chunk_size):
    warnings = []
    connection.execute("CREATE INDEX legacy_rank ON reference(legacy,country,(link_count=0),rank,id)")
    connection.execute("UPDATE reference SET role='train' WHERE legacy='train'")
    strata = connection.execute(
        "SELECT country,link_count=0,COUNT(*) FROM reference WHERE legacy='validation' GROUP BY country,link_count=0 ORDER BY country,link_count=0"
    ).fetchall()
    for country, singleton, count in strata:
        if count == 1:
            development_count = int(development_fraction >= 0.5)
            warnings.append(f"Only one validation business in country={country!r}, singleton={bool(singleton)}; cannot represent it in both partitions.")
        else:
            development_count = min(count - 1, max(1, int(count * development_fraction)))
        cursor = connection.execute(
            "SELECT id FROM reference WHERE legacy='validation' AND country=? AND (link_count=0)=? ORDER BY rank,id",
            (country, singleton),
        )
        position = 0
        while rows := cursor.fetchmany(chunk_size):
            updates = [("development" if position + index < development_count else "evaluation", row[0]) for index, row in enumerate(rows)]
            connection.executemany("UPDATE reference SET role=? WHERE id=?", updates)
            position += len(rows)
        connection.commit()
    connection.execute("DROP INDEX legacy_rank")
    connection.execute("CREATE INDEX role_rank ON reference(role,country,(link_count=0),rank,id)")
    connection.commit()
    return warnings


def _sample_quotas(strata, requested):
    total = sum(row[2] for row in strata)
    desired = min(total, requested)
    if desired < len(strata):
        raise ValueError(f"Sample size {requested} is too small to represent {len(strata)} available strata")
    quotas = {(country, singleton): 1 for country, singleton, _ in strata}
    capacity = total - len(strata)
    remaining = desired - len(strata)
    if capacity:
        remainders = []
        for country, singleton, count in strata:
            extra, remainder = divmod(remaining * (count - 1), capacity)
            quotas[(country, singleton)] += extra
            remainders.append((-remainder, country, singleton))
        left = desired - sum(quotas.values())
        for _, country, singleton in sorted(remainders)[:left]:
            quotas[(country, singleton)] += 1
    return quotas


def _strata_summary(strata):
    return [{"country": country, "singleton": bool(singleton), "rows": count} for country, singleton, count in strata]


def _write_ids(path, rows):
    count = 0
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
        writer.writerow(ID_COLUMNS)
        for row in rows:
            writer.writerow(row)
            count += 1
    return {"rows": count, **fingerprint_file(path)}


def _export(connection, destination, sample_sizes):
    all_strata, quotas = {}, {}
    for role, requested in sample_sizes.items():
        strata = connection.execute(
            "SELECT country,link_count=0,COUNT(*) FROM reference WHERE role=? GROUP BY country,link_count=0 ORDER BY country,link_count=0", (role,),
        ).fetchall()
        if not strata:
            raise ValueError(f"The {role} partition is empty; use a larger validation set or different development fraction")
        all_strata[role], quotas[role] = strata, _sample_quotas(strata, requested)
    files = {}
    for role in sample_sizes:
        name = f"{role}_ids.tsv"
        files[name] = _write_ids(destination / name, connection.execute("SELECT id FROM reference WHERE role=? ORDER BY id", (role,)))
        files[name]["strata"] = _strata_summary(all_strata[role])
        selected = []
        for (country, singleton), quota in quotas[role].items():
            selected.extend(connection.execute(
                "SELECT id FROM reference WHERE role=? AND country=? AND (link_count=0)=? ORDER BY rank,id LIMIT ?",
                (role, country, singleton, quota),
            ).fetchall())
        name = f"sample_{role}_ids.tsv"
        files[name] = _write_ids(destination / name, sorted(selected))
        files[name]["strata"] = _strata_summary([(country, singleton, count) for (country, singleton), count in quotas[role].items()])
    return files


def prepare_experiment(dataset_dir, output_dir, *, split_dir=None, seed=42, development_fraction=0.5,
                       train_sample_size=5000, development_sample_size=1000, evaluation_sample_size=1000,
                       chunk_size=50_000):
    dataset_dir, output_dir = Path(dataset_dir).resolve(), Path(output_dir).resolve()
    split_dir = Path(split_dir).resolve() if split_dir is not None else dataset_dir
    if any(output_dir == root or root in output_dir.parents for root in (dataset_dir, split_dir)):
        raise ValueError("Experiment output must be outside the source and legacy split directories")
    if output_dir.exists():
        raise FileExistsError(f"Output already exists: {output_dir}; choose a new version directory to preserve frozen IDs")
    if not 0 < development_fraction < 1:
        raise ValueError("development_fraction must be between 0 and 1")
    sample_sizes = {"train": train_sample_size, "development": development_sample_size, "evaluation": evaluation_sample_size}
    if chunk_size <= 0 or any(size <= 0 for size in sample_sizes.values()):
        raise ValueError("Chunk and sample sizes must be positive")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    start = monotonic()
    started_at = datetime.now(timezone.utc).isoformat()
    memory_before = memory_snapshot()
    with tempfile.TemporaryDirectory(prefix=".pass2-", dir=output_dir.parent) as temporary:
        workspace = Path(temporary)
        staged = workspace / "result"
        staged.mkdir()
        connection = sqlite3.connect(workspace / "audit.sqlite")
        try:
            # This database is disposable scratch space, never a saved model/data artifact.
            connection.execute("PRAGMA journal_mode=OFF")
            connection.execute("PRAGMA synchronous=OFF")
            connection.execute("PRAGMA cache_size=-65536")
            connection.execute("PRAGMA temp_store=FILE")
            connection.executescript(SCHEMA)
            audit = _audit(connection, dataset_dir, split_dir, chunk_size, seed)
            print("Audit passed. Assigning development/evaluation businesses and small samples...", flush=True)
            warnings = _assign_roles(connection, development_fraction, chunk_size)
            files = _export(connection, staged, sample_sizes)
            audit.update({"started_at_utc": started_at, "finished_at_utc": datetime.now(timezone.utc).isoformat(),
                          "elapsed_seconds_including_manifests": round(monotonic() - start, 3),
                          "peak_process_memory_bytes": peak_process_memory_bytes(),
                          "memory_before": memory_before, "memory_after": memory_snapshot(),
                          "environment": {"python": platform.python_version(), "sqlite": sqlite3.sqlite_version, "platform": platform.platform()},
                          "chunk_size": chunk_size})
            manifest = {
                "schema_version": 1, "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "seed": seed, "development_fraction_of_legacy_validation": development_fraction,
                "selection_method": "SHA-256(seed + NUL + entity_id), first 16 bytes, tie-break by ID; stratify by country and zero/nonzero true links. Development uses the first floor(n*fraction) ranks, clamped to keep both roles when n>=2.",
                "sample_method": "One per available stratum, remaining slots allocated proportionally by largest remainder, then lowest stable ranks within each role/stratum. Oversized requests are capped to available rows.",
                "requested_sample_sizes": sample_sizes, "files": files, "warnings": warnings,
                "audit_report": "audit.json", "audit_passed": True,
                "inputs": audit["inputs"],
                "target_pool": {"scope": "Complete original training Source 2 and Source 3; never label-selected candidates",
                                "records": audit["counts"]["source2"] + audit["counts"]["source3"],
                                "paths": [str(dataset_dir / "train/train_source2.tsv"), str(dataset_dir / "train/train_source3.tsv")]},
                "evaluation_policy": "Choose settings on development only. Evaluation IDs are reserved; no model accuracy has been measured during preparation.",
            }
            _write_json(staged / "audit.json", audit)
            _write_json(staged / "manifest.json", manifest)
            staged.rename(output_dir)
            print(f"Saved frozen experiment files: {output_dir}", flush=True)
            return manifest
        except (OSError, ValueError, sqlite3.Error) as exc:
            if not output_dir.exists():
                output_dir.mkdir()
                _write_json(output_dir / "audit.json", {"status": "fail", "error": str(exc), "started_at_utc": started_at,
                                                       "finished_at_utc": datetime.now(timezone.utc).isoformat()})
            raise
        finally:
            connection.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    add_dataset_argument(parser)
    parser.add_argument("--split-dir", type=Path, help="Existing train/val split root (default: dataset directory).")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "artifacts/evaluation/pass2_v1", help="New version directory; existing directories are never overwritten.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--development-fraction", type=float, default=0.5)
    parser.add_argument("--train-sample-size", type=positive_int, default=5000)
    parser.add_argument("--development-sample-size", type=positive_int, default=1000)
    parser.add_argument("--evaluation-sample-size", type=positive_int, default=1000)
    parser.add_argument("--chunk-size", type=positive_int, default=50_000)
    args = parser.parse_args(argv)
    try:
        prepare_experiment(args.dataset_dir, args.output_dir, split_dir=args.split_dir, seed=args.seed,
                           development_fraction=args.development_fraction, train_sample_size=args.train_sample_size,
                           development_sample_size=args.development_sample_size, evaluation_sample_size=args.evaluation_sample_size,
                           chunk_size=args.chunk_size)
    except (OSError, ValueError, sqlite3.Error) as exc:
        parser.exit(1, f"Experiment preparation failed: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
