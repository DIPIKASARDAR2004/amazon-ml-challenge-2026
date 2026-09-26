"""Measure retrieval on frozen development queries against the full target index."""

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sqlite3
import tempfile
from time import monotonic

import numpy as np
import pandas as pd
from rapidfuzz import fuzz

from src.config import PROJECT_ROOT, add_dataset_argument, positive_int
from src.data_loading import SOURCE_COLUMNS, DataFormatError, iter_tsv, write_tsv
from src.evaluation_io import CANDIDATE_COLUMNS, fingerprint_file, read_entity_ids, require_coverage
from src.inspect_data import peak_process_memory_bytes
from src.normalization import normalize_text, prepare_records
from src.retrieval_index import CHANNELS, DiskCandidateIndex, _write_json, retrieval_config
from src.scoring import business_f0_5, load_id_lists


def _fingerprint_matches(path, expected):
    if fingerprint_file(path) != {key: expected[key] for key in ("sha256", "bytes")}:
        raise DataFormatError(f"Fingerprint mismatch: {path}")


def load_development_queries(dataset_dir, experiment_dir):
    """Only the saved development sample is accepted; no evaluation selector."""
    manifest_path = experiment_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1 or not manifest.get("audit_passed"):
        raise DataFormatError("A successful Pass 2 experiment manifest is required")
    ids_path = experiment_dir / "sample_development_ids.tsv"
    _fingerprint_matches(ids_path, manifest["files"][ids_path.name])
    selected = read_entity_ids(ids_path)
    if len(selected) != manifest["files"][ids_path.name]["rows"]:
        raise DataFormatError("Development sample count does not match manifest")
    source = dataset_dir / "train/train_source1.tsv"
    _fingerprint_matches(source, manifest["inputs"]["source1"])
    chunks = [chunk.loc[chunk["entity_id"].isin(selected)] for chunk in iter_tsv(source, SOURCE_COLUMNS)]
    queries = prepare_records(pd.concat(chunks, ignore_index=True), ("S1-",)).sort_index()
    require_coverage(queries.index, selected, context="development queries")
    return queries, manifest, fingerprint_file(manifest_path)


def summarize_retrieval(truth, candidates):
    require_coverage(candidates, truth, context="retrieval summary")
    ids = sorted(truth)
    if not ids:
        raise DataFormatError("Cannot summarize an empty query set")
    true_links = sum(len(truth[key]) for key in ids)
    recovered = sum(len(set(candidates[key]) & truth[key]) for key in ids)
    sizes = [len(candidates[key]) for key in ids]
    positive = [key for key in ids if truth[key]]
    return {
        "queries": len(ids), "true_links": true_links, "recovered_true_links": recovered,
        "pair_recall": recovered / true_links if true_links else None,
        "positive_queries_all_links_retrieved": sum(truth[key] <= set(candidates[key]) for key in positive),
        "positive_queries": len(positive), "no_match_queries": len(ids) - len(positive),
        "total_candidates": sum(sizes), "empty_queries": sizes.count(0),
        "mean_candidates": float(np.mean(sizes)), "p50_candidates": float(np.percentile(sizes, 50)),
        "p95_candidates": float(np.percentile(sizes, 95)), "p99_candidates": float(np.percentile(sizes, 99)),
        "maximum_candidates": max(sizes),
        "oracle_macro_f0_5": math.fsum(business_f0_5(truth[key], truth[key] & set(candidates[key])) for key in ids) / len(ids),
    }


def _candidate_file(path, mapping):
    rows = [(key, ",".join(sorted(values))) for key, values in sorted(mapping.items())]
    write_tsv(pd.DataFrame(rows, columns=CANDIDATE_COLUMNS, dtype=str), path, CANDIDATE_COLUMNS)


def evaluate_retrieval(dataset_dir, experiment_dir, index_dir, output_dir, *, config=None, progress=None):
    started = monotonic()
    started_at = datetime.now(timezone.utc).isoformat()
    dataset_dir, experiment_dir, index_dir, output_dir = map(lambda p: Path(p).resolve(),
                                                            (dataset_dir, experiment_dir, index_dir, output_dir))
    if any(output_dir == root or root in output_dir.parents for root in (dataset_dir, experiment_dir, index_dir)):
        raise ValueError("Results must be outside input dataset, experiment, and index directories")
    if output_dir.exists():
        raise FileExistsError("Retrieval results already exist; choose a new version directory")
    queries, manifest, manifest_hash = load_development_queries(dataset_dir, experiment_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with DiskCandidateIndex(index_dir, config=config) as index:
        if index.metadata["scope"] != "full_training_targets":
            raise DataFormatError("Development benchmark requires the full training target index")
        for key in ("source2", "source3"):
            actual = index.metadata["inputs"][key]
            if any(actual[k] != manifest["inputs"][key][k] for k in ("sha256", "bytes")):
                raise DataFormatError("Index targets do not match the audited experiment")
        if index.metadata["total_targets"] != manifest["target_pool"]["records"]:
            raise DataFormatError("Index target coverage differs from the audited full pool")
        variants = {name: {} for name in ("name_only", "exact_name_address", "hybrid")}
        channels = {name: {} for name in CHANNELS}
        union, traces, latencies, selected_terms = {}, [], {}, {}
        for position, (entity_id, query) in enumerate(queries.iterrows(), 1):
            query_started = monotonic()
            result = index.retrieve_one(query)
            latencies[entity_id] = monotonic() - query_started
            for name in variants:
                variants[name][entity_id] = result["variants"][name]
            for name in channels:
                channels[name][entity_id] = result["channels"][name]
            union[entity_id] = result["union_before_cap"]
            selected_terms[entity_id] = result["selected_terms"]
            traces.extend({"source1_entity_id": entity_id, **row} for row in result["trace"])
            if progress and (position % 25 == 0 or position == len(queries)):
                progress(f"Retrieved {position:,}/{len(queries):,} queries ({sum(latencies.values()):.1f}s in search)")
        with tempfile.TemporaryDirectory(prefix=".retrieval-eval-", dir=output_dir.parent) as temporary:
            staged = Path(temporary) / "result"
            staged.mkdir()
            for name, mapping in variants.items():
                filename = "candidate_pairs.tsv" if name == "hybrid" else name + "_candidates.tsv"
                _candidate_file(staged / filename, mapping)
            trace_columns = ("source1_entity_id", "target_entity_id", "rank", "retrieval_score", "channels")
            pd.DataFrame(traces, columns=trace_columns).to_csv(staged / "retrieval_trace.tsv", sep="\t", index=False, lineterminator="\n")
            written = load_id_lists(staged / "candidate_pairs.tsv", column="candidate_entity_ids")
            require_coverage(written, queries.index, context="saved candidates")
            if any(set(variants["hybrid"][key]) != written[key] for key in written):
                raise DataFormatError("Saved candidate lists differ from retrieval")
            if len(traces) != sum(map(len, written.values())) or {(r["source1_entity_id"], r["target_entity_id"]) for r in traces} != {(q, t) for q, values in written.items() for t in values}:
                raise DataFormatError("Candidate trace does not equal exported candidates")

            # Answers are first loaded after all candidate generation/export.
            truth_path = dataset_dir / "train/train_ground_truth.tsv"
            _fingerprint_matches(truth_path, manifest["inputs"]["ground_truth"])
            truth = load_id_lists(truth_path, selected_ids=set(queries.index))
            truth_targets = index.get_records(set().union(*truth.values()))
            diagnostics = {"true_links": sum(map(len, truth.values())), "country_mismatch_links": 0,
                           "nonexact_name_links": 0, "nonexact_address_links": 0,
                           "short_target_name_links": 0, "non_ascii_name_links": 0,
                           "true_links_lost_at_final_cap": 0}
            failures, query_rows = [], []
            for entity_id, query in queries.iterrows():
                qname, qaddress = normalize_text(query["business_name"]), normalize_text(query["business_address"])
                for target_id in sorted(truth[entity_id]):
                    target = truth_targets[target_id]
                    diagnostics["country_mismatch_links"] += normalize_text(query["country"]) != normalize_text(target["country"])
                    diagnostics["nonexact_name_links"] += not (qname and qname == target["name"])
                    diagnostics["nonexact_address_links"] += not (qaddress and qaddress == target["address"])
                    diagnostics["short_target_name_links"] += 0 < len(target["name"]) <= 3
                    diagnostics["non_ascii_name_links"] += not (query["business_name"].isascii() and target["business_name"].isascii())
                    if target_id not in written[entity_id]:
                        at_cap = target_id in union[entity_id]
                        diagnostics["true_links_lost_at_final_cap"] += at_cap
                        failures.append({"source1_entity_id": entity_id, "target_entity_id": target_id,
                                         "query_name": query["business_name"], "target_name": target["business_name"],
                                         "query_address": query["business_address"], "target_address": target["business_address"],
                                         "query_country": query["country"], "target_country": target["country"],
                                         "name_ratio": fuzz.ratio(qname, target["name"]) / 100 if qname and target["name"] else 0,
                                         "address_ratio": fuzz.ratio(qaddress, target["address"]) / 100 if qaddress and target["address"] else 0,
                                         "miss_stage": "final_candidate_cap" if at_cap else "not_returned_by_channels"})
                query_rows.append({"source1_entity_id": entity_id, "business_name": query["business_name"],
                                   "country": query["country"], "true_links": len(truth[entity_id]),
                                   "name_only_recovered": len(truth[entity_id] & set(variants["name_only"][entity_id])),
                                   "hybrid_recovered": len(truth[entity_id] & written[entity_id]),
                                   "hybrid_candidates": len(written[entity_id]), "query_seconds": latencies[entity_id],
                                   "name_terms": ",".join(selected_terms[entity_id]["name"]),
                                   "address_terms": ",".join(selected_terms[entity_id]["address"]),
                                   "missed_ids": ",".join(sorted(truth[entity_id] - written[entity_id]))})
            failure_columns = ("source1_entity_id", "target_entity_id", "query_name", "target_name", "query_address", "target_address",
                               "query_country", "target_country", "name_ratio", "address_ratio", "miss_stage")
            pd.DataFrame(failures, columns=failure_columns).to_csv(staged / "missed_links.tsv", sep="\t", index=False, lineterminator="\n")
            pd.DataFrame(query_rows).to_csv(staged / "query_details.tsv", sep="\t", index=False, lineterminator="\n")
            groups = {}
            for country in sorted(set(queries["country"])):
                ids = list(queries.index[queries["country"] == country])
                groups[country] = summarize_retrieval({k: truth[k] for k in ids}, {k: written[k] for k in ids})
            latency_values = list(latencies.values())
            report = {
                "status": "complete", "scope": "saved_development_sample_against_full_training_targets",
                "stage": "candidate_retrieval_only_no_model_scoring", "config": index.config,
                "started_at_utc": started_at, "elapsed_seconds": monotonic() - started,
                "query_seconds_total": sum(latency_values), "query_seconds_mean": float(np.mean(latency_values)),
                "query_seconds_p95": float(np.percentile(latency_values, 95)), "query_seconds_maximum": max(latency_values),
                "peak_process_rss_bytes": peak_process_memory_bytes(), "target_records": index.metadata["total_targets"],
                "index": {"path": str(index_dir), "metadata": fingerprint_file(index_dir / "index.json"),
                          "database": index.metadata["database"], "build_seconds": index.metadata["elapsed_seconds"],
                          "build_peak_rss_bytes": index.metadata["peak_process_rss_bytes"]},
                "experiment": {"path": str(experiment_dir), "manifest": manifest_hash,
                               "sample_file": "sample_development_ids.tsv", "sample": manifest["files"]["sample_development_ids.tsv"]},
                "variants": {name: summarize_retrieval(truth, mapping) for name, mapping in variants.items()},
                "channels_before_final_cap": {name: summarize_retrieval(truth, mapping) for name, mapping in channels.items()},
                "union_before_final_cap": summarize_retrieval(truth, union), "by_country": groups,
                "true_link_diagnostics": diagnostics,
                "outputs": {path.name: fingerprint_file(path) for path in staged.iterdir()},
                "limitations": ["Development retrieval metrics are not model accuracy or leaderboard scores.",
                                "Oracle macro F0.5 assumes perfect decisions among retrieved candidates; it is an upper bound.",
                                "Word retrieval cannot translate scripts or recover every typo/alias; exact lookup is complementary.",
                                "Per-channel limits, token frequency limits, and the final cap may exclude true links.",
                                "Search time covers all four channels and reranking; it is not an isolated baseline timing.",
                                "No reserved evaluation queries or competition test records were used."],
            }
            _write_json(staged / "report.json", report)
            staged.rename(output_dir)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    add_dataset_argument(parser)
    parser.add_argument("--experiment-dir", type=Path, default=PROJECT_ROOT / "artifacts/evaluation/pass2_v1")
    parser.add_argument("--index-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--per-channel", type=positive_int, default=50)
    parser.add_argument("--max-candidates", type=positive_int, default=100)
    parser.add_argument("--max-terms", type=positive_int, default=4)
    parser.add_argument("--max-token-documents", type=positive_int, default=20_000)
    args = parser.parse_args(argv)
    config = retrieval_config(per_channel=args.per_channel, max_candidates=args.max_candidates,
                              max_terms=args.max_terms, max_token_documents=args.max_token_documents)
    try:
        report = evaluate_retrieval(args.dataset_dir, args.experiment_dir, args.index_dir, args.output_dir,
                                    config=config, progress=lambda message: print(message, flush=True))
    except (OSError, ValueError, KeyError, sqlite3.Error) as exc:
        parser.exit(1, f"Retrieval evaluation failed: {exc}\n")
    for name, result in report["variants"].items():
        print(f"{name}: recall={result['pair_recall']}; mean candidates={result['mean_candidates']:.2f}")
    print(f"Saved development retrieval report: {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
