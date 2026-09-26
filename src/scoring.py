"""Official per-business macro F0.5 and separately labeled pair/retrieval diagnostics."""

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path

from src.data_loading import GROUND_TRUTH_COLUMNS, SOURCE_COLUMNS, DataFormatError, iter_tsv
from src.evaluation_io import (
    parse_targets, read_entity_ids, require_coverage, target_set, validate_id,
)


def load_id_lists(path, *, column="matched_entity_ids", selected_ids=None):
    """Load lists without hiding duplicate rows/targets behind set conversion.

    Selection is intended for streaming full ground truth into a small query set.
    Predictions/candidates are always loaded unfiltered so extra rows are errors.
    """
    result = {}
    for chunk in iter_tsv(path, ("source1_entity_id", column)):
        for entity_id, text in chunk.itertuples(index=False, name=None):
            validate_id(entity_id, context=str(path))
            if selected_ids is not None and entity_id not in selected_ids:
                continue
            if entity_id in result:
                raise DataFormatError(f"{path}: duplicate Source 1 row {entity_id}")
            result[entity_id] = parse_targets(text, context=f"{path}: {entity_id}")
    if selected_ids is not None:
        require_coverage(result, selected_ids, context=f"{path}: selected truth")
    return result


def load_countries(path, selected_ids):
    result = {}
    for chunk in iter_tsv(path, SOURCE_COLUMNS):
        for entity_id, country in chunk.loc[chunk["entity_id"].isin(selected_ids), ["entity_id", "country"]].itertuples(index=False, name=None):
            if entity_id in result:
                raise DataFormatError(f"{path}: duplicate Source 1 row {entity_id}")
            result[entity_id] = country
    require_coverage(result, selected_ids, context="country records")
    return result


def business_f0_5(truth, predicted):
    if not truth:
        return float(not predicted)
    tp = len(truth & predicted)
    fp, fn = len(predicted - truth), len(truth - predicted)
    return 5 * tp / (5 * tp + 4 * fp + fn)


def _normalized(mapping, label):
    result = {}
    for entity_id, targets in mapping.items():
        validate_id(entity_id, context=label)
        result[entity_id] = target_set(targets, context=f"{label}: {entity_id}")
    return result


def _summarize(truth, predicted, ids):
    tp = fp = fn = singleton_count = correct_empty = 0
    scores = []
    for entity_id in ids:
        actual, chosen = truth[entity_id], predicted[entity_id]
        tp += len(actual & chosen)
        fp += len(chosen - actual)
        fn += len(actual - chosen)
        scores.append(business_f0_5(actual, chosen))
        if not actual:
            singleton_count += 1
            correct_empty += not chosen
    return {
        "entities": len(ids),
        "macro_f0_5": math.fsum(scores) / len(ids),
        "pair_diagnostics": {"tp": tp, "fp": fp, "fn": fn,
                             "precision": tp / (tp + fp) if tp + fp else None,
                             "recall": tp / (tp + fn) if tp + fn else None},
        "singletons": {"entities": singleton_count, "correct_empty": correct_empty,
                       "false_merges": singleton_count - correct_empty,
                       "accuracy": correct_empty / singleton_count if singleton_count else None},
    }


def score_predictions(truth, predictions, *, candidates=None, countries=None):
    """Require exact query coverage; never restrict truth to retrieved targets."""
    truth = _normalized(truth, "truth")
    if not truth:
        raise DataFormatError("Cannot score an empty evaluation set")
    predictions = _normalized(predictions, "predictions")
    require_coverage(predictions, truth, context="predictions")
    if candidates is not None:
        candidates = _normalized(candidates, "candidates")
        require_coverage(candidates, truth, context="candidates")
        for entity_id in truth:
            if predictions[entity_id] - candidates[entity_id]:
                raise DataFormatError(f"{entity_id}: predicted targets are absent from candidates")
    if countries is not None:
        require_coverage(countries, truth, context="countries")
    ids = sorted(truth)
    result = _summarize(truth, predictions, ids)
    result["candidate_diagnostics"] = None
    if candidates is not None:
        true_links = sum(map(len, truth.values()))
        recovered = sum(len(truth[key] & candidates[key]) for key in ids)
        counts = [len(candidates[key]) for key in ids]
        result["candidate_diagnostics"] = {
            "true_links": true_links, "recovered_true_links": recovered,
            "pair_recall": recovered / true_links if true_links else None,
            "total_candidates": sum(counts), "empty_entities": counts.count(0),
            "mean_candidates": sum(counts) / len(ids), "maximum_candidates": max(counts),
            "oracle_macro_f0_5": math.fsum(
                business_f0_5(truth[key], truth[key] & candidates[key]) for key in ids
            ) / len(ids),
        }
    result["by_country"] = {}
    if countries is not None:
        groups = defaultdict(list)
        for entity_id in ids:
            groups[countries[entity_id]].append(entity_id)
        result["by_country"] = {country: _summarize(truth, predictions, group)
                                for country, group in sorted(groups.items())}
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--truth", type=Path, required=True, help="Labeled training ground truth, never test data.")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, help="Optional exact candidate lists scored by the model.")
    parser.add_argument("--entities", type=Path, help="Saved query-ID TSV. Omit only when scoring all truth rows, e.g. a tiny fixture.")
    parser.add_argument("--source1", type=Path, help="Source 1 TSV for optional country breakdowns.")
    parser.add_argument("--report", type=Path, help="Optional JSON result; must not replace an input file.")
    args = parser.parse_args(argv)
    paths = {name: getattr(args, name) for name in ("truth", "predictions", "candidates", "entities", "source1") if getattr(args, name)}
    if args.report and args.report.resolve() in {path.resolve() for path in paths.values()}:
        parser.error("The report must not replace an input file")
    try:
        selected = read_entity_ids(args.entities) if args.entities else None
        truth = load_id_lists(args.truth, selected_ids=selected)
        predictions = load_id_lists(args.predictions)
        candidates = load_id_lists(args.candidates, column="candidate_entity_ids") if args.candidates else None
        countries = load_countries(args.source1, set(truth)) if args.source1 else None
        result = score_predictions(truth, predictions, candidates=candidates, countries=countries)
        result["scored_at_utc"] = datetime.now(timezone.utc).isoformat()
        result["inputs"] = {name: str(path.resolve()) for name, path in paths.items()}
        result["scope"] = "Selected entity IDs" if selected is not None else "All ground-truth rows"
        result["limitations"] = [
            "Pair precision/recall are diagnostics, not the official macro score.",
            "Target-ID existence is checked by the audit/output validator, not by this scorer.",
            "When entities are selected, unselected truth rows are not semantically audited here.",
        ]
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    except (OSError, ValueError) as exc:
        parser.exit(1, f"Scoring failed: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
