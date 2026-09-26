"""Run the Pass 3 synthetic fixture from retrieval through validated outputs.

This bounded, in-memory teaching runner is not full-dataset inference. The
separate expected_test_matches.tsv is an invented answer key read only after
predictions are written, unlike the competition test set which has no labels.
"""

import argparse
from contextlib import redirect_stdout
from datetime import datetime, timezone
from io import StringIO
import json
from pathlib import Path
import platform
import tempfile
from time import monotonic

import numpy as np
import pandas as pd

from src.candidates import CandidateIndex, candidate_pairs, retrieval_settings
from src.config import PROJECT_ROOT, positive_int
from src.data_loading import GROUND_TRUTH_COLUMNS, SOURCE_COLUMNS, DataFormatError, read_tsv
from src.evaluation_io import fingerprint_file, parse_targets, require_coverage
from src.features import make_features
from src.model_io import RELOAD_TOLERANCE, fit_model, load_model, predict_scores, save_model, validate_threshold
from src.normalization import prepare_records
from src.pair_labels import label_pairs
from src.scoring import score_predictions
from src.submission import choose_matches, validate_output_files, write_outputs


DEFAULT_FIXTURE = PROJECT_ROOT / "tests/fixtures/pipeline"
MAX_FIXTURE_ROWS = 1_000


def _small_table(path, columns):
    table = read_tsv(path, columns, limit=MAX_FIXTURE_ROWS + 1)
    if len(table) > MAX_FIXTURE_ROWS:
        raise DataFormatError(f"{path}: exceeds the {MAX_FIXTURE_ROWS}-row fixture limit; this command is for tiny data")
    return table


def _sources(directory, split):
    source1 = _small_table(directory / split / f"{split}_source1.tsv", SOURCE_COLUMNS)
    targets = []
    for number in (2, 3):
        table = _small_table(directory / split / f"{split}_source{number}.tsv", SOURCE_COLUMNS)
        prepare_records(table, (f"S{number}-",))
        targets.append(table)
    if source1.empty:
        raise DataFormatError(f"{split}: fixture Source 1 must not be empty")
    return prepare_records(source1, ("S1-",)), prepare_records(pd.concat(targets, ignore_index=True), ("S2-", "S3-"))


def _truth(path, query_ids, target_ids):
    table = _small_table(path, GROUND_TRUTH_COLUMNS)
    if table["source1_entity_id"].duplicated().any():
        raise DataFormatError(f"{path}: duplicate truth rows")
    truth = {entity_id: parse_targets(text, context=entity_id)
             for entity_id, text in table.itertuples(index=False, name=None)}
    require_coverage(truth, query_ids, context="fixture truth")
    allowed, seen = set(target_ids), set()
    for values in truth.values():
        if values - allowed:
            raise DataFormatError(f"{path}: unknown truth targets")
        if seen & values:
            raise DataFormatError(f"{path}: target belongs to multiple Source 1 businesses")
        seen.update(values)
    return truth


def infer(model, metadata, queries, targets):
    """No answer key: rebuild the small index with the saved retrieval settings."""
    settings = metadata["retrieval"]
    candidates = CandidateIndex(targets, top_k=settings["top_k"],
                                minimum_similarity=settings["minimum_similarity"]).retrieve(queries)
    pairs = candidate_pairs(candidates, queries.index, targets.index)
    features = make_features(pairs, queries, targets)
    scores = predict_scores(model, features)
    predictions = choose_matches(candidates, pairs, scores, queries.index, targets.index,
                                 threshold=metadata["threshold"])
    return candidates, pairs, features, scores, predictions


def _write_trace(path, pairs, features, queries, targets, **extra_columns):
    rows = []
    for query_id, target_id in pairs:
        row = {"source1_entity_id": query_id, "target_entity_id": target_id}
        for side, records, entity_id in (("query", queries, query_id), ("target", targets, target_id)):
            for field in ("business_name", "business_address", "business_name_normalized", "business_address_normalized"):
                row[f"{side}_{field}"] = records.loc[entity_id, field]
        rows.append(row)
    columns = ["source1_entity_id", "target_entity_id"] + [f"{side}_{field}" for side in ("query", "target")
               for field in ("business_name", "business_address", "business_name_normalized", "business_address_normalized")]
    trace = pd.concat([pd.DataFrame(rows, columns=columns), features.reset_index(drop=True)], axis=1)
    for name, values in extra_columns.items():
        trace[name] = values
    trace.to_csv(path, sep="\t", index=False, lineterminator="\n", float_format="%.12g")


def run_fixture(fixture_dir, output_dir, *, top_k=4, minimum_similarity=0.25, threshold=0.5, seed=42):
    from utils.validate_submission import validate

    started, started_at = monotonic(), datetime.now(timezone.utc).isoformat()
    settings = retrieval_settings(top_k, minimum_similarity)
    validate_threshold(threshold)
    fixture_dir, output_dir = Path(fixture_dir).resolve(), Path(output_dir).resolve()
    if output_dir == fixture_dir or fixture_dir in output_dir.parents:
        raise ValueError("Output directory must be outside the fixture directory")
    if output_dir.exists():
        raise FileExistsError("Output directory already exists; choose a new version directory")
    train_queries, train_targets = _sources(fixture_dir, "train")
    test_queries, test_targets = _sources(fixture_dir, "test")
    if set(train_queries.index) & set(test_queries.index) or set(train_targets.index) & set(test_targets.index):
        raise DataFormatError("Fixture training and prediction IDs must be disjoint")
    training_truth = _truth(fixture_dir / "train/train_ground_truth.tsv", train_queries.index, train_targets.index)
    train_candidates = CandidateIndex(train_targets, top_k=top_k, minimum_similarity=minimum_similarity).retrieve(train_queries)
    train_pairs = candidate_pairs(train_candidates, train_queries.index, train_targets.index)
    labels = label_pairs(train_pairs, training_truth)
    training_features = make_features(train_pairs, train_queries, train_targets)
    model = fit_model(training_features, labels, seed=seed)

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".pass3-", dir=output_dir.parent) as temporary:
        staged = Path(temporary) / "result"
        staged.mkdir()
        save_model(model, staged / "model", retrieval=settings, threshold=threshold, seed=seed)
        restored, metadata = load_model(staged / "model")
        candidates, pairs, features, scores, predictions = infer(restored, metadata, test_queries, test_targets)
        differences = [np.abs(predict_scores(model, training_features) - predict_scores(restored, training_features)),
                       np.abs(predict_scores(model, features) - scores)]
        maximum_difference = max((float(values.max()) for values in differences if len(values)), default=0.0)
        if maximum_difference > RELOAD_TOLERANCE:
            raise DataFormatError("Saved/reloaded model predictions differ beyond tolerance")
        write_outputs(staged, predictions, candidates, test_queries.index, test_targets.index)
        written_predictions, written_candidates = validate_output_files(staged, test_queries.index, test_targets.index)
        if set(candidate_pairs(written_candidates, test_queries.index, test_targets.index)) != set(pairs):
            raise DataFormatError("Written candidates differ from the scored pairs")

        validator_log = StringIO()
        with redirect_stdout(validator_log):
            errors, warnings = validate(str(staged / "matching_results.tsv"), str(staged / "candidate_pairs.tsv"),
                                        str(fixture_dir / "test"), check_ids=True)
        if errors or warnings:
            raise DataFormatError(f"Fixture validator failed: {errors + warnings}")
        (staged / "validator.txt").write_text(validator_log.getvalue() + "PASS: ID existence checked; no errors or warnings.\n", encoding="utf-8")
        _write_trace(staged / "training_pairs.tsv", train_pairs, training_features, train_queries, train_targets, label=labels)
        _write_trace(staged / "scored_pairs.tsv", pairs, features, test_queries, test_targets,
                     model_score=scores, accepted=(scores >= metadata["threshold"]).astype(int))

        # Evaluation answers are accessed only after inference and both outputs.
        expected = _truth(fixture_dir / "expected_test_matches.tsv", test_queries.index, test_targets.index)
        evaluation = score_predictions(expected, written_predictions, candidates=written_candidates,
                                       countries=test_queries["country"].to_dict())
        input_paths = [fixture_dir / split / f"{split}_source{number}.tsv"
                       for split in ("train", "test") for number in (1, 2, 3)]
        input_paths += [fixture_dir / "train/train_ground_truth.tsv", fixture_dir / "expected_test_matches.tsv"]
        report = {
            "status": "passed", "scope": "synthetic_fixture_only", "fixture_dir": str(fixture_dir),
            "started_at_utc": started_at, "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": monotonic() - started, "python": platform.python_version(),
            "seed": seed, "threshold": threshold, "retrieval": settings,
            "training": {"queries": len(train_queries), "targets": len(train_targets), "retrieved_pairs": len(train_pairs),
                         "positive_pairs": int(labels.sum()), "negative_pairs": int(len(labels) - labels.sum()),
                         "true_links": sum(map(len, training_truth.values()))},
            "inference": {"queries": len(test_queries), "targets": len(test_targets), "scored_pairs": len(pairs)},
            "reload": {"maximum_absolute_difference": maximum_difference, "absolute_tolerance": RELOAD_TOLERANCE},
            "validation": {"project_checks": "passed", "supplied_validator": "passed", "check_ids": True,
                           "errors": errors, "warnings": warnings},
            "evaluation": evaluation,
            "inputs": {str(path.relative_to(fixture_dir)): fingerprint_file(path) for path in input_paths},
            "outputs": {str(path.relative_to(staged)): fingerprint_file(path) for path in sorted(staged.rglob("*")) if path.is_file()},
            "limitations": ["Invented fixture score, not competition accuracy or a held-out real-data estimate.",
                            "Threshold 0.5 is a fixed demonstration setting, not a tuned/calibrated confidence cutoff.",
                            "In-memory TF-IDF scans the tiny target pool per query; full-pool scaling is Pass 4.",
                            "Real Pass 2 experiment manifests and the reserved evaluation set are not used here."],
        }
        (staged / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        staged.rename(output_dir)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture-dir", type=Path, default=DEFAULT_FIXTURE,
                        help="Tiny synthetic fixture with train/, test/, and expected_test_matches.tsv; NOT the real dataset.")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "artifacts/pipeline/pass3_v1")
    parser.add_argument("--top-k", type=positive_int, default=4)
    parser.add_argument("--minimum-similarity", type=float, default=0.25)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    try:
        report = run_fixture(args.fixture_dir, args.output_dir, top_k=args.top_k,
                             minimum_similarity=args.minimum_similarity, threshold=args.threshold, seed=args.seed)
    except (OSError, ValueError) as exc:
        parser.exit(1, f"Fixture pipeline failed: {exc}\n")
    print(f"Pass 3 fixture complete: {args.output_dir.resolve()}")
    print(f"Trained on {report['training']['retrieved_pairs']} retrieved pairs; scored {report['inference']['scored_pairs']} fixture pairs.")
    print(f"Fixture macro F0.5: {report['evaluation']['macro_f0_5']:.6f} (NOT competition accuracy)")
    print("Saved-model reload and both output validators passed, including target-ID existence.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
