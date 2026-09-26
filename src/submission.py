"""Strict output contracts, including the exact candidate set actually scored."""

from pathlib import Path

import numpy as np
import pandas as pd

from src.candidates import candidate_pairs
from src.data_loading import GROUND_TRUTH_COLUMNS, DataFormatError, write_tsv
from src.evaluation_io import CANDIDATE_COLUMNS, require_coverage, target_set
from src.model_io import validate_threshold
from src.scoring import load_id_lists


def choose_matches(candidates, pairs, scores, query_ids, target_ids, *, threshold):
    validate_threshold(threshold)
    expected = candidate_pairs(candidates, query_ids, target_ids)
    if len(pairs) != len(set(pairs)) or set(pairs) != set(expected):
        raise DataFormatError("Scored pairs must equal the final candidate pairs exactly")
    scores = np.asarray(scores, dtype=np.float64)
    if scores.shape != (len(pairs),) or not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise DataFormatError("Invalid pair scores")
    result = {entity_id: [] for entity_id in sorted(candidates)}
    for (query_id, target_id), score in zip(pairs, scores):
        if score >= threshold:
            result[query_id].append(target_id)
    return {key: sorted(values) for key, values in result.items()}


def validate_outputs(predictions, candidates, query_ids, target_ids):
    candidate_pairs(candidates, query_ids, target_ids)
    require_coverage(predictions, query_ids, context="predictions")
    for entity_id, values in predictions.items():
        chosen = target_set(values, context=entity_id)
        if chosen - set(candidates[entity_id]):
            raise DataFormatError(f"{entity_id}: predictions must be a subset of candidates")


def write_outputs(directory, predictions, candidates, query_ids, target_ids):
    validate_outputs(predictions, candidates, query_ids, target_ids)
    directory = Path(directory)
    files = [("matching_results.tsv", predictions, GROUND_TRUTH_COLUMNS),
             ("candidate_pairs.tsv", candidates, CANDIDATE_COLUMNS)]
    if any((directory / name).exists() for name, _, _ in files):
        raise FileExistsError("Refusing to replace existing output files")
    directory.mkdir(parents=True, exist_ok=True)
    for name, mapping, columns in files:
        rows = [(entity_id, ",".join(sorted(mapping[entity_id]))) for entity_id in sorted(mapping)]
        write_tsv(pd.DataFrame(rows, columns=columns, dtype=str), directory / name, columns)


def validate_output_files(directory, query_ids, target_ids):
    directory = Path(directory)
    predictions = load_id_lists(directory / "matching_results.tsv")
    candidates = load_id_lists(directory / "candidate_pairs.tsv", column="candidate_entity_ids")
    validate_outputs(predictions, candidates, query_ids, target_ids)
    return predictions, candidates
