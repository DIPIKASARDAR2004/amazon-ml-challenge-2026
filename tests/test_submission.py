import numpy as np
import pytest

from src.candidates import candidate_pairs
from src.data_loading import DataFormatError
from src.submission import choose_matches, validate_output_files, validate_outputs, write_outputs
from utils.validate_submission import validate


@pytest.fixture
def output_case(tmp_path):
    queries = ["S1-empty", "S1-one", "S1-many"]
    targets = [f"S2-{i}" for i in range(7)]
    candidates = {"S1-empty": [], "S1-one": ["S2-0", "S2-1"], "S1-many": targets}
    pairs = candidate_pairs(candidates, queries, targets)
    scores = [0.5 if pair != ("S1-one", "S2-1") else np.nextafter(0.5, 0) for pair in pairs]
    predictions = choose_matches(candidates, pairs, scores, queries, targets, threshold=0.5)
    output = tmp_path / "output"
    write_outputs(output, predictions, candidates, queries, targets)
    sources = tmp_path / "test"
    sources.mkdir()
    header = "entity_id\tbusiness_name\tbusiness_address\tcountry\n"
    for number, ids in [(1, queries), (2, targets), (3, [])]:
        (sources / f"test_source{number}.tsv").write_text(header + "".join(f"{i}\tExample\t\tUS\n" for i in ids))
    return queries, targets, candidates, predictions, output, sources


def test_zero_one_many_and_threshold_boundary_survive_both_validators(output_case):
    queries, targets, candidates, predictions, output, sources = output_case
    assert predictions == {"S1-empty": [], "S1-many": targets, "S1-one": ["S2-0"]}
    assert "S1-empty\t\n" in (output / "matching_results.tsv").read_text()
    assert (output / "candidate_pairs.tsv").read_text().startswith("source1_entity_id\tcandidate_entity_ids\n")
    saved, _ = validate_output_files(output, queries, targets)
    assert saved["S1-many"] == frozenset(targets)  # More than five accepted matches are allowed.
    assert validate(str(output / "matching_results.tsv"), str(output / "candidate_pairs.tsv"), str(sources), check_ids=True) == ([], [])


@pytest.mark.parametrize("mutation, message", [
    ("missing", "coverage"), ("extra", "coverage"), ("duplicate", "duplicate"),
    ("unknown", "unknown"), ("subset", "subset"),
])
def test_strict_project_output_contracts(output_case, mutation, message):
    queries, targets, candidates, predictions, _, _ = output_case
    if mutation == "missing":
        candidates.pop("S1-empty")
    elif mutation == "extra":
        predictions["S1-extra"] = []
    elif mutation == "duplicate":
        candidates["S1-one"].append("S2-0")
    elif mutation == "unknown":
        candidates["S1-one"].append("S3-unknown")
    else:
        predictions["S1-empty"] = ["S2-0"]
    with pytest.raises(DataFormatError, match=message):
        validate_outputs(predictions, candidates, queries, targets)


@pytest.mark.parametrize("kind", ["missing_pair", "extra_pair", "duplicate_pair", "nan_score", "short_scores", "bad_threshold"])
def test_output_candidates_must_equal_scored_pairs_and_scores_must_be_valid(kind):
    candidates = {"S1-a": ["S2-a"]}
    pairs, scores, threshold = [("S1-a", "S2-a")], [0.8], 0.5
    if kind == "missing_pair":
        pairs, scores = [], []
    elif kind == "extra_pair":
        pairs.append(("S1-a", "S3-other")); scores.append(0.9)
    elif kind == "duplicate_pair":
        pairs *= 2; scores *= 2
    elif kind == "nan_score":
        scores = [float("nan")]
    elif kind == "short_scores":
        scores = []
    else:
        threshold = float("nan")
    with pytest.raises(ValueError):
        choose_matches(candidates, pairs, scores, ["S1-a"], ["S2-a", "S3-other"], threshold=threshold)


@pytest.mark.parametrize("kind", ["missing_row", "duplicate_row", "unknown_target", "bad_header", "missing_tab"])
def test_supplied_validator_rejects_invalid_outputs(output_case, kind):
    _, _, _, _, output, sources = output_case
    path = output / "matching_results.tsv"
    text = path.read_text()
    if kind == "missing_row":
        text = text.replace("S1-empty\t\n", "")
    elif kind == "duplicate_row":
        text += "S1-empty\t\n"
    elif kind == "unknown_target":
        text = text.replace("S1-empty\t\n", "S1-empty\tS3-unknown\n")
    elif kind == "bad_header":
        text = text.replace("matched_entity_ids", "wrong_column")
    else:
        text = text.replace("S1-empty\t\n", "S1-empty\n")
    path.write_text(text)
    errors, _ = validate(str(path), str(output / "candidate_pairs.tsv"), str(sources), check_ids=True)
    assert errors


def test_project_rejects_candidate_subset_violation_that_supplied_validator_only_warns_about(output_case):
    queries, targets, _, _, output, sources = output_case
    path = output / "matching_results.tsv"
    path.write_text(path.read_text().replace("S1-empty\t\n", "S1-empty\tS2-0\n"))
    errors, warnings = validate(str(path), str(output / "candidate_pairs.tsv"), str(sources), check_ids=True)
    assert not errors and warnings
    with pytest.raises(DataFormatError, match="subset"):
        validate_output_files(output, queries, targets)
    (output / "candidate_pairs.tsv").unlink()
    with pytest.raises(FileNotFoundError):
        validate_output_files(output, queries, targets)
