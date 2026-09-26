import json
import shutil

import pandas as pd
import pytest

from src.candidates import candidate_pairs
from src.data_loading import SOURCE_COLUMNS, DataFormatError, read_tsv
from src.evaluation_io import fingerprint_file
from src.model_io import load_model
from src.pipeline import DEFAULT_FIXTURE, _sources, infer, run_fixture
from src.submission import validate_output_files


@pytest.fixture(scope="module")
def completed(tmp_path_factory):
    output = tmp_path_factory.mktemp("pass3") / "run"
    before = {str(p): fingerprint_file(p) for p in DEFAULT_FIXTURE.rglob("*.tsv")}
    report = run_fixture(DEFAULT_FIXTURE, output)
    assert before == {str(p): fingerprint_file(p) for p in DEFAULT_FIXTURE.rglob("*.tsv")}
    return output, report


def test_connected_pipeline_trains_reloads_scores_and_validates_every_query(completed):
    output, report = completed
    queries, targets = _sources(DEFAULT_FIXTURE, "test")
    predictions, candidates = validate_output_files(output, queries.index, targets.index)
    assert report["scope"] == "synthetic_fixture_only"
    assert report["training"]["positive_pairs"] > 0 and report["training"]["negative_pairs"] > 0
    assert report["inference"]["queries"] == 8
    assert report["reload"]["maximum_absolute_difference"] <= 1e-12
    assert report["validation"] == {"project_checks": "passed", "supplied_validator": "passed",
                                     "check_ids": True, "errors": [], "warnings": []}
    assert predictions["S1-demo-empty"] == candidates["S1-demo-empty"] == frozenset()
    assert "S3-demo-hidden" not in candidates["S1-demo-missed"]
    assert report["evaluation"]["pair_diagnostics"]["fn"] >= 1
    assert report["evaluation"]["candidate_diagnostics"]["true_links"] == 7
    assert report["evaluation"]["candidate_diagnostics"]["pair_recall"] < 1
    trace = pd.read_csv(output / "scored_pairs.tsv", sep="\t", keep_default_na=False)
    traced = list(trace[["source1_entity_id", "target_entity_id"]].itertuples(index=False, name=None))
    assert set(traced) == set(candidate_pairs(candidates, queries.index, targets.index))
    assert len(traced) == len(set(traced)) == report["inference"]["scored_pairs"]
    assert "NA" in trace["target_business_name"].values
    for path, expected_hash in report["outputs"].items():
        assert fingerprint_file(output / path) == expected_hash
    assert json.loads((output / "report.json").read_text())["status"] == "passed"


def test_training_trace_labels_are_consistent_with_answer_key(completed):
    from src.scoring import load_id_lists

    output, _ = completed
    truth = load_id_lists(DEFAULT_FIXTURE / "train/train_ground_truth.tsv")
    trace = pd.read_csv(output / "training_pairs.tsv", sep="\t", keep_default_na=False)
    assert not trace.duplicated(["source1_entity_id", "target_entity_id"]).any()
    for row in trace.itertuples():
        assert row.label == int(row.target_entity_id in truth[row.source1_entity_id])
    test_ids = set(read_tsv(DEFAULT_FIXTURE / "test/test_source1.tsv", SOURCE_COLUMNS)["entity_id"])
    assert not test_ids & set(trace["source1_entity_id"])


def test_saved_model_can_predict_again_without_training_or_an_answer_key(completed, monkeypatch):
    import src.pipeline

    output, _ = completed
    def forbidden(*args, **kwargs):
        raise AssertionError("Inference must not train or read labels")
    monkeypatch.setattr(src.pipeline, "fit_model", forbidden)
    monkeypatch.setattr(src.pipeline, "_truth", forbidden)
    queries, targets = _sources(DEFAULT_FIXTURE, "test")
    model, metadata = load_model(output / "model")
    candidates, _, _, _, predictions = infer(model, metadata, queries, targets)
    saved_predictions, saved_candidates = validate_output_files(output, queries.index, targets.index)
    assert {k: set(v) for k, v in predictions.items()} == saved_predictions
    assert {k: set(v) for k, v in candidates.items()} == saved_candidates


def test_changing_only_evaluation_answers_changes_score_but_never_predictions(tmp_path, completed):
    output, report = completed
    fixture = tmp_path / "fixture"
    shutil.copytree(DEFAULT_FIXTURE, fixture)
    expected = fixture / "expected_test_matches.tsv"
    expected.write_text(expected.read_text().replace("S1-demo-empty\t\n", "S1-demo-empty\tS2-demo-distractor\n"))
    changed = tmp_path / "changed"
    result = run_fixture(fixture, changed)
    for filename in ("matching_results.tsv", "candidate_pairs.tsv", "scored_pairs.tsv", "training_pairs.tsv", "model/model.txt"):
        assert (changed / filename).read_bytes() == (output / filename).read_bytes()
    assert result["evaluation"]["macro_f0_5"] < report["evaluation"]["macro_f0_5"]


def test_existing_outputs_and_input_directory_are_protected(tmp_path, completed):
    output, _ = completed
    with pytest.raises(FileExistsError, match="already exists"):
        run_fixture(DEFAULT_FIXTURE, output)
    with pytest.raises(ValueError, match="outside"):
        run_fixture(DEFAULT_FIXTURE, DEFAULT_FIXTURE / "result")


def test_large_inputs_fail_instead_of_silently_truncating(tmp_path, monkeypatch):
    import src.pipeline

    monkeypatch.setattr(src.pipeline, "MAX_FIXTURE_ROWS", 2)
    with pytest.raises(DataFormatError, match="fixture limit"):
        run_fixture(DEFAULT_FIXTURE, tmp_path / "output")
    assert not (tmp_path / "output").exists()


def test_bad_evaluation_truth_leaves_no_partial_success_artifacts(tmp_path):
    fixture = tmp_path / "fixture"
    shutil.copytree(DEFAULT_FIXTURE, fixture)
    path = fixture / "expected_test_matches.tsv"
    path.write_text(path.read_text().replace("S3-demo-hidden", "S3-nonexistent"))
    with pytest.raises(DataFormatError, match="unknown truth"):
        run_fixture(fixture, tmp_path / "failed")
    assert not (tmp_path / "failed").exists()
    assert not list(tmp_path.glob(".pass3-*"))


def test_entire_prediction_batch_can_have_no_candidates(tmp_path):
    fixture = tmp_path / "fixture"
    shutil.copytree(DEFAULT_FIXTURE, fixture)
    for number in (2, 3):
        (fixture / f"test/test_source{number}.tsv").write_text("\t".join(SOURCE_COLUMNS) + "\n")
    ids = read_tsv(fixture / "test/test_source1.tsv", SOURCE_COLUMNS)["entity_id"]
    (fixture / "expected_test_matches.tsv").write_text(
        "source1_entity_id\tmatched_entity_ids\n" + "".join(f"{i}\t\n" for i in ids)
    )
    output = tmp_path / "all_empty"
    report = run_fixture(fixture, output)
    predictions, candidates = validate_output_files(output, ids, [])
    assert all(not values for values in predictions.values())
    assert all(not values for values in candidates.values())
    assert report["inference"]["scored_pairs"] == 0
    assert pd.read_csv(output / "scored_pairs.tsv", sep="\t").empty
