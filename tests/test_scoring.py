"""Independent metric examples and strict prediction coverage contracts."""

from pathlib import Path

import pytest

from src.data_loading import DataFormatError
from src.scoring import load_id_lists, score_predictions


A, B, C = "S2-A", "S2-B", "S3-C"


@pytest.mark.parametrize("truth,predicted,expected", [
    ([], [], 1), ([], [A], 0), ([A], [], 0),
    ([A, B], [A, B], 1), ([A, B], [A], 5 / 6),
    ([A, B], [A, B, C], 5 / 7), ([A, B], [A, C], 1 / 2),
])
def test_official_hand_calculated_cases(truth, predicted, expected):
    result = score_predictions({"S1-1": truth}, {"S1-1": predicted})
    assert result["macro_f0_5"] == pytest.approx(expected)


def test_macro_averages_businesses_not_links():
    truth = {"S1-1": [A, B, C, "S3-D"], "S1-2": [A], "S1-3": []}
    predicted = {"S1-1": truth["S1-1"], "S1-2": [], "S1-3": []}
    result = score_predictions(truth, predicted, countries={"S1-1": "US", "S1-2": "India", "S1-3": "US"})
    assert result["macro_f0_5"] == pytest.approx(2 / 3)
    assert result["pair_diagnostics"] == {"tp": 4, "fp": 0, "fn": 1, "precision": 1, "recall": 0.8}
    assert result["by_country"]["US"]["macro_f0_5"] == 1
    assert result["by_country"]["India"]["macro_f0_5"] == 0


def test_retrieval_misses_stay_in_denominator():
    result = score_predictions({"S1-1": [A, B]}, {"S1-1": [A]}, candidates={"S1-1": [A]})
    assert result["macro_f0_5"] == pytest.approx(5 / 6)
    assert result["pair_diagnostics"]["fn"] == 1
    assert result["candidate_diagnostics"]["pair_recall"] == 0.5
    assert result["candidate_diagnostics"]["oracle_macro_f0_5"] == pytest.approx(5 / 6)


def test_zero_candidates_and_singletons_are_included():
    result = score_predictions({"S1-1": [A], "S1-2": []}, {"S1-1": [], "S1-2": []}, candidates={"S1-1": [], "S1-2": []})
    assert result["entities"] == 2
    assert result["macro_f0_5"] == 0.5
    assert result["candidate_diagnostics"]["empty_entities"] == 2
    assert result["pair_diagnostics"]["precision"] is None
    assert result["singletons"]["correct_empty"] == 1


def test_undefined_diagnostics_are_null_not_invented_scores():
    result = score_predictions({"S1-1": []}, {"S1-1": []}, candidates={"S1-1": []})
    assert result["macro_f0_5"] == 1
    assert result["pair_diagnostics"]["recall"] is None
    assert result["candidate_diagnostics"]["pair_recall"] is None
    assert score_predictions({"S1-1": [A]}, {"S1-1": [A]})["candidate_diagnostics"] is None


@pytest.mark.parametrize("predictions", [{}, {"S1-1": [], "S1-extra": []}])
def test_prediction_coverage_must_be_exact(predictions):
    with pytest.raises(DataFormatError, match="coverage"):
        score_predictions({"S1-1": [A]}, predictions)


def test_candidate_coverage_and_subset_are_enforced():
    with pytest.raises(DataFormatError, match="coverage"):
        score_predictions({"S1-1": [A]}, {"S1-1": []}, candidates={})
    with pytest.raises(DataFormatError, match="candidate"):
        score_predictions({"S1-1": [A]}, {"S1-1": [A]}, candidates={"S1-1": []})


@pytest.mark.parametrize("targets", [[A, A], ["S1-wrong"], [" S2-A"], [""], ["S2-"], ["S2-A,S3-B"]])
def test_invalid_target_collections_are_rejected(targets):
    with pytest.raises(DataFormatError):
        score_predictions({"S1-1": [A]}, {"S1-1": targets})


@pytest.mark.parametrize("rows", [
    "S1-1\tS2-A\nS1-1\t\n", "S1-1\tS2-A,S2-A\n",
    "S1-1\tS2-A,\n", "S1-1\tS2-A, S3-C\n", "S1-1\tS1-2\n",
])
def test_tsv_errors_are_caught_before_set_conversion(tmp_path, rows):
    path = tmp_path / "predictions.tsv"
    path.write_text("source1_entity_id\tmatched_entity_ids\n" + rows)
    with pytest.raises(DataFormatError):
        load_id_lists(path)


def test_selected_truth_can_be_streamed_but_predictions_cannot_hide_extra_rows(tmp_path):
    path = tmp_path / "truth.tsv"
    path.write_text("source1_entity_id\tmatched_entity_ids\nS1-1\tS2-A\nS1-2\t\n")
    assert load_id_lists(path, selected_ids={"S1-2"}) == {"S1-2": frozenset()}
    with pytest.raises(DataFormatError, match="coverage"):
        load_id_lists(path, selected_ids={"S1-missing"})
    with pytest.raises(DataFormatError, match="coverage"):
        score_predictions({"S1-2": []}, load_id_lists(path))


def test_empty_evaluation_and_incomplete_country_mapping_are_errors():
    with pytest.raises(DataFormatError, match="empty"):
        score_predictions({}, {})
    with pytest.raises(DataFormatError, match="coverage"):
        score_predictions({"S1-1": []}, {"S1-1": []}, countries={})
