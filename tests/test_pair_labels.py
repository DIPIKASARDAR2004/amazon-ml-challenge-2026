import pytest

from src.data_loading import DataFormatError
from src.pair_labels import label_pairs


def test_only_retrieved_pairs_are_labeled_without_contradictions_or_truth_injection():
    pairs = [("S1-a", "S2-correct"), ("S1-a", "S3-wrong"), ("S1-empty", "S2-correct")]
    truth = {"S1-a": ["S2-correct", "S3-missed"], "S1-empty": []}
    assert label_pairs(pairs, truth).tolist() == [1, 0, 0]
    assert len(pairs) == 3  # The missing positive was not inserted into retrieval.


@pytest.mark.parametrize("pairs,truth,message", [
    ([("S1-a", "S2-a")] * 2, {"S1-a": ["S2-a"]}, "Duplicate"),
    ([("S1-a", "S2-a")], {}, "Missing"),
    ([("S1-a", "S2-a")], {"S1-a": ["S2-a", "S2-a"]}, "duplicate"),
])
def test_invalid_training_pairs_fail(pairs, truth, message):
    with pytest.raises(DataFormatError, match=message):
        label_pairs(pairs, truth)
