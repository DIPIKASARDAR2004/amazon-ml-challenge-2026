import pytest

from src.candidates import CandidateIndex, candidate_pairs
from src.data_loading import DataFormatError


def test_address_can_retrieve_na_without_country_filter_and_blanks_return_nothing(records):
    queries = records([("S1-a", "National Atelier", "7 Indigo Square", "ExampleCountry"),
                       ("S1-empty", " ", "", "US")])
    targets = records([("S2-a", "NA", "7 Indigo Square", "India"),
                       ("S3-other", "NA", "900 Birch Boulevard", "India")])
    result = CandidateIndex(targets, minimum_similarity=0.9).retrieve(queries)
    assert result == {"S1-a": ["S2-a"], "S1-empty": []}


def test_deterministic_ties_limits_deduplication_and_query_batches(records):
    queries = records([("S1-a", "Same", "Same Address", "US"), ("S1-b", "Same", "", "France")])
    targets = records([(f"S2-{i}", "Same", "Same Address", "US") for i in range(8)])
    index = CandidateIndex(targets.iloc[::-1], top_k=7)
    all_results = index.retrieve(queries)
    assert all_results["S1-a"] == [f"S2-{i}" for i in range(7)]
    assert all_results == {**index.retrieve(queries.iloc[:1]), **index.retrieve(queries.iloc[1:])}
    assert CandidateIndex(targets, top_k=7).retrieve(queries) == all_results
    assert len(candidate_pairs(all_results, queries.index, targets.index)) == 14


def test_unicode_and_empty_target_pools(records):
    queries = records([("S1-a", "చంద్ర స్టోర్స్", "", "India")])
    targets = records([("S3-a", "చంద్ర స్టోర్స్", "", "India")])
    assert CandidateIndex(targets).retrieve(queries) == {"S1-a": ["S3-a"]}
    assert CandidateIndex(targets.iloc[:0]).retrieve(queries) == {"S1-a": []}
    assert CandidateIndex(records([("S3-b", "", " ", "US")])).retrieve(queries) == {"S1-a": []}


@pytest.mark.parametrize("mapping, message", [
    ({}, "coverage"), ({"S1-a": ["S2-a", "S2-a"]}, "duplicate"),
    ({"S1-a": ["S2-missing"]}, "unknown"), ({"S1-a": ["S1-a"]}, "invalid ID"),
])
def test_candidate_contract_rejects_invalid_pairs(mapping, message):
    with pytest.raises(DataFormatError, match=message):
        candidate_pairs(mapping, ["S1-a"], ["S2-a"])


@pytest.mark.parametrize("options", [{"top_k": 0}, {"minimum_similarity": 0},
                                     {"minimum_similarity": float("nan")}, {"minimum_similarity": 1.1}])
def test_invalid_retrieval_settings_fail(records, options):
    with pytest.raises(ValueError):
        CandidateIndex(records([]), **options)
