import json
import pandas as pd
import pytest

from src.data_loading import SOURCE_COLUMNS, DataFormatError, read_tsv, write_tsv
from src.retrieval_index import DiskCandidateIndex, build_index, retrieval_config, word_tokens


def test_index_keeps_all_targets_raw_tokens_and_works_without_truth(tmp_path, retrieval_dataset):
    (retrieval_dataset / "train/train_ground_truth.tsv").unlink()
    (retrieval_dataset / "train/train_source1.tsv").unlink()
    output = tmp_path / "index"
    report = build_index(retrieval_dataset, output, chunk_size=2)
    assert report["total_targets"] == 14
    assert report["scope"] == "full_training_targets"
    with DiskCandidateIndex(output) as index:
        row = index.get_records(["S3-tr03"])["S3-tr03"]
        assert row["business_name"] == "NA" and row["name"] == "na"
        query = {"entity_id": "S1-query", "business_name": "A Completely Different Alias",
                 "business_address": "7 Indigo Square", "country": "NeverSeenCountry"}
        result = index.retrieve_one(query)
        assert "S3-tr03" in result["channels"]["exact_address"]
        assert "S3-tr03" in result["variants"]["hybrid"]
        with pytest.raises(DataFormatError, match="Unknown"):
            index.get_records(["S2-unknown"])


def test_word_retrieval_handles_reordered_names_and_unshared_script_addresses(tmp_path, retrieval_dataset):
    output = tmp_path / "index"
    build_index(retrieval_dataset, output)
    with DiskCandidateIndex(output) as index:
        query = {"entity_id": "S1-query", "business_name": "Books River Shop",
                 "business_address": "", "country": "France"}
        result = index.retrieve_one(query)
        assert result["channels"]["exact_name"] == []
        assert "S2-tr03" in result["channels"]["name_words"]
        query.update(business_name="నామం", business_address="22 Pine Road")
        assert "S2-tr03" in index.retrieve_one(query)["variants"]["hybrid"]
        query.update(business_name="సూర్య స్టోర్స్", business_address="")
        assert "S2-tr05" in index.retrieve_one(query)["channels"]["name_words"]


def test_word_tokens_preserve_accents_marks_and_escape_fts_operators():
    assert word_tokens('ÉCOLE "OR" (Cafe*)') == ["cafe", "or", "école"]
    assert word_tokens("చంద్ర స్టోర్స్") == ["చంద్ర", "స్టోర్స్"]
    assert word_tokens("NA") == ["na"]


def test_ties_limits_and_batches_are_independent_of_input_order(tmp_path, retrieval_dataset):
    rows = [(f"S2-{i:02}", "Same Name", "Same Address", "ExampleCountry") for i in range(9)]
    source = retrieval_dataset / "train/train_source2.tsv"
    write_tsv(pd.DataFrame(rows, columns=SOURCE_COLUMNS), source, SOURCE_COLUMNS)
    config = retrieval_config(per_channel=7, max_candidates=6)
    first = tmp_path / "first"
    build_index(retrieval_dataset, first, chunk_size=2)
    write_tsv(pd.DataFrame(rows[::-1], columns=SOURCE_COLUMNS), source, SOURCE_COLUMNS)
    second = tmp_path / "second"
    build_index(retrieval_dataset, second, chunk_size=3)
    query = {"entity_id": "S1-a", "business_name": "Same Name", "business_address": "Same Address", "country": "US"}
    with DiskCandidateIndex(first, config=config) as one, DiskCandidateIndex(second, config=config) as two:
        expected = one.retrieve_one(query)
        assert expected == two.retrieve_one(query)
        assert expected["variants"]["hybrid"] == [f"S2-{i:02}" for i in range(6)]
        assert len(expected["trace"]) == 6
        one.retrieve_one({**query, "entity_id": "S1-other", "business_name": "Elsewhere"})
        assert one.retrieve_one(query) == expected


def test_blank_queries_empty_pool_and_frequent_tokens_are_explicit(tmp_path, retrieval_dataset):
    output = tmp_path / "index"
    build_index(retrieval_dataset, output)
    with DiskCandidateIndex(output, config=retrieval_config(max_token_documents=1)) as index:
        blank = {"entity_id": "S1-a", "business_name": " ", "business_address": "", "country": "US"}
        assert index.retrieve_one(blank)["variants"]["hybrid"] == []
        query = {**blank, "business_name": "Amber Bakery"}
        result = index.retrieve_one(query)
        assert "amber" not in result["selected_terms"]["name"]
        assert "S2-tr01" in result["channels"]["exact_name"]  # Frequency filtering only affects word search.
    for number in (2, 3):
        write_tsv(pd.DataFrame(columns=SOURCE_COLUMNS), retrieval_dataset / f"train/train_source{number}.tsv", SOURCE_COLUMNS)
    build_index(retrieval_dataset, tmp_path / "empty")
    with DiskCandidateIndex(tmp_path / "empty") as index:
        assert index.retrieve_one(query)["variants"]["hybrid"] == []


def test_duplicate_ids_fail_without_publishing_index(tmp_path, retrieval_dataset):
    path = retrieval_dataset / "train/train_source2.tsv"
    frame = read_tsv(path, SOURCE_COLUMNS)
    write_tsv(pd.concat([frame, frame.iloc[:1]], ignore_index=True), path, SOURCE_COLUMNS)
    with pytest.raises(DataFormatError, match="Duplicate"):
        build_index(retrieval_dataset, tmp_path / "bad")
    assert not (tmp_path / "bad").exists()
    assert not list(tmp_path.glob(".retrieval-build-*"))


def test_build_protects_inputs_and_existing_indexes_and_labels_pilot(tmp_path, retrieval_dataset):
    with pytest.raises(ValueError, match="outside"):
        build_index(retrieval_dataset, retrieval_dataset / "index")
    output = tmp_path / "pilot"
    report = build_index(retrieval_dataset, output, limit_per_source=2)
    assert report["scope"] == "restricted_first_rows_pilot" and report["total_targets"] == 4
    with pytest.raises(FileExistsError):
        build_index(retrieval_dataset, output)


@pytest.mark.parametrize("kind", ["normalization", "database"])
def test_incompatible_or_changed_index_fails(tmp_path, retrieval_dataset, kind):
    output = tmp_path / "index"
    build_index(retrieval_dataset, output)
    if kind == "normalization":
        path = output / "index.json"
        metadata = json.loads(path.read_text()); metadata["normalization"] = {}
        path.write_text(json.dumps(metadata))
    else:
        with (output / "targets.sqlite").open("ab") as stream:
            stream.write(b"changed")
    with pytest.raises(DataFormatError):
        DiskCandidateIndex(output)


@pytest.mark.parametrize("options", [{"per_channel": 0}, {"max_candidates": -1}, {"max_terms": 1.5}, {"max_token_documents": True}])
def test_invalid_limits_fail(options):
    with pytest.raises(ValueError):
        retrieval_config(**options)
