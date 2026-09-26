"""Pass 1 regressions: raw text, strict TSV structure, and bounded inspection."""

from pathlib import Path

import pandas as pd
import pytest

from src.data_loading import (
    GROUND_TRUTH_COLUMNS,
    SOURCE_COLUMNS,
    DataFormatError,
    blank_mask,
    iter_tsv,
    read_tsv,
    write_tsv,
)
from src.inspect_data import inspect_ground_truth, inspect_source


FIXTURES = Path(__file__).parent / "fixtures" / "dataset"
SOURCE = FIXTURES / "train" / "train_source1.tsv"


def test_na_is_text_and_blanks_are_separate():
    frame = read_tsv(SOURCE, SOURCE_COLUMNS)
    assert frame["business_name"].tolist() == ["NA", "", "   ", "École Café", "డ్రీమ్"]
    assert blank_mask(frame["business_name"]).tolist() == [False, True, True, False, False]
    assert not frame.isna().any().any()
    assert frame.loc[0, "entity_id"] == "S1-00001"
    assert frame.loc[3, "business_address"] == "5 Rue Exemple,\tBâtiment A"
    assert frame.loc[4, "country"] == "ExampleCountry"


def test_other_parser_sensitive_words_are_also_preserved():
    frame = read_tsv(FIXTURES / "train" / "train_source2.tsv", SOURCE_COLUMNS)
    assert frame["business_name"].tolist() == ["National Atelier", "NULL", "nan"]
    assert not frame.isna().any().any()


def test_chunk_boundaries_and_row_limit():
    chunks = list(iter_tsv(SOURCE, SOURCE_COLUMNS, chunk_size=2))
    assert [len(chunk) for chunk in chunks] == [2, 2, 1]
    pd.testing.assert_frame_equal(pd.concat(chunks, ignore_index=True), read_tsv(SOURCE, SOURCE_COLUMNS))
    limited = list(iter_tsv(SOURCE, SOURCE_COLUMNS, chunk_size=2, limit=3))
    assert [len(chunk) for chunk in limited] == [2, 1]
    assert limited[-1].iloc[-1]["entity_id"] == "S1-00003"


def test_sample_does_not_parse_rows_beyond_its_limit(tmp_path):
    path = tmp_path / "sample.tsv"
    path.write_text("\t".join(SOURCE_COLUMNS) + "\nS1-1\tNA\t\tUS\nbroken row\n", encoding="utf-8")
    assert len(read_tsv(path, SOURCE_COLUMNS, limit=1)) == 1
    with pytest.raises(DataFormatError, match="fields"):
        read_tsv(path, SOURCE_COLUMNS)


def test_round_trip_preserves_raw_values_and_empty_match_lists(tmp_path):
    for name, path, columns in [
        ("source.tsv", SOURCE, SOURCE_COLUMNS),
        ("truth.tsv", FIXTURES / "train" / "train_ground_truth.tsv", GROUND_TRUTH_COLUMNS),
    ]:
        before = read_tsv(path, columns)
        output = tmp_path / name
        write_tsv(before, output, columns)
        after = read_tsv(output, columns)
        pd.testing.assert_frame_equal(before, after)
    assert "S1-00002\t\n" in (tmp_path / "truth.tsv").read_text(encoding="utf-8")


def test_quoted_newlines_are_one_record_and_round_trip(tmp_path):
    path = tmp_path / "multiline.tsv"
    path.write_text(
        "\t".join(SOURCE_COLUMNS) + '\nS1-1\t"Name\nSecond line"\t"An ""Example"" Road"\tUS\n',
        encoding="utf-8",
    )
    frame = read_tsv(path, SOURCE_COLUMNS)
    assert len(frame) == 1
    assert frame.loc[0, "business_name"] == "Name\nSecond line"
    assert frame.loc[0, "business_address"] == 'An "Example" Road'
    output = tmp_path / "roundtrip.tsv"
    write_tsv(frame, output, SOURCE_COLUMNS)
    pd.testing.assert_frame_equal(frame, read_tsv(output, SOURCE_COLUMNS))


@pytest.mark.parametrize("header", [
    "entity_id,business_name,business_address,country",
    "entity_id\tbusiness_name\tbusiness_address",
    "entity_id\tbusiness_name\tbusiness_name\tcountry",
    "country\tentity_id\tbusiness_name\tbusiness_address",
])
def test_wrong_schema_fails_clearly(tmp_path, header):
    path = tmp_path / "wrong.tsv"
    path.write_text(header + "\n", encoding="utf-8")
    with pytest.raises(DataFormatError, match="header"):
        read_tsv(path, SOURCE_COLUMNS)


@pytest.mark.parametrize("row", [
    "S1-1\tName\tUS\n",
    "S1-1\tName\tAddress\tUS\textra\n",
    "\n",
    'S1-1\t"unterminated\tAddress\tUS\n',
])
def test_malformed_rows_are_not_silently_repaired(tmp_path, row):
    path = tmp_path / "malformed.tsv"
    path.write_text("\t".join(SOURCE_COLUMNS) + "\n" + row, encoding="utf-8")
    with pytest.raises(DataFormatError, match="line"):
        read_tsv(path, SOURCE_COLUMNS)


def test_empty_file_is_invalid_but_header_only_table_is_valid(tmp_path):
    path = tmp_path / "empty.tsv"
    path.write_text("", encoding="utf-8")
    with pytest.raises(DataFormatError, match="header"):
        read_tsv(path, SOURCE_COLUMNS)
    path.write_text("\t".join(SOURCE_COLUMNS) + "\n", encoding="utf-8")
    result = read_tsv(path, SOURCE_COLUMNS)
    assert result.empty and tuple(result.columns) == SOURCE_COLUMNS


@pytest.mark.parametrize("options", [{"chunk_size": 0}, {"chunk_size": -1}, {"limit": 0}, {"limit": -1}])
def test_nonpositive_read_limits_are_rejected(options):
    with pytest.raises(ValueError, match="positive"):
        list(iter_tsv(SOURCE, SOURCE_COLUMNS, **options))


def test_writer_rejects_implicit_missing_values_before_writing(tmp_path):
    frame = read_tsv(SOURCE, SOURCE_COLUMNS)
    frame.loc[0, "business_name"] = None
    output = tmp_path / "bad.tsv"
    with pytest.raises(DataFormatError, match="string"):
        write_tsv(frame, output, SOURCE_COLUMNS)
    assert not output.exists()


def test_inspection_counts_blanks_and_tokens_without_reclassifying_them():
    summary = inspect_source(SOURCE, source_number=1, chunk_size=2)
    assert summary["records"] == 5
    assert summary["blank_names"] == 2
    assert summary["blank_addresses"] == 2
    assert summary["literal_name_tokens"] == {"NA": 1}
    assert summary["countries"] == {"India": 2, "US": 1, "France": 1, "ExampleCountry": 1}
    assert summary["non_ascii_names"] == 2
    assert summary["blank_ids"] == summary["unexpected_id_prefix"] == 0


def test_inspection_counts_empty_ground_truth_as_zero_matches():
    summary = inspect_ground_truth(FIXTURES / "train" / "train_ground_truth.tsv", chunk_size=2)
    assert summary["records"] == 5
    assert summary["zero_matches"] == 3
    assert summary["total_true_links"] == 3
    assert summary["maximum_matches"] == 2
    assert summary["mean_matches"] == pytest.approx(0.6)


def test_inspection_reports_zero_mean_for_header_only_truth(tmp_path):
    path = tmp_path / "truth.tsv"
    path.write_text("\t".join(GROUND_TRUTH_COLUMNS) + "\n", encoding="utf-8")
    summary = inspect_ground_truth(path)
    assert summary["records"] == summary["maximum_matches"] == summary["mean_matches"] == 0
