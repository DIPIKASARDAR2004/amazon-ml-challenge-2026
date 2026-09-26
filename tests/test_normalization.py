import pandas as pd
import pytest

from src.data_loading import SOURCE_COLUMNS, DataFormatError
from src.normalization import normalize_text, prepare_records


@pytest.mark.parametrize("raw, expected", [
    ("NA", "na"), ("  \t\n", ""), ("  ÉCOLE   Café ", "école café"),
    ("డ్రీమ్", "డ్రీమ్"), ("ＡＢＣ", "abc"), ("NULL", "null"), ("nan", "nan"),
])
def test_normalization_preserves_tokens_and_scripts_and_is_idempotent(raw, expected):
    assert normalize_text(raw) == expected
    assert normalize_text(normalize_text(raw)) == expected


def test_raw_fields_and_ids_are_not_modified():
    raw = pd.DataFrame([("S1-ABC", " NA ", "  ", "ExampleCountry")], columns=SOURCE_COLUMNS)
    before = raw.copy(deep=True)
    derived = prepare_records(raw, ("S1-",))
    pd.testing.assert_frame_equal(raw, before)
    assert derived.loc["S1-ABC", "business_name"] == " NA "
    assert derived.loc["S1-ABC", "business_address"] == "  "
    assert derived.loc["S1-ABC", "business_name_normalized"] == "na"
    assert derived.loc["S1-ABC", "business_address_normalized"] == ""


@pytest.mark.parametrize("value", [None, float("nan"), 10])
def test_implicit_missing_or_nontext_values_fail(value):
    with pytest.raises(DataFormatError, match="strings"):
        normalize_text(value)


def test_duplicate_or_wrong_source_ids_fail():
    raw = pd.DataFrame([("S1-a", "A", "", "US")] * 2, columns=SOURCE_COLUMNS)
    with pytest.raises(DataFormatError, match="Duplicate"):
        prepare_records(raw, ("S1-",))
    with pytest.raises(DataFormatError, match="invalid ID"):
        prepare_records(raw.iloc[:1], ("S2-",))
