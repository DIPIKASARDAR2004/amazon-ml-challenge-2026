"""Derived text for matching; original fields remain available unchanged."""

import unicodedata

from src.data_loading import SOURCE_COLUMNS, DataFormatError
from src.evaluation_io import validate_id


NORMALIZATION = {"version": 1, "unicode": "NFKC", "casefold": True,
                 "whitespace": "collapse", "transliteration": False}


def normalize_text(value):
    if not isinstance(value, str):
        raise DataFormatError("Normalization requires raw strings, including explicit blanks")
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def prepare_records(frame, prefixes):
    """Validate raw records and add normalized copies; never normalize IDs."""
    if tuple(frame.columns) != SOURCE_COLUMNS:
        raise DataFormatError("Unexpected source schema")
    if not all(isinstance(value, str) for row in frame.itertuples(index=False, name=None) for value in row):
        raise DataFormatError("Source fields must be strings")
    for value in frame["entity_id"]:
        validate_id(value, prefixes)
    if frame["entity_id"].duplicated().any():
        raise DataFormatError("Duplicate source IDs")
    result = frame.copy().set_index("entity_id", drop=False)
    for column in ("business_name", "business_address", "country"):
        result[column + "_normalized"] = result[column].map(normalize_text)
    return result
