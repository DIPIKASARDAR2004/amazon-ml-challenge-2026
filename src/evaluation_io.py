"""ID contracts shared by the split audit and scorer."""

import hashlib
from pathlib import Path

from src.data_loading import DataFormatError, iter_tsv


ID_COLUMNS = ("source1_entity_id",)
CANDIDATE_COLUMNS = ("source1_entity_id", "candidate_entity_ids")


def validate_id(value, prefixes=("S1-",), *, context="record"):
    if (not isinstance(value, str) or not value.startswith(prefixes) or len(value) <= 3
            or any(char.isspace() or char == "," for char in value)):
        raise DataFormatError(f"{context}: invalid ID {value!r}; expected prefix {prefixes}")
    return value


def target_set(values, *, context="record"):
    if isinstance(values, str):
        raise DataFormatError(f"{context}: expected a collection of target IDs, not a string")
    items = tuple(values)
    for value in items:
        validate_id(value, ("S2-", "S3-"), context=context)
    result = frozenset(items)
    if len(items) != len(result):
        raise DataFormatError(f"{context}: duplicate target IDs")
    return result


def parse_targets(text, *, context="record"):
    return target_set(text.split(",") if text.strip() else (), context=context)


def read_entity_ids(path, *, chunk_size=50_000):
    ids = set()
    for chunk in iter_tsv(path, ID_COLUMNS, chunk_size=chunk_size):
        for value in chunk["source1_entity_id"]:
            validate_id(value, context=str(path))
            if value in ids:
                raise DataFormatError(f"{path}: duplicate Source 1 ID {value}")
            ids.add(value)
    if not ids:
        raise DataFormatError(f"{path}: empty entity selection")
    return ids


def require_coverage(actual, expected, *, context):
    actual, expected = set(actual), set(expected)
    if actual != expected:
        missing, extra = expected - actual, actual - expected
        raise DataFormatError(
            f"{context}: coverage mismatch: {len(missing)} missing, {len(extra)} extra; "
            f"examples missing={sorted(missing)[:3]}, extra={sorted(extra)[:3]}"
        )


def fingerprint_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return {"sha256": digest.hexdigest(), "bytes": Path(path).stat().st_size}
