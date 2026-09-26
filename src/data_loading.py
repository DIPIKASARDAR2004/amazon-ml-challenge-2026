"""Strict, text-preserving TSV loading into bounded-size Pandas tables.

The stdlib CSV reader validates each row's field count before Pandas sees it.
This prevents both automatic null-token conversion and silent padding of short
rows. Raw values stay unchanged; blank indicators are derived separately.
"""

import csv
from pathlib import Path

import pandas as pd


SOURCE_COLUMNS = ("entity_id", "business_name", "business_address", "country")
GROUND_TRUTH_COLUMNS = ("source1_entity_id", "matched_entity_ids")


class DataFormatError(ValueError):
    """The input is not a TSV with the requested schema."""


def iter_tsv(path, expected_columns, *, chunk_size=50_000, limit=None):
    """Yield string-only DataFrames, checking exactly the rows consumed.

    ``limit`` caps data rows, excluding the header; None reads the entire file.
    Empty and whitespace-only fields are preserved, including empty match lists.
    A blank physical record is malformed; an explicitly empty field is valid.
    """
    if chunk_size <= 0 or (limit is not None and limit <= 0):
        raise ValueError("chunk_size and limit must be positive")
    path = Path(path)
    columns = tuple(expected_columns)
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.reader(stream, delimiter="\t", strict=True)
        try:
            header = next(reader, None)
            if header != list(columns):
                raise DataFormatError(
                    f"{path}: unexpected header {header!r}; expected {list(columns)!r}"
                )
            rows = []
            count = 0
            while limit is None or count < limit:
                row = next(reader, None)
                if row is None:
                    break
                if len(row) != len(columns):
                    raise DataFormatError(
                        f"{path}: line {reader.line_num}: expected {len(columns)} "
                        f"fields, found {len(row)}"
                    )
                rows.append(row)
                count += 1
                if len(rows) == chunk_size:
                    yield pd.DataFrame(rows, columns=columns, dtype=str)
                    rows = []
            if rows:
                yield pd.DataFrame(rows, columns=columns, dtype=str)
        except csv.Error as exc:
            raise DataFormatError(f"{path}: line {reader.line_num}: {exc}") from exc


def read_tsv(path, expected_columns, *, limit=None, chunk_size=50_000):
    """Materialize a table. Use a limit for samples; use iter_tsv for large scans."""
    chunks = list(iter_tsv(path, expected_columns, chunk_size=chunk_size, limit=limit))
    if not chunks:
        return pd.DataFrame(columns=expected_columns, dtype=str)
    return pd.concat(chunks, ignore_index=True)


def blank_mask(values):
    """Identify empty/whitespace-only raw strings without altering them."""
    return values.str.strip().eq("")


def write_tsv(frame, path, expected_columns):
    """Write raw string fields with UTF-8 and standard TSV quoting."""
    if tuple(frame.columns) != tuple(expected_columns):
        raise DataFormatError(f"Unexpected output header: {list(frame.columns)!r}")
    for name in frame.columns:
        if not frame[name].map(lambda value: isinstance(value, str)).all():
            raise DataFormatError(f"Column {name!r} must contain strings, including explicit empty strings")
    frame.to_csv(Path(path), sep="\t", index=False, encoding="utf-8", lineterminator="\n")
