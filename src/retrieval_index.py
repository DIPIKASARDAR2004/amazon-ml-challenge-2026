"""Disk-backed target retrieval for Pass 4; no labels enter indexing or search."""

import argparse
from collections import Counter
from contextlib import closing
from datetime import datetime, timezone
from functools import lru_cache
import json
from pathlib import Path
import sqlite3
import tempfile
from time import monotonic
import unicodedata

from rapidfuzz import fuzz

from src.config import add_dataset_argument, positive_int
from src.data_loading import SOURCE_COLUMNS, DataFormatError, iter_tsv
from src.evaluation_io import fingerprint_file, validate_id
from src.inspect_data import peak_process_memory_bytes
from src.normalization import NORMALIZATION, normalize_text


INDEX_VERSION = 1
TOKENIZER = "unicode61 remove_diacritics 0 categories 'L* N* Co M*'"
CHANNELS = ("exact_name", "exact_address", "name_words", "address_words")
SCHEMA = f"""
CREATE TABLE records (
    id INTEGER PRIMARY KEY, entity_id TEXT NOT NULL UNIQUE,
    business_name TEXT NOT NULL, business_address TEXT NOT NULL, country TEXT NOT NULL,
    name TEXT NOT NULL, address TEXT NOT NULL
);
CREATE VIRTUAL TABLE search USING fts5(name,address,content='records',content_rowid='id',
    tokenize="{TOKENIZER}");
CREATE VIRTUAL TABLE vocabulary USING fts5vocab(search,col);
PRAGMA user_version={INDEX_VERSION};
"""


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def word_tokens(text):
    """Retain letters/numbers/marks across scripts; punctuation splits tokens."""
    tokens, current = [], []
    for char in normalize_text(text):
        if unicodedata.category(char)[0] in "LNM" or unicodedata.category(char) == "Co":
            current.append(char)
        elif current:
            tokens.append("".join(current)); current = []
    if current:
        tokens.append("".join(current))
    return sorted(set(tokens))


def build_index(dataset_dir, output_dir, *, chunk_size=20_000, limit_per_source=None, progress=None):
    """Index every supplied training target, or explicitly label a limited pilot.

    Input batches and SQLite's 64 MiB page cache bound working memory. Original
    text is stored once in records; FTS uses external content. Temporary output
    is published only on success. Source ID uniqueness is enforced on disk.
    """
    if chunk_size <= 0 or (limit_per_source is not None and limit_per_source <= 0):
        raise ValueError("Chunk size and optional limit must be positive")
    dataset_dir, output_dir = Path(dataset_dir).resolve(), Path(output_dir).resolve()
    if dataset_dir == output_dir or dataset_dir in output_dir.parents:
        raise ValueError("Index output must be outside the dataset directory")
    if output_dir.exists():
        raise FileExistsError("Index output already exists; choose a new version directory")
    started, started_at = monotonic(), datetime.now(timezone.utc).isoformat()
    paths = {f"source{number}": dataset_dir / "train" / f"train_source{number}.tsv" for number in (2, 3)}
    inputs = {key: {"path": str(path), **fingerprint_file(path)} for key, path in paths.items()}
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    counts, countries = {}, Counter()
    with tempfile.TemporaryDirectory(prefix=".retrieval-build-", dir=output_dir.parent) as temporary:
        staged = Path(temporary) / "result"
        staged.mkdir()
        db_path = staged / "targets.sqlite"
        with closing(sqlite3.connect(db_path)) as connection:
            connection.execute("PRAGMA journal_mode=OFF")  # Disposable staged build only.
            connection.execute("PRAGMA synchronous=OFF")
            connection.execute("PRAGMA cache_size=-65536")
            connection.execute("PRAGMA temp_store=FILE")
            connection.executescript(SCHEMA)
            total = 0
            for number in (2, 3):
                count = 0
                for chunk in iter_tsv(paths[f"source{number}"], SOURCE_COLUMNS,
                                      chunk_size=chunk_size, limit=limit_per_source):
                    rows = []
                    for entity_id, name, address, country in chunk.itertuples(index=False, name=None):
                        validate_id(entity_id, (f"S{number}-",))
                        rows.append((entity_id, name, address, country, normalize_text(name), normalize_text(address)))
                        countries[country] += 1
                    try:
                        connection.executemany("INSERT INTO records(entity_id,business_name,business_address,country,name,address) VALUES (?,?,?,?,?,?)", rows)
                    except sqlite3.IntegrityError as exc:
                        raise DataFormatError("Duplicate target IDs while building index") from exc
                    connection.execute("INSERT INTO search(rowid,name,address) SELECT id,name,address FROM records WHERE id > ?", (total,))
                    count += len(rows); total += len(rows)
                    connection.commit()
                    if progress:
                        progress(f"Indexed {total:,} targets ({monotonic() - started:.1f}s)")
                counts[f"source{number}"] = count
            if progress:
                progress("Building exact-name/address indexes and token frequencies")
            connection.execute("CREATE INDEX records_name ON records(name,entity_id)")
            connection.execute("CREATE INDEX records_address ON records(address,entity_id)")
            connection.execute("CREATE TABLE term_frequency(term TEXT, field TEXT, documents INTEGER NOT NULL, PRIMARY KEY(term,field)) WITHOUT ROWID")
            connection.execute("INSERT INTO term_frequency SELECT term,col,doc FROM vocabulary")
            connection.commit()
            connection.execute("INSERT INTO search(search) VALUES ('integrity-check')")
            connection.commit()
            if connection.execute("PRAGMA quick_check").fetchone()[0] != "ok":
                raise DataFormatError("SQLite integrity check failed")
            terms = connection.execute("SELECT count(*) FROM term_frequency").fetchone()[0]
        # Verify the source bytes were stable throughout the build.
        for key, path in paths.items():
            if fingerprint_file(path) != {k: inputs[key][k] for k in ("sha256", "bytes")}:
                raise DataFormatError("Source changed during index construction")
        metadata = {
            "schema_version": INDEX_VERSION, "status": "complete", "normalization": NORMALIZATION,
            "tokenizer": TOKENIZER, "sqlite_version": sqlite3.sqlite_version,
            "scope": "full_training_targets" if limit_per_source is None else "restricted_first_rows_pilot",
            "limit_per_source": limit_per_source, "chunk_size": chunk_size, "inputs": inputs,
            "counts": counts, "total_targets": sum(counts.values()), "countries": dict(countries),
            "term_field_entries": terms, "started_at_utc": started_at,
            "elapsed_seconds": monotonic() - started, "peak_process_rss_bytes": peak_process_memory_bytes(),
            "database": fingerprint_file(db_path),
        }
        _write_json(staged / "index.json", metadata)
        staged.rename(output_dir)
    return metadata


def retrieval_config(*, per_channel=50, max_candidates=100, max_terms=4, max_token_documents=20_000):
    values = locals().copy()
    if any(isinstance(v, bool) or not isinstance(v, int) or v <= 0 for v in values.values()):
        raise ValueError("Retrieval limits must be positive integers")
    return {"version": 1, **values, "channels": list(CHANNELS), "country_filter": False,
            "word_selection": "rarest indexed tokens of length >= 2; OR within field",
            "word_order": "bm25 then target ID", "final_order": "similarity descending then target ID",
            "similarity": "max(name_ratio,address_ratio) + 0.1 * min(name_ratio,address_ratio)"}


class DiskCandidateIndex:
    def __init__(self, directory, *, config=None, verify_database=True):
        directory = Path(directory).resolve()
        self.metadata = json.loads((directory / "index.json").read_text(encoding="utf-8"))
        if (self.metadata.get("schema_version") != INDEX_VERSION or self.metadata.get("normalization") != NORMALIZATION
                or self.metadata.get("tokenizer") != TOKENIZER or self.metadata.get("status") != "complete"):
            raise DataFormatError("Incompatible or incomplete retrieval index")
        self.config = retrieval_config() if config is None else config
        if self.config != retrieval_config(**{k: self.config[k] for k in ("per_channel", "max_candidates", "max_terms", "max_token_documents")}):
            raise DataFormatError("Incompatible retrieval configuration")
        path = directory / "targets.sqlite"
        if verify_database and fingerprint_file(path) != self.metadata["database"]:
            raise DataFormatError("Retrieval database fingerprint mismatch")
        self.connection = sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA cache_size=-65536")
        self.connection.execute("PRAGMA temp_store=FILE")

    def close(self):
        self._frequency.cache_clear()
        self.connection.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    @lru_cache(maxsize=20_000)
    def _frequency(self, token, field):
        row = self.connection.execute("SELECT documents FROM term_frequency WHERE term=? AND field=?", (token, field)).fetchone()
        return row[0] if row else 0

    def get_records(self, ids):
        ids = sorted(set(ids))
        result = {}
        for start in range(0, len(ids), 500):
            batch = ids[start:start + 500]
            placeholders = ",".join("?" for _ in batch)
            for row in self.connection.execute(f"SELECT * FROM records WHERE entity_id IN ({placeholders})", batch):
                result[row["entity_id"]] = dict(row)
        if set(result) != set(ids):
            raise DataFormatError("Unknown target IDs requested from retrieval index")
        return result

    def _field_channels(self, text, field):
        if not text:
            return [], [], []
        exact = [row[0] for row in self.connection.execute(
            f"SELECT entity_id FROM records WHERE {field}=? ORDER BY entity_id LIMIT ?",
            (text, self.config["per_channel"]))]
        terms = [(self._frequency(token, field), token) for token in word_tokens(text) if len(token) >= 2]
        chosen = [token for count, token in sorted(terms) if 0 < count <= self.config["max_token_documents"]][:self.config["max_terms"]]
        if not chosen:
            return exact, [], chosen
        # Tokens contain no quotes/operators, and the expression itself is a bound value.
        expression = field + " : (" + " OR ".join('"' + token + '"' for token in chosen) + ")"
        words = [row[0] for row in self.connection.execute(
            "SELECT r.entity_id, bm25(search) AS score FROM search JOIN records r ON r.id=search.rowid "
            "WHERE search MATCH ? ORDER BY score,r.entity_id LIMIT ?", (expression, self.config["per_channel"]))]
        return exact, words, chosen

    def retrieve_one(self, query):
        validate_id(query["entity_id"])
        name, address = normalize_text(query["business_name"]), normalize_text(query["business_address"])
        exact_name, name_words, name_terms = self._field_channels(name, "name")
        exact_address, address_words, address_terms = self._field_channels(address, "address")
        channels = dict(zip(CHANNELS, (exact_name, exact_address, name_words, address_words)))
        pool = set().union(*channels.values())
        targets = self.get_records(pool)
        similarities = {}
        for entity_id, target in targets.items():
            n = fuzz.ratio(name, target["name"]) / 100 if name and target["name"] else 0.0
            a = fuzz.ratio(address, target["address"]) / 100 if address and target["address"] else 0.0
            similarities[entity_id] = (n, a)
        variants = {}
        definitions = {"name_only": ("exact_name", "name_words"),
                       "exact_name_address": ("exact_name", "exact_address"), "hybrid": CHANNELS}
        for variant, routes in definitions.items():
            ids = set().union(*(set(channels[route]) for route in routes))
            def priority(entity_id):
                n, a = similarities[entity_id]
                score = n if variant == "name_only" else max(n, a) + 0.1 * min(n, a)
                return (-score, entity_id)
            variants[variant] = sorted(ids, key=priority)[:self.config["max_candidates"]]
        trace = [{"target_entity_id": entity_id, "rank": rank,
                  "retrieval_score": max(similarities[entity_id]) + 0.1 * min(similarities[entity_id]),
                  "channels": ",".join(route for route in CHANNELS if entity_id in channels[route])}
                 for rank, entity_id in enumerate(variants["hybrid"], 1)]
        return {"variants": variants, "channels": channels, "union_before_cap": sorted(pool),
                "trace": trace, "selected_terms": {"name": name_terms, "address": address_terms}}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    add_dataset_argument(parser)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--chunk-size", type=positive_int, default=20_000)
    parser.add_argument("--limit-per-source", type=positive_int, help="Explicit restricted pilot only; omit for complete indexing")
    args = parser.parse_args(argv)
    last_print = [0.0]
    def progress(message):
        now = monotonic()
        if now - last_print[0] >= 10 or not message.startswith("Indexed"):
            print(message, flush=True); last_print[0] = now
    try:
        report = build_index(args.dataset_dir, args.output_dir, chunk_size=args.chunk_size,
                             limit_per_source=args.limit_per_source, progress=progress)
    except (OSError, ValueError, sqlite3.Error) as exc:
        parser.exit(1, f"Retrieval index failed: {exc}\n")
    print(f"Index complete: {report['total_targets']:,} targets; {report['scope']}; {report['elapsed_seconds']:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
