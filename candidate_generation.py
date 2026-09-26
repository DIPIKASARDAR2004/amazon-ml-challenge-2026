"""Small TF-IDF candidate demonstration; this is not full-pool retrieval."""

import argparse
from time import monotonic

from src.config import add_dataset_argument, positive_int


def run_demo(dataset_dir, *, query_rows=10, targets_per_source=100_000, neighbors=5):
    import pandas as pd
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.neighbors import NearestNeighbors
    from src.data_loading import SOURCE_COLUMNS, read_tsv

    print("Candidate demonstration: restricted target pool; no accuracy claim.")
    queries = read_tsv(dataset_dir / "val" / "val_source1.tsv", SOURCE_COLUMNS, limit=query_rows)
    candidates = pd.concat([
        read_tsv(dataset_dir / "train" / f"train_source{number}.tsv", SOURCE_COLUMNS, limit=targets_per_source)
        [["entity_id", "business_name"]]
        for number in (2, 3)
    ], ignore_index=True)
    if queries.empty or candidates.empty:
        raise ValueError("The demo requires nonempty query and candidate tables.")
    started = monotonic()
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 3), max_df=0.5, min_df=2)
    candidate_matrix = vectorizer.fit_transform(candidates["business_name"])
    nn = NearestNeighbors(n_neighbors=min(neighbors, len(candidates)), metric="cosine", n_jobs=-1)
    nn.fit(candidate_matrix)
    print(f"TF-IDF matrix: {candidate_matrix.shape}; index built in {monotonic() - started:.2f}s")
    distances, indices = nn.kneighbors(vectorizer.transform(queries["business_name"]))
    for i in range(min(5, len(queries))):
        print(f"\nTarget: {queries.iloc[i]['business_name']} ({queries.iloc[i]['entity_id']})")
        for index, distance in zip(indices[i][:3], distances[i][:3]):
            row = candidates.iloc[index]
            print(f"  {row['business_name']} ({row['entity_id']}) — distance {distance:.3f}")
    print("Candidate demonstration complete; no files written.")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    add_dataset_argument(parser)
    parser.add_argument("--query-rows", type=positive_int, default=10)
    parser.add_argument("--targets-per-source", type=positive_int, default=100_000)
    parser.add_argument("--neighbors", type=positive_int, default=5)
    args = parser.parse_args(argv)
    try:
        run_demo(args.dataset_dir, query_rows=args.query_rows, targets_per_source=args.targets_per_source, neighbors=args.neighbors)
    except (OSError, ValueError) as exc:
        parser.exit(1, f"Candidate demo failed: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
