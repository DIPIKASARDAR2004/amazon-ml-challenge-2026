"""Legacy training demonstration, not held-out evaluation or the official scorer."""

import argparse

from src.config import add_dataset_argument, positive_int


def run_demo(dataset_dir, *, sample_rows=5000, chunk_size=100_000, seed=42):
    import lightgbm as lgb
    import numpy as np
    import pandas as pd
    from rapidfuzz import fuzz
    from sklearn.metrics import fbeta_score, precision_score, recall_score
    from src.data_loading import GROUND_TRUTH_COLUMNS, SOURCE_COLUMNS, iter_tsv, read_tsv

    print("Training demonstration only: scores below use training pairs, not held-out businesses.")
    print("Random negatives are the legacy method and may contradict truth; replace them in Pass 3.")
    train_dir = dataset_dir / "train"
    source1 = read_tsv(train_dir / "train_source1_sampled.tsv", SOURCE_COLUMNS, limit=sample_rows)
    source_ids = set(source1["entity_id"])
    truth_parts = []
    for chunk in iter_tsv(train_dir / "train_ground_truth_sampled.tsv", GROUND_TRUTH_COLUMNS, chunk_size=chunk_size):
        selected = chunk[chunk["source1_entity_id"].isin(source_ids)]
        if not selected.empty:
            truth_parts.append(selected)
    if source1.empty or not truth_parts:
        raise ValueError("No labeled Source 1 records found for the training sample.")
    truth = pd.concat(truth_parts, ignore_index=True)
    # The challenge format is comma-separated IDs, not a Python-list string.
    truth["matched_entity_ids"] = truth["matched_entity_ids"].map(lambda text: text.split(",") if text else [])
    positives = [(row.source1_entity_id, target, 1)
                 for row in truth.itertuples(index=False) for target in row.matched_entity_ids]
    if not positives:
        raise ValueError("The training demo requires at least one labeled positive pair.")
    rng = np.random.RandomState(seed)
    source_array = source1["entity_id"].to_numpy()
    all_targets = np.array([target for targets in truth["matched_entity_ids"] for target in targets])
    negatives = [(rng.choice(source_array), rng.choice(all_targets), 0) for _ in range(len(positives) * 2)]
    pairs = pd.DataFrame(positives + negatives, columns=["s1_id", "target_id", "label"])
    wanted = set(pairs["target_id"])
    target_records = {}
    for number in (2, 3):
        ids = {target for target in wanted if target.startswith(f"S{number}-")}
        for chunk in iter_tsv(train_dir / f"train_source{number}.tsv", SOURCE_COLUMNS, chunk_size=chunk_size):
            for row in chunk.loc[chunk["entity_id"].isin(ids)].to_dict("records"):
                target_records[row["entity_id"]] = row
    source_records = source1.set_index("entity_id").to_dict("index")
    features, labels = [], []
    for row in pairs.itertuples(index=False):
        left = source_records.get(row.s1_id)
        right = target_records.get(row.target_id)
        if left is None or right is None:
            continue
        features.append([
            fuzz.ratio(left["business_name"].lower(), right["business_name"].lower()),
            fuzz.ratio(left["business_address"].lower(), right["business_address"].lower()),
        ])
        labels.append(row.label)
    if not features:
        raise ValueError("No training pairs had text available in both sources.")
    x, y = np.array(features), np.array(labels)
    model = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.05, random_state=seed)
    model.fit(x, y)
    predictions = model.predict(x)
    print("\nTRAINING-PAIR DIAGNOSTICS — not competition validation")
    print(f"Precision: {precision_score(y, predictions, zero_division=0):.4f}")
    print(f"Recall: {recall_score(y, predictions, zero_division=0):.4f}")
    print(f"Pair-level F0.5: {fbeta_score(y, predictions, beta=0.5, zero_division=0):.4f}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    add_dataset_argument(parser)
    parser.add_argument("--sample-rows", type=positive_int, default=5000)
    parser.add_argument("--chunk-size", type=positive_int, default=100_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    try:
        run_demo(args.dataset_dir, sample_rows=args.sample_rows, chunk_size=args.chunk_size, seed=args.seed)
    except (OSError, ValueError) as exc:
        parser.exit(1, f"Training demo failed: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
