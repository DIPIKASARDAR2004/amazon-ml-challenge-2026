"""Existing 90/10 split demonstration. Split integrity auditing belongs to Pass 2."""

import argparse
from pathlib import Path

from src.config import add_dataset_argument


def create_split(dataset_dir, *, output_dir, validation_fraction=0.1, seed=42):
    from src.data_loading import GROUND_TRUTH_COLUMNS, SOURCE_COLUMNS, read_tsv, write_tsv

    print("Loading complete Source 1 and ground truth; this split demo materializes both tables.")
    source1 = read_tsv(dataset_dir / "train" / "train_source1.tsv", SOURCE_COLUMNS)
    ground_truth = read_tsv(dataset_dir / "train" / "train_ground_truth.tsv", GROUND_TRUTH_COLUMNS)
    validation = source1.sample(frac=validation_fraction, random_state=seed)
    training = source1.drop(validation.index)
    training_truth = ground_truth[ground_truth["source1_entity_id"].isin(training["entity_id"])]
    validation_truth = ground_truth[ground_truth["source1_entity_id"].isin(validation["entity_id"])]
    (output_dir / "train").mkdir(parents=True, exist_ok=True)
    (output_dir / "val").mkdir(parents=True, exist_ok=True)
    for frame, relative, columns in [
        (training, "train/train_source1_sampled.tsv", SOURCE_COLUMNS),
        (training_truth, "train/train_ground_truth_sampled.tsv", GROUND_TRUTH_COLUMNS),
        (validation, "val/val_source1.tsv", SOURCE_COLUMNS),
        (validation_truth, "val/val_ground_truth.tsv", GROUND_TRUTH_COLUMNS),
    ]:
        write_tsv(frame, output_dir / relative, columns)
    print(f"Training: {len(training):,}; validation: {len(validation):,}; wrote derived split files to {output_dir.resolve()}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    add_dataset_argument(parser)
    parser.add_argument("--output-dir", type=Path, help="Root for train/ and val/ derived files (default: --dataset-dir, retaining the existing layout).")
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    if not 0 < args.validation_fraction < 1:
        parser.error("--validation-fraction must be between 0 and 1")
    try:
        create_split(args.dataset_dir, output_dir=args.output_dir or args.dataset_dir, validation_fraction=args.validation_fraction, seed=args.seed)
    except (OSError, ValueError) as exc:
        parser.exit(1, f"Split failed: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
