"""Audit corrupted inputs and freeze entity-level, stratified experiment IDs."""

import csv
import json
from pathlib import Path

import pytest

from src.data_loading import DataFormatError
from src.splits import _sample_quotas, prepare_experiment


def write_rows(path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)


@pytest.fixture
def dataset(tmp_path):
    root = tmp_path / "dataset"
    source_columns = ["entity_id", "business_name", "business_address", "country"]
    truth_columns = ["source1_entity_id", "matched_entity_ids"]
    original, truth, training, validation, train_truth, val_truth = [], [], [], [], [], []
    targets = {2: [], 3: []}
    for country in ("US", "India"):
        for status in ("match", "empty"):
            for index in range(3):
                suffix = f"{country}-{status}-{index}"
                source = [f"S1-{suffix}", "NA" if index == 0 else "Example", "1 Example Road", country]
                target = f"S{2 + index % 2}-{suffix}" if status == "match" else ""
                label = [source[0], target]
                original.append(source)
                truth.append(label)
                (training if index == 0 else validation).append(source)
                (train_truth if index == 0 else val_truth).append(label)
                if target:
                    targets[2 + index % 2].append([target, "Example", "1 Example Road", country])
    for name, rows, columns in [
        ("train/train_source1.tsv", original, source_columns),
        ("train/train_ground_truth.tsv", truth[::-1], truth_columns),
        ("train/train_source1_sampled.tsv", training, source_columns),
        ("val/val_source1.tsv", validation[::-1], source_columns),
        ("train/train_ground_truth_sampled.tsv", train_truth[::-1], truth_columns),
        ("val/val_ground_truth.tsv", val_truth, truth_columns),
        ("train/train_source2.tsv", targets[2], source_columns),
        ("train/train_source3.tsv", targets[3], source_columns),
    ]:
        write_rows(root / name, columns, rows)
    return root


def prepare(dataset, output, **options):
    return prepare_experiment(dataset, output, chunk_size=2, train_sample_size=4,
                              development_sample_size=4, evaluation_sample_size=4, **options)


def ids(path):
    with path.open() as stream:
        return {row["source1_entity_id"] for row in csv.DictReader(stream, delimiter="\t")}


def change_rows(path, transform):
    with path.open(newline="") as stream:
        reader = csv.reader(stream, delimiter="\t")
        header, rows = next(reader), list(reader)
    write_rows(path, header, transform(rows))


def test_full_audit_and_stratified_disjoint_manifests(dataset, tmp_path):
    output = tmp_path / "experiment"
    manifest = prepare(dataset, output)
    groups = {name: ids(output / f"{name}_ids.tsv") for name in ("train", "development", "evaluation")}
    assert [len(groups[name]) for name in groups] == [4, 4, 4]
    assert not (groups["train"] & groups["development"] or groups["train"] & groups["evaluation"] or groups["development"] & groups["evaluation"])
    assert set.union(*groups.values()) == ids(dataset / "train/train_ground_truth.tsv")
    for name in groups:
        assert ids(output / f"sample_{name}_ids.tsv") <= groups[name]
        assert len(manifest["files"][f"sample_{name}_ids.tsv"]["strata"]) == 4
    assert manifest["seed"] == 42
    assert manifest["target_pool"]["records"] == 6
    assert json.loads((output / "audit.json").read_text())["status"] == "pass"
    assert not list(output.glob("*.sqlite*"))


def test_order_independent_reproducibility_and_immutable_output(dataset, tmp_path):
    first, second = tmp_path / "first", tmp_path / "second"
    prepare(dataset, first)
    for path in dataset.glob("*/*.tsv"):
        change_rows(path, lambda rows: rows[::-1])
    prepare(dataset, second)
    for path in first.glob("*ids.tsv"):
        assert path.read_bytes() == (second / path.name).read_bytes()
    with pytest.raises(FileExistsError):
        prepare(dataset, first)


@pytest.mark.parametrize("relative,operation,error", [
    ("train/train_source1.tsv", lambda rows: rows + rows[:1], "duplicate"),
    ("train/train_source2.tsv", lambda rows: rows + rows[:1], "duplicate"),
    ("train/train_source1_sampled.tsv", lambda rows: rows + rows[:1], "split"),
    ("val/val_source1.tsv", lambda rows: rows[1:], "coverage"),
    ("train/train_ground_truth.tsv", lambda rows: rows + rows[:1], "ground.truth"),
    ("train/train_ground_truth.tsv", lambda rows: rows[1:], "coverage"),
    ("val/val_ground_truth.tsv", lambda rows: rows[1:], "coverage"),
])
def test_audit_rejects_duplicates_and_missing_rows(dataset, tmp_path, relative, operation, error):
    change_rows(dataset / relative, operation)
    output = tmp_path / "invalid"
    with pytest.raises(DataFormatError, match=error):
        prepare(dataset, output)
    assert not list(output.glob("*ids.tsv"))
    assert json.loads((output / "audit.json").read_text())["status"] == "fail"


def test_overlap_between_legacy_partitions_is_rejected(dataset, tmp_path):
    with (dataset / "train/train_source1_sampled.tsv").open() as stream:
        row = list(csv.reader(stream, delimiter="\t"))[1]
    change_rows(dataset / "val/val_source1.tsv", lambda rows: rows + [row])
    with pytest.raises(DataFormatError, match="split"):
        prepare(dataset, tmp_path / "overlap")


@pytest.mark.parametrize("relative,column,replacement,error", [
    ("train/train_source1_sampled.tsv", 1, "Changed name", "split"),
    ("train/train_ground_truth.tsv", 1, "S2-nonexistent", "target"),
    ("train/train_ground_truth.tsv", 1, "S2-duplicate,S2-duplicate", "duplicate"),
    ("train/train_ground_truth.tsv", 1, "S1-not-a-target", "ID"),
    ("train/train_ground_truth.tsv", 0, "S1-nonexistent", "ground.truth"),
    ("train/train_ground_truth_sampled.tsv", 1, "S2-other", "ground.truth"),
    ("train/train_source2.tsv", 0, "S3-wrong-file", "ID"),
])
def test_audit_rejects_changed_records_and_invalid_labels(dataset, tmp_path, relative, column, replacement, error):
    def mutate(rows):
        rows[0][column] = replacement
        return rows
    change_rows(dataset / relative, mutate)
    with pytest.raises(DataFormatError, match=error):
        prepare(dataset, tmp_path / "bad-label")


def test_sample_size_must_allow_all_available_strata(dataset, tmp_path):
    with pytest.raises(ValueError, match="strata"):
        prepare_experiment(dataset, tmp_path / "small", train_sample_size=2)


def test_output_cannot_be_inside_dataset(dataset):
    with pytest.raises(ValueError, match="outside"):
        prepare(dataset, dataset / "experiment")


def test_shared_target_cannot_leak_between_reference_businesses(dataset, tmp_path):
    def reuse_target(rows):
        rows[0][1] = "S2-US-match-0"
        return rows
    change_rows(dataset / "train/train_ground_truth.tsv", reuse_target)
    with pytest.raises(DataFormatError, match="ownership"):
        prepare(dataset, tmp_path / "shared-target")


def test_seed_changes_assignment_but_chunk_size_does_not(dataset, tmp_path):
    first, second, third = (tmp_path / name for name in ("seed42", "chunk1", "seed7"))
    prepare(dataset, first)
    prepare_experiment(dataset, second, chunk_size=1)
    prepare(dataset, third, seed=7)
    for path in first.glob("*ids.tsv"):
        assert path.read_bytes() == (second / path.name).read_bytes()
    assert (first / "development_ids.tsv").read_bytes() != (third / "development_ids.tsv").read_bytes()


def test_rare_stratum_is_reported_instead_of_silently_claiming_full_stratification(dataset, tmp_path):
    removed = "S1-US-empty-2"
    for relative in ("train/train_source1.tsv", "train/train_ground_truth.tsv", "val/val_source1.tsv", "val/val_ground_truth.tsv"):
        change_rows(dataset / relative, lambda rows: [row for row in rows if row[0] != removed])
    manifest = prepare(dataset, tmp_path / "rare")
    assert len(manifest["warnings"]) == 1
    assert manifest["files"]["development_ids.tsv"]["rows"] == 4
    assert manifest["files"]["evaluation_ids.tsv"]["rows"] == 3


@pytest.mark.parametrize("sizes,requested", [([90, 10], 10), ([100, 2, 1, 1], 20), ([1, 1], 100)])
def test_sample_quotas_include_rare_groups_and_respect_available_counts(sizes, requested):
    strata = [(str(index), False, count) for index, count in enumerate(sizes)]
    quotas = _sample_quotas(strata, requested)
    assert sum(quotas.values()) == min(sum(sizes), requested)
    assert all(1 <= quotas[(country, singleton)] <= count for country, singleton, count in strata)


def test_ground_truth_list_order_does_not_change_its_meaning(dataset, tmp_path):
    extra = "S3-extra"
    change_rows(dataset / "train/train_source3.tsv", lambda rows: rows + [[extra, "Extra", "1 Example Road", "US"]])
    for relative, text in [("train/train_ground_truth.tsv", "S2-US-match-0,S3-extra"),
                           ("train/train_ground_truth_sampled.tsv", "S3-extra,S2-US-match-0")]:
        def change(rows):
            for row in rows:
                if row[0] == "S1-US-match-0":
                    row[1] = text
            return rows
        change_rows(dataset / relative, change)
    manifest = prepare(dataset, tmp_path / "reordered-labels")
    assert manifest["target_pool"]["records"] == 7
