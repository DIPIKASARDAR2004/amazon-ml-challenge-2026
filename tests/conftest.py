import pandas as pd
import pytest

from src.data_loading import SOURCE_COLUMNS
from src.normalization import prepare_records


@pytest.fixture
def records():
    def build(rows):
        return prepare_records(pd.DataFrame(rows, columns=SOURCE_COLUMNS, dtype=str), ("S1-", "S2-", "S3-"))
    return build


@pytest.fixture
def retrieval_dataset(tmp_path):
    import shutil
    from src.pipeline import DEFAULT_FIXTURE

    destination = tmp_path / "dataset"
    shutil.copytree(DEFAULT_FIXTURE, destination)
    return destination


@pytest.fixture
def retrieval_experiment(tmp_path, retrieval_dataset):
    import json
    from src.data_loading import read_tsv
    from src.evaluation_io import fingerprint_file

    path = tmp_path / "experiment"
    path.mkdir()
    ids = read_tsv(retrieval_dataset / "train/train_source1.tsv", SOURCE_COLUMNS)["entity_id"].tolist()
    sample = path / "sample_development_ids.tsv"
    sample.write_text("source1_entity_id\n" + "\n".join(sorted(ids)) + "\n")
    inputs = {f"source{number}": fingerprint_file(retrieval_dataset / f"train/train_source{number}.tsv") for number in (1, 2, 3)}
    inputs["ground_truth"] = fingerprint_file(retrieval_dataset / "train/train_ground_truth.tsv")
    manifest = {"schema_version": 1, "audit_passed": True, "inputs": inputs,
                "files": {sample.name: {"rows": len(ids), **fingerprint_file(sample)}},
                "target_pool": {"records": 14}}
    (path / "manifest.json").write_text(json.dumps(manifest))
    return path
