"""Check user-facing commands from outside the repository and safe imports."""

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures" / "dataset"


def run_python(arguments, cwd):
    env = dict(os.environ, PYTHONPATH=str(ROOT), PYTHONDONTWRITEBYTECODE="1")
    return subprocess.run(
        [sys.executable, "-B", *arguments], cwd=cwd, env=env,
        text=True, capture_output=True, timeout=30,
    )


def test_imports_do_not_read_data_train_print_or_write(tmp_path):
    result = run_python(["-c", """
import importlib
from unittest.mock import patch
import pandas
import lightgbm
import src.data_loading

def forbidden(*args, **kwargs):
    raise AssertionError('Import attempted data access or training')

with patch('pandas.read_csv', forbidden), \
     patch('src.data_loading.iter_tsv', forbidden), \
     patch('src.data_loading.read_tsv', forbidden), \
     patch('src.data_loading.write_tsv', forbidden), \
     patch('pathlib.Path.open', forbidden), \
     patch('lightgbm.LGBMClassifier.fit', forbidden):
    for name in ['candidate_generation', 'split_validation', 'train_model',
                 'inspect_dataset', 'src.inspect_data', 'utils.validate_submission',
                 'src.evaluation_io', 'src.scoring', 'src.splits',
                 'src.normalization', 'src.candidates', 'src.pair_labels',
                 'src.features', 'src.model_io', 'src.submission', 'src.pipeline',
                 'src.retrieval_index', 'src.retrieval_evaluation']:
        importlib.import_module(name)
"""], tmp_path)
    assert result.returncode == 0, result.stderr
    assert result.stdout == ""
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("script", [
    "inspect_dataset.py", "src/inspect_data.py", "candidate_generation.py",
    "split_validation.py", "train_model.py", "utils/validate_submission.py",
])
def test_help_does_not_need_a_dataset_or_create_files(tmp_path, script):
    result = run_python([str(ROOT / script), "--help"], tmp_path)
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout.lower()
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("entrypoint", [["-m", "src.inspect_data"], [str(ROOT / "inspect_dataset.py")]])
def test_custom_dataset_and_report_paths_work_outside_repo(tmp_path, entrypoint):
    report_path = tmp_path / "reports" / "fixture.json"
    result = run_python([
        *entrypoint, "--dataset-dir", str(FIXTURES), "--full", "--chunk-size", "2",
        "--report", str(report_path),
    ], tmp_path)
    assert result.returncode == 0, result.stderr
    report = json.loads(report_path.read_text())
    assert report["mode"] == "full"
    assert report["dataset_dir"] == str(FIXTURES)
    assert report["sources"]["train_source1.tsv"]["records"] == 5
    assert report["ground_truth"]["zero_matches"] == 3
    assert report["chunk_size"] == 2
    assert report["started_at_utc"]
    assert report["environment"]["python"]


def test_sample_report_cannot_be_mistaken_for_full_inspection(tmp_path):
    report_path = tmp_path / "sample.json"
    result = run_python([
        "-m", "src.inspect_data", "--dataset-dir", str(FIXTURES),
        "--sample-rows", "1", "--report", str(report_path),
    ], tmp_path)
    assert result.returncode == 0, result.stderr
    report = json.loads(report_path.read_text())
    assert report["mode"] == "sample" and report["rows_per_file_limit"] == 1
    assert all(source["records"] == 1 for source in report["sources"].values())
    assert report["ground_truth"]["records"] == 1


def test_missing_dataset_fails_without_creating_a_success_report(tmp_path):
    report_path = tmp_path / "should_not_exist.json"
    result = run_python([
        "-m", "src.inspect_data", "--dataset-dir", str(tmp_path / "missing"),
        "--report", str(report_path),
    ], tmp_path)
    assert result.returncode != 0
    assert "not found" in result.stderr.lower()
    assert not report_path.exists()


def test_report_path_cannot_replace_dataset_input(tmp_path):
    result = run_python([
        "-m", "src.inspect_data", "--dataset-dir", str(FIXTURES),
        "--report", str(FIXTURES / "train" / "train_source1.tsv"),
    ], tmp_path)
    assert result.returncode == 2
    assert "outside the dataset directory" in result.stderr


def test_default_dataset_path_is_anchored_to_repository(tmp_path):
    result = run_python(["-c", "from src.config import DEFAULT_DATASET_DIR; print(DEFAULT_DATASET_DIR)"], tmp_path)
    assert result.returncode == 0, result.stderr
    assert Path(result.stdout.strip()) == ROOT / "dataset"


@pytest.mark.parametrize("options", [["--chunk-size", "0"], ["--sample-rows", "-1"], ["--full", "--sample-rows", "2"]])
def test_invalid_inspection_options_fail_before_work(tmp_path, options):
    result = run_python(["-m", "src.inspect_data", *options], tmp_path)
    assert result.returncode == 2
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("module", ["src.scoring", "src.splits", "src.pipeline", "src.retrieval_index", "src.retrieval_evaluation"])
def test_pass2_help_is_safe_outside_repository(tmp_path, module):
    result = run_python(["-m", module, "--help"], tmp_path)
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout
    assert list(tmp_path.iterdir()) == []


def test_hand_checked_scoring_command(tmp_path):
    fixture = ROOT / "tests/fixtures/scoring"
    report_path = tmp_path / "score.json"
    result = run_python([
        "-m", "src.scoring", "--truth", str(fixture / "truth.tsv"),
        "--predictions", str(fixture / "predictions.tsv"),
        "--candidates", str(fixture / "candidates.tsv"),
        "--source1", str(fixture / "source1.tsv"), "--report", str(report_path),
    ], tmp_path)
    assert result.returncode == 0, result.stderr
    report = json.loads(report_path.read_text())
    assert report["macro_f0_5"] == pytest.approx(2 / 3)
    assert report["candidate_diagnostics"]["pair_recall"] == 0.8


def test_scoring_cli_filters_truth_to_selected_entities_and_requires_all_predictions(tmp_path):
    fixture = ROOT / "tests/fixtures/scoring"
    selection, predicted = tmp_path / "ids.tsv", tmp_path / "predictions.tsv"
    selection.write_text("source1_entity_id\nS1-C\n")
    predicted.write_text("source1_entity_id\tmatched_entity_ids\nS1-C\t\n")
    command = ["-m", "src.scoring", "--truth", str(fixture / "truth.tsv"),
               "--entities", str(selection), "--predictions", str(predicted),
               "--source1", str(fixture / "source1.tsv")]
    result = run_python(command, tmp_path)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["macro_f0_5"] == 1
    predicted.write_text("source1_entity_id\tmatched_entity_ids\n")
    result = run_python(command, tmp_path)
    assert result.returncode == 1 and "coverage" in result.stderr


def test_pass3_pipeline_command_works_outside_repo(tmp_path):
    output = tmp_path / "fixture_run"
    result = run_python(["-m", "src.pipeline", "--output-dir", str(output)], tmp_path)
    assert result.returncode == 0, result.stderr
    assert "NOT competition accuracy" in result.stdout
    assert json.loads((output / "report.json").read_text())["validation"]["check_ids"] is True
    repeat = run_python(["-m", "src.pipeline", "--output-dir", str(output)], tmp_path)
    assert repeat.returncode == 1 and "already exists" in repeat.stderr
