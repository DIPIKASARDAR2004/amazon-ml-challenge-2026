import json

import pandas as pd
import pytest

from src.data_loading import DataFormatError
from src.evaluation_io import fingerprint_file
from src.retrieval_evaluation import evaluate_retrieval, summarize_retrieval
from src.retrieval_index import DiskCandidateIndex, build_index, retrieval_config
from src.scoring import load_id_lists


def test_retrieval_summary_uses_full_truth_and_empty_queries():
    truth = {"S1-a": frozenset(["S2-a", "S3-b"]), "S1-empty": frozenset()}
    summary = summarize_retrieval(truth, {"S1-a": ["S2-a"], "S1-empty": []})
    assert summary["pair_recall"] == 0.5
    assert summary["oracle_macro_f0_5"] == pytest.approx((5 / 6 + 1) / 2)
    assert summary["empty_queries"] == 1 and summary["mean_candidates"] == 0.5
    assert summary["positive_queries_all_links_retrieved"] == 0
    with pytest.raises(DataFormatError, match="coverage"):
        summarize_retrieval(truth, {"S1-a": ["S2-a"]})


def test_full_fixture_benchmark_persists_exact_candidates_and_separate_diagnostics(tmp_path, retrieval_dataset, retrieval_experiment):
    index = tmp_path / "index"
    build_index(retrieval_dataset, index)
    output = tmp_path / "results"
    report = evaluate_retrieval(retrieval_dataset, retrieval_experiment, index, output)
    assert report["stage"] == "candidate_retrieval_only_no_model_scoring"
    assert report["target_records"] == 14
    assert report["variants"]["hybrid"]["queries"] == 10
    candidates = load_id_lists(output / "candidate_pairs.tsv", column="candidate_entity_ids")
    assert candidates["S1-tr09"] == frozenset()
    assert "S3-tr03" in candidates["S1-tr03"]
    trace = pd.read_csv(output / "retrieval_trace.tsv", sep="\t", keep_default_na=False)
    assert not trace.duplicated(["source1_entity_id", "target_entity_id"]).any()
    assert set(trace[["source1_entity_id", "target_entity_id"]].itertuples(index=False, name=None)) == {(q,t) for q,ts in candidates.items() for t in ts}
    assert not (output / "matching_results.tsv").exists()
    for path, expected in report["outputs"].items():
        assert fingerprint_file(output / path) == expected
    with pytest.raises(FileExistsError):
        evaluate_retrieval(retrieval_dataset, retrieval_experiment, index, output)


def test_evaluation_answers_are_not_read_during_search_and_do_not_change_candidates(tmp_path, retrieval_dataset, retrieval_experiment, monkeypatch):
    import src.retrieval_evaluation as evaluation

    index = tmp_path / "index"
    build_index(retrieval_dataset, index)
    original_load = evaluation.load_id_lists
    answer_read = [False]
    original_retrieve = DiskCandidateIndex.retrieve_one
    def observed_load(path, **kwargs):
        if str(path).endswith("train_ground_truth.tsv"):
            answer_read[0] = True
        return original_load(path, **kwargs)
    def observed_search(self, query):
        assert not answer_read[0]
        return original_retrieve(self, query)
    monkeypatch.setattr(evaluation, "load_id_lists", observed_load)
    monkeypatch.setattr(DiskCandidateIndex, "retrieve_one", observed_search)
    first = evaluate_retrieval(retrieval_dataset, retrieval_experiment, index, tmp_path / "first")
    assert answer_read[0]
    truth = retrieval_dataset / "train/train_ground_truth.tsv"
    truth.write_text(truth.read_text().replace("S1-tr01\tS2-tr01,S3-tr01", "S1-tr01\t"))
    manifest_path = retrieval_experiment / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["inputs"]["ground_truth"] = fingerprint_file(truth)
    manifest_path.write_text(json.dumps(manifest))
    answer_read[0] = False
    second = evaluate_retrieval(retrieval_dataset, retrieval_experiment, index, tmp_path / "second")
    assert first["variants"]["hybrid"]["true_links"] != second["variants"]["hybrid"]["true_links"]
    assert (tmp_path / "first/candidate_pairs.tsv").read_bytes() == (tmp_path / "second/candidate_pairs.tsv").read_bytes()


@pytest.mark.parametrize("kind", ["restricted", "sample", "source", "target_count", "target_hash"])
def test_incompatible_experiments_fail_without_results(tmp_path, retrieval_dataset, retrieval_experiment, kind):
    index = tmp_path / "index"
    build_index(retrieval_dataset, index, limit_per_source=2 if kind == "restricted" else None)
    if kind == "sample":
        with (retrieval_experiment / "sample_development_ids.tsv").open("a") as stream:
            stream.write("S1-evaluation-must-not-enter\n")
    elif kind == "source":
        with (retrieval_dataset / "train/train_source1.tsv").open("a") as stream:
            stream.write("changed\n")
    elif kind in ("target_count", "target_hash"):
        path = retrieval_experiment / "manifest.json"
        metadata = json.loads(path.read_text())
        if kind == "target_count":
            metadata["target_pool"]["records"] = 999
        else:
            metadata["inputs"]["source2"]["sha256"] = "different"
        path.write_text(json.dumps(metadata))
    with pytest.raises(DataFormatError):
        evaluate_retrieval(retrieval_dataset, retrieval_experiment, index, tmp_path / "bad")
    assert not (tmp_path / "bad").exists()


def test_capped_out_true_links_remain_in_recall_and_failure_report(tmp_path, retrieval_dataset, retrieval_experiment):
    index = tmp_path / "index"
    build_index(retrieval_dataset, index)
    report = evaluate_retrieval(retrieval_dataset, retrieval_experiment, index, tmp_path / "result",
                                config=retrieval_config(max_candidates=1))
    assert report["true_link_diagnostics"]["true_links_lost_at_final_cap"] >= 1
    assert report["variants"]["hybrid"]["pair_recall"] < 1
    misses = pd.read_csv(tmp_path / "result/missed_links.tsv", sep="\t")
    assert "final_candidate_cap" in set(misses.miss_stage)


def test_failure_after_candidate_export_does_not_publish_partial_results(tmp_path, retrieval_dataset, retrieval_experiment):
    index = tmp_path / "index"
    build_index(retrieval_dataset, index)
    truth = retrieval_dataset / "train/train_ground_truth.tsv"
    truth.write_text(truth.read_text().replace("S2-tr01", "S2-unknown"))
    path = retrieval_experiment / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["inputs"]["ground_truth"] = fingerprint_file(truth)
    path.write_text(json.dumps(manifest))
    with pytest.raises(DataFormatError, match="Unknown target"):
        evaluate_retrieval(retrieval_dataset, retrieval_experiment, index, tmp_path / "failed")
    assert not (tmp_path / "failed").exists()
    assert not list(tmp_path.glob(".retrieval-eval-*"))
