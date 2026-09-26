import json

import numpy as np
import pandas as pd
import pytest

from src.candidates import retrieval_settings
from src.data_loading import DataFormatError
from src.features import FEATURE_NAMES
from src.model_io import RELOAD_TOLERANCE, fit_model, load_model, predict_scores, save_model


@pytest.fixture
def trained():
    features = pd.DataFrame([[0] * len(FEATURE_NAMES), [1] * len(FEATURE_NAMES)] * 3,
                            columns=FEATURE_NAMES, dtype=float)
    model = fit_model(features, [0, 1] * 3)
    return model, features


def save(model, directory):
    return save_model(model, directory, retrieval=retrieval_settings(), threshold=0.5, seed=42)


def test_saved_model_learned_distinct_scores_and_reloads_with_same_settings(tmp_path, trained):
    model, features = trained
    before = predict_scores(model, features)
    assert before[1] > before[0]
    expected = save(model, tmp_path / "model")
    restored, metadata = load_model(tmp_path / "model")
    np.testing.assert_allclose(predict_scores(restored, features), before, rtol=0, atol=RELOAD_TOLERANCE)
    assert metadata == expected
    assert predict_scores(restored, features.iloc[:0]).shape == (0,)
    with pytest.raises(DataFormatError, match="order/schema"):
        predict_scores(restored, features.loc[:, list(reversed(FEATURE_NAMES))])
    with pytest.raises(FileExistsError):
        save(model, tmp_path / "model")


@pytest.mark.parametrize("field, value, message", [
    ("features", list(reversed(FEATURE_NAMES)), "schema"),
    ("normalization", {"version": 999}, "normalization"),
    ("retrieval", {"top_k": 4, "minimum_similarity": 0.25, "method": "unknown"}, "retrieval"),
    ("threshold", 2, "threshold"),
])
def test_incompatible_metadata_is_rejected(tmp_path, trained, field, value, message):
    model, _ = trained
    metadata = save(model, tmp_path / "model")
    metadata[field] = value
    (tmp_path / "model/metadata.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match=message):
        load_model(tmp_path / "model")


def test_changed_native_model_fails_fingerprint_check(tmp_path, trained):
    model, _ = trained
    save(model, tmp_path / "model")
    with (tmp_path / "model/model.txt").open("a") as stream:
        stream.write("changed\n")
    with pytest.raises(DataFormatError, match="fingerprint"):
        load_model(tmp_path / "model")


@pytest.mark.parametrize("labels", [[1] * 6, [0] * 6, [0, 1], [0, 1, 2, 0, 1, 2]])
def test_training_rejects_unlearnable_or_invalid_labels(trained, labels):
    _, features = trained
    with pytest.raises(DataFormatError, match="both classes"):
        fit_model(features, labels)
