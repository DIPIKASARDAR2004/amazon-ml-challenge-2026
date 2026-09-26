"""Small LightGBM model with explicit feature/preprocessing metadata, no pickle."""

import json
import math
from importlib.metadata import version
from pathlib import Path

import numpy as np

from src.candidates import retrieval_settings
from src.data_loading import DataFormatError
from src.evaluation_io import fingerprint_file
from src.features import FEATURE_NAMES, validate_features
from src.normalization import NORMALIZATION


RELOAD_TOLERANCE = 1e-12
TRAINING_ROUNDS = 40


def validate_threshold(threshold):
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("threshold must be finite and in [0, 1]")


def training_settings(seed):
    return {"objective": "binary", "verbosity": -1, "num_threads": 1,
            "deterministic": True, "force_col_wise": True, "seed": seed,
            "learning_rate": 0.1, "num_leaves": 7, "max_depth": 3,
            "min_data_in_leaf": 1, "min_data_in_bin": 1}


def fit_model(features, labels, *, seed=42):
    import lightgbm as lgb

    validate_features(features)
    labels = np.asarray(labels)
    if labels.ndim != 1 or len(labels) != len(features) or set(labels.tolist()) != {0, 1}:
        raise DataFormatError("Training requires aligned binary labels with both classes")
    data = lgb.Dataset(features, label=labels, feature_name=list(FEATURE_NAMES))
    return lgb.train(training_settings(seed), data, num_boost_round=TRAINING_ROUNDS)


def predict_scores(model, features):
    validate_features(features)
    if model.feature_name() != list(FEATURE_NAMES):
        raise DataFormatError("Model feature names do not match the feature schema")
    if features.empty:
        return np.empty(0, dtype=np.float64)
    scores = np.asarray(model.predict(features, num_threads=1), dtype=np.float64)
    if scores.shape != (len(features),) or not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise DataFormatError("Model returned invalid binary scores")
    return scores


def save_model(model, directory, *, retrieval, threshold, seed):
    validate_threshold(threshold)
    if retrieval != retrieval_settings(retrieval["top_k"], retrieval["minimum_similarity"]):
        raise DataFormatError("Unsupported retrieval settings")
    if model.feature_name() != list(FEATURE_NAMES):
        raise DataFormatError("Model feature names do not match the feature schema")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    model_path = directory / "model.txt"
    model.save_model(str(model_path))
    metadata = {
        "schema_version": 1, "features": list(FEATURE_NAMES), "normalization": NORMALIZATION,
        "retrieval": retrieval, "threshold": threshold, "seed": seed,
        "training_parameters": training_settings(seed), "training_rounds": TRAINING_ROUNDS,
        "reload_absolute_tolerance": RELOAD_TOLERANCE,
        "versions": {name: version(name) for name in ("lightgbm", "numpy", "pandas", "scikit-learn", "RapidFuzz")},
        "model_file": fingerprint_file(model_path),
    }
    (directory / "metadata.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return metadata


def load_model(directory):
    import lightgbm as lgb

    directory = Path(directory)
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    if metadata.get("schema_version") != 1 or metadata.get("features") != list(FEATURE_NAMES):
        raise DataFormatError("Incompatible model feature schema")
    if metadata.get("normalization") != NORMALIZATION:
        raise DataFormatError("Incompatible model normalization settings")
    retrieval = metadata.get("retrieval", {})
    if retrieval != retrieval_settings(retrieval.get("top_k"), retrieval.get("minimum_similarity", float("nan"))):
        raise DataFormatError("Incompatible model retrieval settings")
    validate_threshold(metadata.get("threshold", float("nan")))
    model_path = directory / "model.txt"
    if metadata.get("model_file") != fingerprint_file(model_path):
        raise DataFormatError("Model file fingerprint mismatch")
    model = lgb.Booster(model_file=str(model_path))
    if model.feature_name() != list(FEATURE_NAMES) or model.params.get("objective") != "binary":
        raise DataFormatError("Incompatible saved model")
    return model, metadata
