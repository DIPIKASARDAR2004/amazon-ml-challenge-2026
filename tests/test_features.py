import numpy as np
import pandas as pd
import pytest

from src.data_loading import DataFormatError
from src.features import FEATURE_NAMES, make_features, validate_features


def test_hand_calculated_features_and_training_inference_consistency(records):
    left = records([("S1-a", "AB", "Same", "US")])
    right = records([("S2-a", "AC", " same ", "us")])
    features = make_features([("S1-a", "S2-a")], left, right)
    assert tuple(features.columns) == FEATURE_NAMES
    assert features.iloc[0].tolist() == [0.5, 0.5, 1, 0, 1, 0, 0, 0, 0, 1]
    pd.testing.assert_frame_equal(features, make_features([("S1-a", "S2-a")], left.copy(), right.copy()))


def test_two_blank_fields_never_count_as_positive_similarity(records):
    left = records([("S1-a", " ", "", "")])
    right = records([("S2-a", "", "   ", "")])
    feature = make_features([("S1-a", "S2-a")], left, right).iloc[0]
    assert feature.tolist() == [0, 0, 0, 0, 0, 1, 1, 1, 1, 0]
    assert np.isfinite(feature).all()


def test_na_is_not_a_blank_feature(records):
    left = records([("S1-a", "NA", "", "US")])
    right = records([("S2-a", "na", "", "US")])
    feature = make_features([("S1-a", "S2-a")], left, right).iloc[0]
    assert feature.name_exact == 1
    assert feature.query_name_blank == feature.target_name_blank == 0


def test_feature_order_missing_values_and_unknown_pairs_are_rejected(records):
    empty = make_features([], records([]), records([]))
    validate_features(empty)
    with pytest.raises(DataFormatError, match="order/schema"):
        validate_features(empty.loc[:, list(reversed(FEATURE_NAMES))])
    with pytest.raises(DataFormatError, match="finite"):
        validate_features(pd.DataFrame([[float("nan")] * len(FEATURE_NAMES)], columns=FEATURE_NAMES))
    with pytest.raises(DataFormatError, match="Unknown"):
        make_features([("S1-a", "S2-a")], records([]), records([]))
