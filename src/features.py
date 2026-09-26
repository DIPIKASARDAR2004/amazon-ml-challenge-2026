"""One feature function for both training and inference."""

import numpy as np
import pandas as pd
from rapidfuzz import fuzz

from src.data_loading import DataFormatError


FEATURE_NAMES = ("name_ratio", "name_token_sort_ratio", "address_ratio", "name_exact",
                 "address_exact", "query_name_blank", "target_name_blank",
                 "query_address_blank", "target_address_blank", "country_equal")


def make_features(pairs, queries, targets):
    if len(pairs) != len(set(pairs)):
        raise DataFormatError("Duplicate feature pairs")
    rows = []
    for query_id, target_id in pairs:
        if query_id not in queries.index or target_id not in targets.index:
            raise DataFormatError("Unknown ID in feature pair")
        left, right = queries.loc[query_id], targets.loc[target_id]
        name, other_name = left["business_name_normalized"], right["business_name_normalized"]
        address, other_address = left["business_address_normalized"], right["business_address_normalized"]
        country, other_country = left["country_normalized"], right["country_normalized"]
        rows.append([
            fuzz.ratio(name, other_name) / 100 if name and other_name else 0,
            fuzz.token_sort_ratio(name, other_name) / 100 if name and other_name else 0,
            fuzz.ratio(address, other_address) / 100 if address and other_address else 0,
            bool(name and name == other_name), bool(address and address == other_address),
            not name, not other_name, not address, not other_address,
            bool(country and country == other_country),
        ])
    return pd.DataFrame(rows, columns=FEATURE_NAMES, dtype=np.float64)


def validate_features(features):
    if tuple(features.columns) != FEATURE_NAMES:
        raise DataFormatError("Incompatible feature order/schema")
    if not np.isfinite(features.to_numpy(dtype=np.float64)).all():
        raise DataFormatError("Features must be finite")
