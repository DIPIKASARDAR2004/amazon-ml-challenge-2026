"""Label retrieved training pairs; never manufacture random negatives."""

import numpy as np

from src.data_loading import DataFormatError
from src.evaluation_io import target_set, validate_id


def label_pairs(pairs, truth):
    known = {}
    for entity_id, values in truth.items():
        validate_id(entity_id)
        known[entity_id] = target_set(values, context=entity_id)
    if len(pairs) != len(set(pairs)):
        raise DataFormatError("Duplicate training pairs")
    labels = []
    for query_id, target_id in pairs:
        validate_id(query_id)
        validate_id(target_id, ("S2-", "S3-"))
        if query_id not in known:
            raise DataFormatError(f"Missing training truth for {query_id}")
        labels.append(int(target_id in known[query_id]))
    return np.asarray(labels, dtype=np.int8)
