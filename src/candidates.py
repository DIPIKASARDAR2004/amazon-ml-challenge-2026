"""Small in-memory TF-IDF retrieval. Not an index for millions of targets."""

import math

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from src.data_loading import DataFormatError
from src.evaluation_io import require_coverage, target_set, validate_id


def retrieval_settings(top_k=4, minimum_similarity=0.25):
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
        raise ValueError("top_k must be a positive integer")
    if not math.isfinite(minimum_similarity) or not 0 < minimum_similarity <= 1:
        raise ValueError("minimum_similarity must be in (0, 1]")
    return {"version": 1, "method": "max_name_address_char_tfidf", "analyzer": "char_wb",
            "ngram_range": [2, 4], "top_k": top_k, "minimum_similarity": minimum_similarity,
            "tie_break": "target_id_ascending", "country_filter": False}


class CandidateIndex:
    """Fits text vocabulary on targets only; has no ground-truth interface.

    Similarity is the maximum of name/address cosine similarity. Query one row
    at a time to avoid a query-by-target dense matrix, but this still scans all
    targets per query. Pass 4 must replace/benchmark this at realistic scale.
    """

    def __init__(self, targets, *, top_k=4, minimum_similarity=0.25):
        self.settings = retrieval_settings(top_k, minimum_similarity)
        ordered = targets.sort_index()
        self.ids = ordered.index.tolist()
        self.fields = []
        for field in ("business_name_normalized", "business_address_normalized"):
            if not any(ordered[field]):
                continue
            vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), lowercase=False)
            matrix = vectorizer.fit_transform(ordered[field])
            self.fields.append((field, vectorizer, matrix))

    def retrieve(self, queries):
        result = {}
        for entity_id, row in queries.sort_index().iterrows():
            scores = np.zeros(len(self.ids))
            for field, vectorizer, matrix in self.fields:
                query = vectorizer.transform([row[field]])
                similarity = (matrix @ query.T).toarray().ravel()
                scores = np.maximum(scores, np.clip(similarity, 0, 1))
            eligible = [i for i, score in enumerate(scores)
                        if score >= self.settings["minimum_similarity"]]
            ranked = sorted(eligible, key=lambda i: (-scores[i], self.ids[i]))
            result[entity_id] = [self.ids[i] for i in ranked[:self.settings["top_k"]]]
        return result


def candidate_pairs(candidates, query_ids, target_ids):
    """Exactly the unique final retrieved pairs, in stable order."""
    require_coverage(candidates, query_ids, context="candidates")
    allowed = set(target_ids)
    pairs = []
    for entity_id in sorted(candidates):
        validate_id(entity_id)
        values = target_set(candidates[entity_id], context=entity_id)
        if values - allowed:
            raise DataFormatError(f"{entity_id}: unknown candidate targets")
        pairs.extend((entity_id, target) for target in sorted(values))
    return pairs
