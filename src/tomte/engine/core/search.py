"""Mechanical ranking options; no interpretation of memory or query meaning."""
from typing import Literal

StoredRetrievalMode = Literal['vector', 'bm25', 'hybrid']


def reciprocal_rank_fusion(*rankings):
    """Fuse unique identifiers with equal weights and deterministic ties."""
    scores = {}
    for ranking in rankings:
        for rank, key in enumerate(dict.fromkeys(ranking), 1):
            scores[key] = scores.get(key, 0.0) + 1.0 / (60 + rank)
    return sorted(scores, key=lambda key: (-scores[key], key)), scores
