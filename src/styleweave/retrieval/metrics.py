"""Recall@K for a single labeled target per query."""

from __future__ import annotations

import numpy as np

from styleweave.retrieval.compose import l2_normalize


def ranks_from_scores(
    query_features: np.ndarray,
    gallery_features: np.ndarray,
    target_indices: np.ndarray,
) -> np.ndarray:
    """Return 0-based rank of each target. Cosine on L2-normalized vectors."""
    queries = l2_normalize(np.asarray(query_features, dtype=np.float32))
    gallery = l2_normalize(np.asarray(gallery_features, dtype=np.float32))
    scores = queries @ gallery.T
    order = np.argsort(-scores, axis=1)
    matches = order == np.asarray(target_indices)[:, None]
    ranks = np.argmax(matches, axis=1)
    if not np.all(matches.any(axis=1)):
        raise ValueError("a target index was not present in the gallery")
    return ranks.astype(np.int64)


def recall_at_k(ranks: np.ndarray, k: int) -> float:
    if k < 1:
        raise ValueError("k must be >= 1")
    return float(np.mean(np.asarray(ranks) < k))
