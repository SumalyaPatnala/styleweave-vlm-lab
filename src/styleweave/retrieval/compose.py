"""Query composition. B0 is Phase 0 L2-sum, not CLIP4Cir's unnormalized add."""

from __future__ import annotations

import numpy as np

EPS = 1e-12


def l2_normalize(vectors: np.ndarray, axis: int = -1) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=axis, keepdims=True)
    return vectors / np.maximum(norms, EPS)


def compose_queries(
    image_features: np.ndarray,
    text_features: np.ndarray,
    mode: str = "l2_sum",
) -> np.ndarray:
    if image_features.shape != text_features.shape:
        raise ValueError("image and text features must have the same shape")
    if mode == "l2_sum":
        return l2_normalize(l2_normalize(image_features) + l2_normalize(text_features))
    if mode == "image_only":
        return l2_normalize(image_features)
    if mode == "text_only":
        return l2_normalize(text_features)
    raise ValueError(f"Unknown query mode: {mode}")


def combine_relative_captions(captions: list[str], mode: str = "clip4cir_concat") -> str:
    cleaned = [c.strip() for c in captions if c and c.strip()]
    if not cleaned:
        raise ValueError("query has no captions")
    if mode != "clip4cir_concat":
        raise ValueError(f"Unknown caption mode: {mode}")
    if len(cleaned) == 1:
        return cleaned[0].strip(".?, ").capitalize()
    first = cleaned[0].strip(".?, ").capitalize()
    rest = [item.strip(".?, ") for item in cleaned[1:]]
    return first + " and " + " and ".join(rest)
