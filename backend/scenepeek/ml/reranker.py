"""Cross-encoder reranking for top-K candidates."""

import numpy as np

from scenepeek.core.config import get_settings
from scenepeek.ml.registry import device, singleton, timed


def _load():
    from sentence_transformers import CrossEncoder

    return CrossEncoder(get_settings().reranker_model, device=device(), max_length=512)


def rerank_scores(query: str, passages: list[str]) -> np.ndarray:
    """Returns sigmoid-normalised relevance in [0,1] per passage."""
    if not passages:
        return np.zeros(0, dtype=np.float32)
    model = singleton("reranker", _load)
    with timed("reranker"):
        logits = model.predict([(query, p) for p in passages], batch_size=16, show_progress_bar=False)
    logits = np.asarray(logits, dtype=np.float32)
    return 1.0 / (1.0 + np.exp(-logits))
