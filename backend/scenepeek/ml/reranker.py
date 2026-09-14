"""Cross-encoder reranking for top-K candidates."""

import numpy as np

from scenepeek.core.config import get_settings
from scenepeek.ml.registry import device, singleton, timed


def _load(name: str):
    def factory():
        from sentence_transformers import CrossEncoder

        return CrossEncoder(name, device=device(), max_length=512)

    return factory


def rerank_scores(query: str, passages: list[str], model_name: str | None = None) -> np.ndarray:
    """Returns sigmoid-normalised relevance in [0,1] per passage. `model_name` may be an HF id or a
    local fine-tuned checkpoint directory; each is cached separately so two can be compared."""
    if not passages:
        return np.zeros(0, dtype=np.float32)
    name = model_name or get_settings().reranker_model
    model = singleton(f"reranker:{name}", _load(name))
    import torch

    with timed("reranker"):
        logits = model.predict(
            [(query, p) for p in passages],
            batch_size=16,
            show_progress_bar=False,
            activation_fn=torch.nn.Identity(),  # raw logits; we apply the sigmoid ourselves
        )
    logits = np.asarray(logits, dtype=np.float32).reshape(-1)
    return 1.0 / (1.0 + np.exp(-logits))
