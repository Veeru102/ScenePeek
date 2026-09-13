"""Sentence embeddings for transcript segments and queries (bge-small by default)."""

import numpy as np

from scenepeek.core.config import get_settings
from scenepeek.ml.registry import device, singleton, timed

# bge models recommend a query instruction for short queries vs. passages
_QUERY_PREFIX = {
    "BAAI/bge-small-en-v1.5": "Represent this sentence for searching relevant passages: ",
    "BAAI/bge-base-en-v1.5": "Represent this sentence for searching relevant passages: ",
}


def _load():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(get_settings().text_embed_model, device=device())


def model():
    return singleton("text_embed", _load)


def embed_passages(texts: list[str], batch_size: int = 64) -> np.ndarray:
    if not texts:
        return np.zeros((0, get_settings().text_embed_dim), dtype=np.float32)
    with timed("text_embed"):
        return model().encode(texts, batch_size=batch_size, normalize_embeddings=True, convert_to_numpy=True)


def embed_query(text: str) -> np.ndarray:
    prefix = _QUERY_PREFIX.get(get_settings().text_embed_model, "")
    with timed("text_embed_query"):
        return model().encode([prefix + text], normalize_embeddings=True, convert_to_numpy=True)[0]
