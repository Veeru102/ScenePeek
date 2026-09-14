"""Model keys: the single place that names which model produced a stored artifact.

Every embedding / transcript / OCR row is tagged with one of these so experiments record what they
ran against and a new model version can be backfilled next to the old one instead of over it."""

from importlib.metadata import PackageNotFoundError, version

from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from scenepeek.core.config import get_settings
from scenepeek.models import IndexVersion

KINDS = ("asr", "text", "frame", "ocr", "caption", "reranker", "temporal", "router")


def _pkg(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "?"


def short(model_name: str) -> str:
    """'BAAI/bge-small-en-v1.5' -> 'bge-small-en-v1.5'; local checkpoint paths keep their basename."""
    return model_name.rstrip("/").split("/")[-1]


def model_keys() -> dict[str, str]:
    s = get_settings()
    return {
        "asr": f"whisper-{s.whisper_model}",
        "text": short(s.text_embed_model),
        "frame": short(s.visual_embed_model),
        "ocr": f"rapidocr-{_pkg('rapidocr-onnxruntime')}",
        "caption": short(s.caption_model),
        "reranker": short(s.reranker_model),
    }


def dims() -> dict[str, int]:
    s = get_settings()
    return {"text": s.text_embed_dim, "frame": s.visual_embed_dim, "caption": s.text_embed_dim}


def register(s: Session, keys: dict[str, str] | None = None) -> None:
    """Upsert the active models into the registry (idempotent; called by writers and experiments)."""
    keys = keys or model_keys()
    d = dims()
    for kind, key in keys.items():
        stmt = insert(IndexVersion).values(kind=kind, model_key=key, dim=d.get(kind))
        s.execute(stmt.on_conflict_do_nothing(constraint="uq_index_version"))
