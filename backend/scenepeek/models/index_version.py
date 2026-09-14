import uuid
from datetime import datetime

from sqlalchemy import Integer, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from scenepeek.core.db import Base
from scenepeek.models._common import created_at_col, uuid_pk


class IndexVersionStatus:
    BUILDING = "building"
    READY = "ready"
    RETIRED = "retired"


class IndexVersion(Base):
    """Registry of every model that has produced stored artifacts (embeddings, transcripts, OCR,
    reranker checkpoints), so experiments can record exactly which versions they ran against and
    backfills of a new version can coexist with the old one until it is retired."""

    __tablename__ = "index_versions"
    __table_args__ = (UniqueConstraint("kind", "model_key", name="uq_index_version"),)

    id: Mapped[uuid.UUID] = uuid_pk()
    kind: Mapped[str] = mapped_column(String(16), nullable=False)
    model_key: Mapped[str] = mapped_column(String(128), nullable=False)
    dim: Mapped[int | None] = mapped_column(Integer)
    status: Mapped[str] = mapped_column(String(16), server_default=IndexVersionStatus.READY, nullable=False)
    index_name: Mapped[str | None] = mapped_column(String(128))
    created_at: Mapped[datetime] = created_at_col()
