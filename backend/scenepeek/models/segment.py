import uuid

from pgvector.sqlalchemy import Vector
from sqlalchemy import Computed, Float, ForeignKey, Index, Integer, String, Text
from sqlalchemy.dialects.postgresql import TSVECTOR, UUID
from sqlalchemy.orm import Mapped, mapped_column

from scenepeek.core.config import get_settings
from scenepeek.core.db import Base
from scenepeek.models._common import uuid_pk

_TEXT_DIM = get_settings().text_embed_dim


class Segment(Base):
    """Retrieval unit (~10 s). Only inserted once its chunk finishes, so search is incremental."""

    __tablename__ = "segments"
    __table_args__ = (
        Index("ix_segments_video_start", "video_id", "start_s"),
        Index("ix_segments_text_tsv", "text_tsv", postgresql_using="gin"),
        Index("ix_segments_ocr_tsv", "ocr_tsv", postgresql_using="gin"),
        Index(
            "ix_segments_ocr_trgm",
            "ocr_text",
            postgresql_using="gin",
            postgresql_ops={"ocr_text": "gin_trgm_ops"},
        ),
        Index(
            "ix_segments_text_embedding",
            "text_embedding",
            postgresql_using="hnsw",
            postgresql_ops={"text_embedding": "vector_cosine_ops"},
        ),
    )

    id: Mapped[uuid.UUID] = uuid_pk()
    video_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("videos.id", ondelete="CASCADE"), nullable=False
    )
    chunk_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("video_chunks.id", ondelete="CASCADE"), nullable=False, index=True
    )
    index: Mapped[int] = mapped_column(Integer, nullable=False)
    start_s: Mapped[float] = mapped_column(Float, nullable=False)
    end_s: Mapped[float] = mapped_column(Float, nullable=False)
    text: Mapped[str] = mapped_column(Text, server_default="", nullable=False)
    context_text: Mapped[str] = mapped_column(Text, server_default="", nullable=False)
    ocr_text: Mapped[str] = mapped_column(Text, server_default="", nullable=False)
    text_embedding = mapped_column(Vector(_TEXT_DIM))
    embedding_model: Mapped[str | None] = mapped_column(String(128))
    keyframe_key: Mapped[str | None] = mapped_column(String(1024))
    text_tsv = mapped_column(TSVECTOR, Computed("to_tsvector('english', text)", persisted=True))
    ocr_tsv = mapped_column(TSVECTOR, Computed("to_tsvector('simple', ocr_text)", persisted=True))
