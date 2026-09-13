import uuid

from pgvector.sqlalchemy import Vector
from sqlalchemy import Float, ForeignKey, Index, String, Text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from scenepeek.core.config import get_settings
from scenepeek.core.db import Base
from scenepeek.models._common import uuid_pk

_VISUAL_DIM = get_settings().visual_embed_dim


class Frame(Base):
    """Representative keyframe with a vision-language embedding and OCR text."""

    __tablename__ = "frames"
    __table_args__ = (
        Index("ix_frames_video_t", "video_id", "t_s"),
        Index(
            "ix_frames_visual_embedding",
            "visual_embedding",
            postgresql_using="hnsw",
            postgresql_ops={"visual_embedding": "vector_cosine_ops"},
        ),
    )

    id: Mapped[uuid.UUID] = uuid_pk()
    video_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("videos.id", ondelete="CASCADE"), nullable=False
    )
    segment_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("segments.id", ondelete="CASCADE"), nullable=False, index=True
    )
    t_s: Mapped[float] = mapped_column(Float, nullable=False)
    image_key: Mapped[str] = mapped_column(String(1024), nullable=False)
    visual_embedding = mapped_column(Vector(_VISUAL_DIM))
    ocr_text: Mapped[str] = mapped_column(Text, server_default="", nullable=False)
    ocr_boxes: Mapped[list | None] = mapped_column(JSONB)
    phash: Mapped[str | None] = mapped_column(String(32))
    tags: Mapped[list | None] = mapped_column(JSONB)
