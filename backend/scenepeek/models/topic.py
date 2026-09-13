import uuid

from pgvector.sqlalchemy import Vector
from sqlalchemy import Float, ForeignKey, Integer, String
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from scenepeek.core.config import get_settings
from scenepeek.core.db import Base
from scenepeek.models._common import uuid_pk

_TEXT_DIM = get_settings().text_embed_dim


class Topic(Base):
    """Semantic timeline entry: a span of the video about one topic."""

    __tablename__ = "topics"

    id: Mapped[uuid.UUID] = uuid_pk()
    video_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("videos.id", ondelete="CASCADE"), nullable=False, index=True
    )
    index: Mapped[int] = mapped_column(Integer, nullable=False)
    start_s: Mapped[float] = mapped_column(Float, nullable=False)
    end_s: Mapped[float] = mapped_column(Float, nullable=False)
    label: Mapped[str] = mapped_column(String(256), nullable=False)
    keyphrases: Mapped[list | None] = mapped_column(JSONB)
    embedding = mapped_column(Vector(_TEXT_DIM))
    source: Mapped[str] = mapped_column(String(16), server_default="extractive", nullable=False)
