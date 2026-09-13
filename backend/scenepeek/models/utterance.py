import uuid

from sqlalchemy import Float, ForeignKey, Index, Text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from scenepeek.core.db import Base
from scenepeek.models._common import uuid_pk


class Utterance(Base):
    """Transcript line with word timings, used for the transcript panel."""

    __tablename__ = "utterances"
    __table_args__ = (Index("ix_utterances_video_start", "video_id", "start_s"),)

    id: Mapped[uuid.UUID] = uuid_pk()
    video_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("videos.id", ondelete="CASCADE"), nullable=False
    )
    chunk_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("video_chunks.id", ondelete="CASCADE"), nullable=False, index=True
    )
    start_s: Mapped[float] = mapped_column(Float, nullable=False)
    end_s: Mapped[float] = mapped_column(Float, nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    words: Mapped[list | None] = mapped_column(JSONB)
