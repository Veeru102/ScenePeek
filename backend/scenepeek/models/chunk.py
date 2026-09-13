import uuid
from datetime import datetime

from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from scenepeek.core.db import Base
from scenepeek.models._common import created_at_col, uuid_pk


class ChunkStatus:
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


class VideoChunk(Base):
    """Independent processing unit (~60 s of video)."""

    __tablename__ = "video_chunks"
    __table_args__ = (UniqueConstraint("video_id", "index", name="uq_chunk_video_index"),)

    id: Mapped[uuid.UUID] = uuid_pk()
    video_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("videos.id", ondelete="CASCADE"), nullable=False, index=True
    )
    index: Mapped[int] = mapped_column(Integer, nullable=False)
    start_s: Mapped[float] = mapped_column(Float, nullable=False)
    end_s: Mapped[float] = mapped_column(Float, nullable=False)
    status: Mapped[str] = mapped_column(String(16), default=ChunkStatus.PENDING, nullable=False)
    stage: Mapped[str | None] = mapped_column(String(32))
    attempts: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    error: Mapped[str | None] = mapped_column(Text)

    extracted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    transcribed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    embedded_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    ocr_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    done_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = created_at_col()

    video: Mapped["Video"] = relationship(back_populates="chunks")  # noqa: F821
