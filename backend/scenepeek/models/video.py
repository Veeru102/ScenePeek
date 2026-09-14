import uuid
from datetime import datetime

from sqlalchemy import BigInteger, DateTime, Float, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from scenepeek.core.db import Base
from scenepeek.models._common import created_at_col, uuid_pk


class VideoStatus:
    UPLOADING = "uploading"
    UPLOADED = "uploaded"
    PROCESSING = "processing"
    READY = "ready"
    FAILED = "failed"


class Video(Base):
    __tablename__ = "videos"

    id: Mapped[uuid.UUID] = uuid_pk()
    title: Mapped[str] = mapped_column(String(512), nullable=False)
    original_key: Mapped[str] = mapped_column(String(1024), nullable=False)
    web_key: Mapped[str | None] = mapped_column(String(1024))
    poster_key: Mapped[str | None] = mapped_column(String(1024))
    content_type: Mapped[str | None] = mapped_column(String(128))
    size_bytes: Mapped[int | None] = mapped_column(BigInteger)
    duration_s: Mapped[float | None] = mapped_column(Float)
    width: Mapped[int | None] = mapped_column(Integer)
    height: Mapped[int | None] = mapped_column(Integer)
    fps: Mapped[float | None] = mapped_column(Float)
    codec: Mapped[str | None] = mapped_column(String(64))
    status: Mapped[str] = mapped_column(String(32), server_default=VideoStatus.UPLOADING, nullable=False)
    error: Mapped[str | None] = mapped_column(Text)
    chunk_count: Mapped[int] = mapped_column(Integer, server_default="0", nullable=False)
    source: Mapped[str] = mapped_column(String(32), server_default="upload", nullable=False)
    priority_band: Mapped[int] = mapped_column(Integer, server_default="0", nullable=False)

    created_at: Mapped[datetime] = created_at_col()
    upload_completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    processing_started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    first_searchable_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    chunks: Mapped[list["VideoChunk"]] = relationship(  # noqa: F821
        back_populates="video", cascade="all, delete-orphan", order_by="VideoChunk.index"
    )
