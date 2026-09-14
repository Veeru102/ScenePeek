import uuid
from datetime import datetime

from sqlalchemy import Float, ForeignKey, Index, String, Text, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from scenepeek.core.db import Base
from scenepeek.models._common import created_at_col, uuid_pk


class DatasetVideoStatus:
    PENDING = "pending"
    FETCHED = "fetched"
    UNAVAILABLE = "unavailable"
    INDEXED = "indexed"


class Dataset(Base):
    """A benchmark (QVHighlights, the hand-written ScenePeek set, ...) registered in the library."""

    __tablename__ = "datasets"

    id: Mapped[uuid.UUID] = uuid_pk()
    name: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    version: Mapped[str] = mapped_column(String(64), server_default="", nullable=False)
    description: Mapped[str] = mapped_column(Text, server_default="", nullable=False)
    created_at: Mapped[datetime] = created_at_col()


class DatasetVideo(Base):
    """One benchmark clip. `video_id` is set once the clip has been fetched into the library and
    goes through the normal probe/extract/index pipeline like any upload."""

    __tablename__ = "dataset_videos"
    __table_args__ = (UniqueConstraint("dataset_id", "external_id", name="uq_dataset_video"),)

    id: Mapped[uuid.UUID] = uuid_pk()
    dataset_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("datasets.id", ondelete="CASCADE"), nullable=False, index=True
    )
    external_id: Mapped[str] = mapped_column(String(256), nullable=False)
    split: Mapped[str] = mapped_column(String(16), nullable=False)
    duration_s: Mapped[float | None] = mapped_column(Float)
    source_url: Mapped[str | None] = mapped_column(String(1024))
    status: Mapped[str] = mapped_column(String(16), server_default=DatasetVideoStatus.PENDING, nullable=False)
    video_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("videos.id", ondelete="SET NULL"), index=True
    )
    meta: Mapped[dict] = mapped_column(JSONB, server_default="{}", nullable=False)
    created_at: Mapped[datetime] = created_at_col()


class DatasetQuery(Base):
    """A natural-language query with its ground-truth windows ([[start_s, end_s], ...]) inside one
    dataset video."""

    __tablename__ = "dataset_queries"
    __table_args__ = (
        UniqueConstraint("dataset_id", "external_id", name="uq_dataset_query"),
        Index("ix_dataset_queries_split", "dataset_id", "split"),
    )

    id: Mapped[uuid.UUID] = uuid_pk()
    dataset_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("datasets.id", ondelete="CASCADE"), nullable=False
    )
    dataset_video_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("dataset_videos.id", ondelete="CASCADE"), nullable=False, index=True
    )
    external_id: Mapped[str] = mapped_column(String(256), nullable=False)
    split: Mapped[str] = mapped_column(String(16), nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    modality: Mapped[str | None] = mapped_column(String(16))
    relevant: Mapped[list] = mapped_column(JSONB, nullable=False)
    meta: Mapped[dict] = mapped_column(JSONB, server_default="{}", nullable=False)
