import uuid
from datetime import datetime

from sqlalchemy import DateTime, Float, ForeignKey, Integer, String
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from scenepeek.core.db import Base
from scenepeek.models._common import created_at_col, uuid_pk


class ExperimentStatus:
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


class Experiment(Base):
    """One evaluation run: the full search configuration + model versions it ran with, and the
    aggregate metrics. Per-query rows live in `experiment_results` for paired comparisons."""

    __tablename__ = "experiments"

    id: Mapped[uuid.UUID] = uuid_pk()
    name: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    dataset_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("datasets.id", ondelete="SET NULL")
    )
    split: Mapped[str | None] = mapped_column(String(16))
    subset: Mapped[dict] = mapped_column(JSONB, server_default="{}", nullable=False)
    config: Mapped[dict] = mapped_column(JSONB, server_default="{}", nullable=False)
    code_sha: Mapped[str | None] = mapped_column(String(64))
    model_versions: Mapped[dict] = mapped_column(JSONB, server_default="{}", nullable=False)
    metrics: Mapped[dict] = mapped_column(JSONB, server_default="{}", nullable=False)
    by_group: Mapped[dict] = mapped_column(JSONB, server_default="{}", nullable=False)
    n_queries: Mapped[int] = mapped_column(Integer, server_default="0", nullable=False)
    status: Mapped[str] = mapped_column(String(16), server_default=ExperimentStatus.RUNNING, nullable=False)
    started_at: Mapped[datetime] = created_at_col()
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class ExperimentResult(Base):
    __tablename__ = "experiment_results"

    id: Mapped[uuid.UUID] = uuid_pk()
    experiment_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("experiments.id", ondelete="CASCADE"), nullable=False, index=True
    )
    dataset_query_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("dataset_queries.id", ondelete="CASCADE")
    )
    query_key: Mapped[str] = mapped_column(String(256), nullable=False)
    metrics: Mapped[dict] = mapped_column(JSONB, server_default="{}", nullable=False)
    lanes: Mapped[dict] = mapped_column(JSONB, server_default="{}", nullable=False)
    hits: Mapped[list] = mapped_column(JSONB, server_default="[]", nullable=False)
    latency_ms: Mapped[float | None] = mapped_column(Float)
