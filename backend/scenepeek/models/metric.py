from datetime import datetime

from sqlalchemy import BigInteger, DateTime, Float, Index, String, func
from sqlalchemy.orm import Mapped, mapped_column

from scenepeek.core.db import Base


class MetricSample(Base):
    """Append-only timing samples (model inference, job duration, search latency) so the in-app
    System page can compute exact percentiles without a Prometheus dependency."""

    __tablename__ = "metric_samples"
    __table_args__ = (Index("ix_metric_samples_name_ts", "name", "ts"),)

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(64), nullable=False)
    value: Mapped[float] = mapped_column(Float, nullable=False)
    source: Mapped[str | None] = mapped_column(String(128))
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
