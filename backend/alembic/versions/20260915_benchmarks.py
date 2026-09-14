"""benchmark datasets, experiments, model version registry, priority bands

Revision ID: 0004
Revises: 0003
Create Date: 2026-09-15
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from scenepeek.core.db import Base
import scenepeek.models  # noqa: F401

revision: str = "0004"
down_revision: Union[str, None] = "0003"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

NEW_TABLES = ("datasets", "dataset_videos", "dataset_queries", "experiments", "experiment_results", "index_versions")
NEW_COLUMNS = (
    ("videos", sa.Column("source", sa.String(32), server_default="upload", nullable=False)),
    ("videos", sa.Column("priority_band", sa.Integer(), server_default="0", nullable=False)),
    ("frames", sa.Column("visual_model", sa.String(128))),
    ("segments", sa.Column("ocr_model", sa.String(128))),
    ("video_chunks", sa.Column("asr_model", sa.String(128))),
)


def upgrade() -> None:
    # 0001 creates tables from the live models, so a fresh DB already has all of this.
    insp = sa.inspect(op.get_bind())
    existing = set(insp.get_table_names())
    missing = [Base.metadata.tables[t] for t in NEW_TABLES if t not in existing]
    if missing:
        Base.metadata.create_all(op.get_bind(), tables=missing)
    for table, col in NEW_COLUMNS:
        if col.name not in {c["name"] for c in insp.get_columns(table)}:
            op.add_column(table, col)


def downgrade() -> None:
    for table, col in NEW_COLUMNS:
        op.drop_column(table, col.name)
    for t in reversed(NEW_TABLES):
        op.drop_table(t)
