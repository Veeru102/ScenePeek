"""versioned embeddings table + chunk.temporal_at

Revision ID: 0005
Revises: 0004
Create Date: 2026-09-16
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from scenepeek.core.db import Base
import scenepeek.models  # noqa: F401

revision: str = "0005"
down_revision: Union[str, None] = "0004"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    insp = sa.inspect(op.get_bind())
    if "embeddings" not in insp.get_table_names():
        Base.metadata.create_all(op.get_bind(), tables=[Base.metadata.tables["embeddings"]])
    if "temporal_at" not in {c["name"] for c in insp.get_columns("video_chunks")}:
        op.add_column("video_chunks", sa.Column("temporal_at", sa.DateTime(timezone=True)))


def downgrade() -> None:
    op.drop_column("video_chunks", "temporal_at")
    op.drop_table("embeddings")
