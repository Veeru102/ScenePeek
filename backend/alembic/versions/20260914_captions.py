"""keyframe captions

Revision ID: 0003
Revises: 0002
Create Date: 2026-09-14
"""
from typing import Sequence, Union

from alembic import op
from pgvector.sqlalchemy import Vector
import sqlalchemy as sa
from scenepeek.core.config import get_settings

revision: str = "0003"
down_revision: Union[str, None] = "0002"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _has_column(table: str, column: str) -> bool:
    cols = {c["name"] for c in sa.inspect(op.get_bind()).get_columns(table)}
    return column in cols


def upgrade() -> None:
    # 0001 creates tables from the live models, so a fresh DB already has these columns.
    dim = get_settings().text_embed_dim
    if not _has_column("segments", "caption_text"):
        op.add_column("segments", sa.Column("caption_text", sa.Text(), server_default="", nullable=False))
    if not _has_column("segments", "caption_embedding"):
        op.add_column("segments", sa.Column("caption_embedding", Vector(dim)))
        op.create_index(
            "ix_segments_caption_embedding",
            "segments",
            ["caption_embedding"],
            postgresql_using="hnsw",
            postgresql_ops={"caption_embedding": "vector_cosine_ops"},
        )
    if not _has_column("frames", "caption"):
        op.add_column("frames", sa.Column("caption", sa.Text(), server_default="", nullable=False))
    if not _has_column("video_chunks", "captioned_at"):
        op.add_column("video_chunks", sa.Column("captioned_at", sa.DateTime(timezone=True)))


def downgrade() -> None:
    op.drop_index("ix_segments_caption_embedding", table_name="segments")
    op.drop_column("segments", "caption_embedding")
    op.drop_column("segments", "caption_text")
    op.drop_column("frames", "caption")
    op.drop_column("video_chunks", "captioned_at")
