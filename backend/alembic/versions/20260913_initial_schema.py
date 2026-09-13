"""initial schema

Revision ID: 0001
Revises:
Create Date: 2026-09-13
"""
from typing import Sequence, Union

from alembic import op
from scenepeek.core.config import get_settings
from scenepeek.core.db import Base
import scenepeek.models  # noqa: F401

revision: str = "0001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")
    # Embedding dimensions come from settings so model swaps are a config change + fresh DB.
    s = get_settings()
    Base.metadata.create_all(
        op.get_bind(),
        tables=[Base.metadata.tables[t] for t in
                ("videos", "video_chunks", "utterances", "segments", "frames", "topics", "jobs")],
    )
    op.execute(
        f"COMMENT ON TABLE segments IS 'text_embedding dim={s.text_embed_dim} model={s.text_embed_model}'"
    )
    op.execute(
        f"COMMENT ON TABLE frames IS 'visual_embedding dim={s.visual_embed_dim} model={s.visual_embed_model}'"
    )


def downgrade() -> None:
    for t in ("jobs", "topics", "frames", "segments", "utterances", "video_chunks", "videos"):
        op.drop_table(t)
