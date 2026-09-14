"""hard-negative training examples

Revision ID: 0006
Revises: 0005
Create Date: 2026-09-17
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from scenepeek.core.db import Base
import scenepeek.models  # noqa: F401

revision: str = "0006"
down_revision: Union[str, None] = "0005"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    if "training_examples" not in sa.inspect(op.get_bind()).get_table_names():
        Base.metadata.create_all(op.get_bind(), tables=[Base.metadata.tables["training_examples"]])


def downgrade() -> None:
    op.drop_table("training_examples")
