"""add target_column and task_type to jobs

Set when the user confirms the detected schema (Section 3). Nullable because a
job has neither until confirmation; the full confirmed schema lives in a JSON
artifact, and these mirror the two fields every later stage reads constantly.

Revision ID: c3f8a1d2e4b5
Revises: b7c1e2f4a9d0
Create Date: 2026-07-24 11:00:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "c3f8a1d2e4b5"
down_revision: str | None = "b7c1e2f4a9d0"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("jobs", sa.Column("target_column", sa.String(length=255), nullable=True))
    op.add_column("jobs", sa.Column("task_type", sa.String(length=32), nullable=True))


def downgrade() -> None:
    op.drop_column("jobs", "task_type")
    op.drop_column("jobs", "target_column")
