"""create agent_runs table

One row per pipeline node per job, tracking status and timing for the Progress
page (build-plan Section 4).

Revision ID: e5a9c7d1f30b
Revises: c3f8a1d2e4b5
Create Date: 2026-07-24 12:00:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "e5a9c7d1f30b"
down_revision: str | None = "c3f8a1d2e4b5"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "agent_runs",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("job_id", sa.Integer(), nullable=False),
        sa.Column("name", sa.String(length=64), nullable=False),
        sa.Column("sequence", sa.Integer(), nullable=False),
        sa.Column(
            "status",
            sa.Enum(
                "pending",
                "running",
                "completed",
                "failed",
                "skipped",
                name="agent_run_status",
            ),
            nullable=False,
        ),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["job_id"], ["jobs.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("job_id", "name", name="uq_agent_run_job_name"),
    )
    op.create_index(op.f("ix_agent_runs_job_id"), "agent_runs", ["job_id"], unique=False)


def downgrade() -> None:
    op.drop_index(op.f("ix_agent_runs_job_id"), table_name="agent_runs")
    op.drop_table("agent_runs")
