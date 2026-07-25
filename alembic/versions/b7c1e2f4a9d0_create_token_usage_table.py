"""create token_usage table

Records every LLM request per agent per job for cost tracking (spec section 13),
introduced with the Section 2 LLM client.

Revision ID: b7c1e2f4a9d0
Revises: d34db10c97b6
Create Date: 2026-07-24 10:00:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "b7c1e2f4a9d0"
down_revision: str | None = "d34db10c97b6"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "token_usage",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("job_id", sa.Integer(), nullable=False),
        sa.Column("agent", sa.String(length=64), nullable=False),
        sa.Column("model", sa.String(length=128), nullable=False),
        sa.Column("tier", sa.String(length=16), nullable=False),
        sa.Column("input_tokens", sa.Integer(), nullable=False),
        sa.Column("output_tokens", sa.Integer(), nullable=False),
        sa.Column("cost_usd", sa.Numeric(precision=12, scale=6), nullable=False),
        sa.Column("estimated", sa.Boolean(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["job_id"], ["jobs.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_token_usage_job_id"), "token_usage", ["job_id"], unique=False)
    op.create_index(op.f("ix_token_usage_agent"), "token_usage", ["agent"], unique=False)


def downgrade() -> None:
    op.drop_index(op.f("ix_token_usage_agent"), table_name="token_usage")
    op.drop_index(op.f("ix_token_usage_job_id"), table_name="token_usage")
    op.drop_table("token_usage")
