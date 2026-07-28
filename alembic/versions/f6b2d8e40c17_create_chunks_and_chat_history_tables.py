"""create run_chunks and chat_history tables, and the pgvector extension

Section 10's storage (spec 7.13, 12.2). The vector store is Postgres rather than
a seventh container: the spec locks ChromaDB and documents pgvector as the
fallback, and Section 11 would otherwise have to give ChromaDB a persistent EBS
volume of its own.

The embedding column is fixed at 384 dimensions, which is BAAI/bge-small-en-v1.5's
output width. That is deliberately not configurable: pgvector needs a declared
width to build an index, and a vector column silently holding two different
models' output would return nonsense rather than an error.

Revision ID: f6b2d8e40c17
Revises: e5a9c7d1f30b
Create Date: 2026-07-28 19:20:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa
from pgvector.sqlalchemy import Vector

from alembic import op

revision: str = "f6b2d8e40c17"
down_revision: str | None = "e5a9c7d1f30b"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

EMBEDDING_DIMENSIONS = 384


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.create_table(
        "run_chunks",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("job_id", sa.Integer(), nullable=False),
        # Which artifact the passage came from, so an answer can cite it rather
        # than assert it -- the difference between a grounded answer and a
        # confident one.
        sa.Column("source", sa.String(length=64), nullable=False),
        sa.Column("heading", sa.String(length=255), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        # Position within the source, so retrieved passages can be shown in the
        # order they were written rather than by score.
        sa.Column("ordinal", sa.Integer(), nullable=False),
        sa.Column("embedding", Vector(EMBEDDING_DIMENSIONS), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["job_id"], ["jobs.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_run_chunks_job_id"), "run_chunks", ["job_id"], unique=False)

    # An ANN index is deliberately *not* created. A run produces tens of chunks,
    # not millions, and on that scale an exact scan is both faster and exactly
    # right -- while an IVFFlat index needs training data it does not have yet
    # and would return approximate results for no gain. Section 11 is the place
    # to revisit this if many runs ever share one table.

    op.create_table(
        "chat_history",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("job_id", sa.Integer(), nullable=False),
        sa.Column("question", sa.Text(), nullable=False),
        sa.Column("answer", sa.Text(), nullable=False),
        # Which tool answered. Stored because the routing decision is the
        # interesting part of this section, and a stored history that does not
        # record it cannot be used to check the router's behaviour after the fact.
        sa.Column("route", sa.String(length=16), nullable=False),
        # What the answer was grounded in: retrieved chunk ids, or the pandas
        # expression that was evaluated.
        sa.Column("grounding", sa.Text(), nullable=False, server_default=""),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["job_id"], ["jobs.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_chat_history_job_id"), "chat_history", ["job_id"], unique=False)


def downgrade() -> None:
    op.drop_index(op.f("ix_chat_history_job_id"), table_name="chat_history")
    op.drop_table("chat_history")
    op.drop_index(op.f("ix_run_chunks_job_id"), table_name="run_chunks")
    op.drop_table("run_chunks")
    # The extension is left in place. Dropping it would break any other database
    # object using the type, and it is harmless when unused.
