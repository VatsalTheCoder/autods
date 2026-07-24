"""Job model.

One row per uploaded dataset. The row is the job's single source of truth:
the API creates it on upload, Section 3 attaches the confirmed schema, and
Section 4's worker updates its status as the pipeline runs.

The CSV itself is not stored here -- only its object-storage key. Postgres
holds metadata; S3 holds bytes (spec section 13).
"""

from __future__ import annotations

import enum
from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import (
    BigInteger,
    DateTime,
    Enum,
    ForeignKey,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.db import Base

if TYPE_CHECKING:
    from app.models.agent_run import AgentRun
    from app.models.artifact import Artifact
    from app.models.token_usage import TokenUsage
    from app.models.user import User


class JobStatus(enum.StrEnum):
    """Lifecycle of a job.

    Section 1 only ever produces UPLOADED. The later states are declared now
    so the enum type does not need a migration every time a section lands --
    altering an enum in Postgres is more awkward than defining it once.
    """

    UPLOADED = "uploaded"  # CSV stored, awaiting schema confirmation
    CONFIRMED = "confirmed"  # user approved the schema (Section 3)
    QUEUED = "queued"  # handed to the worker (Section 4)
    RUNNING = "running"  # pipeline executing (Section 4)
    COMPLETED = "completed"
    FAILED = "failed"


class Job(Base):
    __tablename__ = "jobs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )

    original_filename: Mapped[str] = mapped_column(String(255), nullable=False)
    s3_key: Mapped[str] = mapped_column(String(512), nullable=False, unique=True)
    size_bytes: Mapped[int] = mapped_column(BigInteger, nullable=False)

    status: Mapped[JobStatus] = mapped_column(
        Enum(JobStatus, name="job_status", values_callable=lambda e: [m.value for m in e]),
        default=JobStatus.UPLOADED,
        nullable=False,
        index=True,
    )

    # Populated at upload from the parsed CSV.
    n_rows: Mapped[int | None] = mapped_column(Integer, nullable=True)
    n_columns: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # Set when the user confirms the schema (Section 3). The full confirmed
    # schema is stored as a JSON artifact; these two fields are mirrored onto
    # the row because every later stage needs the target and task type, and a
    # column read beats fetching and parsing an S3 object each time.
    target_column: Mapped[str | None] = mapped_column(String(255), nullable=True)
    task_type: Mapped[str | None] = mapped_column(String(32), nullable=True)

    # Set when status is FAILED, so the UI can show why rather than just that.
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )

    user: Mapped[User] = relationship(back_populates="jobs")
    artifacts: Mapped[list[Artifact]] = relationship(
        back_populates="job", cascade="all, delete-orphan"
    )
    token_usage: Mapped[list[TokenUsage]] = relationship(
        back_populates="job", cascade="all, delete-orphan"
    )
    agent_runs: Mapped[list[AgentRun]] = relationship(
        back_populates="job",
        cascade="all, delete-orphan",
        # Always hand them back in pipeline order, so the UI never has to sort.
        order_by="AgentRun.sequence",
    )

    def __repr__(self) -> str:
        return f"<Job id={self.id} file={self.original_filename!r} status={self.status.value}>"
