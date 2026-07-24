"""Agent-run model -- one row per pipeline node per job.

The pipeline is a sequence of nodes (Planner, Cleaning, ... Report). This table
records each node's status and timing for a given job, which is exactly what the
Progress page polls: which node is running now, which are done, which failed and
why (spec 12.1, build-plan Section 4).

Rows are created up front as PENDING when the job is queued, so the UI can show
the whole roadmap greyed out and light each step up as the worker reaches it,
rather than steps popping into existence one by one.
"""

from __future__ import annotations

import enum
from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import (
    DateTime,
    Enum,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.db import Base

if TYPE_CHECKING:
    from app.models.job import Job


class AgentRunStatus(enum.StrEnum):
    PENDING = "pending"  # created, not yet reached
    RUNNING = "running"  # currently executing
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"  # a conditional node the Planner turned off (Section 7)


class AgentRun(Base):
    __tablename__ = "agent_runs"
    __table_args__ = (
        # A job runs each node at most once, so (job, name) is unique. Also the
        # natural key the worker uses to update a node's status.
        UniqueConstraint("job_id", "name", name="uq_agent_run_job_name"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    job_id: Mapped[int] = mapped_column(
        ForeignKey("jobs.id", ondelete="CASCADE"), nullable=False, index=True
    )

    name: Mapped[str] = mapped_column(String(64), nullable=False)
    # Execution order, so the UI can render nodes in pipeline order regardless
    # of how the database returns them.
    sequence: Mapped[int] = mapped_column(Integer, nullable=False)

    status: Mapped[AgentRunStatus] = mapped_column(
        Enum(
            AgentRunStatus, name="agent_run_status", values_callable=lambda e: [m.value for m in e]
        ),
        default=AgentRunStatus.PENDING,
        nullable=False,
    )

    # Set when this specific node fails, so the UI can point at the exact step.
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)

    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    job: Mapped[Job] = relationship(back_populates="agent_runs")

    def __repr__(self) -> str:
        return f"<AgentRun job={self.job_id} name={self.name!r} status={self.status.value}>"
