"""Recording pipeline progress to the database.

The Progress page polls the API while the worker runs, so every status change
has to be *committed* the instant it happens -- a transaction held open for the
whole pipeline would make the UI show nothing until the very end, defeating the
point of a progress bar. Each function here therefore opens a short-lived
session and commits immediately.

These helpers own the two status surfaces: the per-node ``agent_runs`` rows and
the overall ``jobs.status``. Keeping both here means the state machine (queued →
running → completed/failed, and pending → running → completed/failed per node)
lives in one readable place.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime

from sqlalchemy import delete, select

from app.core.db import SessionLocal
from app.models.agent_run import AgentRun, AgentRunStatus
from app.models.job import Job, JobStatus

logger = logging.getLogger(__name__)


def _now() -> datetime:
    return datetime.now(UTC)


def init_agent_runs(job_id: int, node_names: list[str]) -> None:
    """Create the PENDING rows for every node, so the UI shows the full roadmap.

    Idempotent: a re-queued job clears its previous rows first, so a retry does
    not collide with the ``(job_id, name)`` uniqueness constraint or show stale
    statuses from the last attempt.
    """
    with SessionLocal() as db:
        db.execute(delete(AgentRun).where(AgentRun.job_id == job_id))
        db.add_all(
            AgentRun(job_id=job_id, name=name, sequence=i, status=AgentRunStatus.PENDING)
            for i, name in enumerate(node_names)
        )
        db.commit()


def start_node(job_id: int, name: str) -> None:
    """Mark a node RUNNING and stamp its start time."""
    _update_node(job_id, name, status=AgentRunStatus.RUNNING, started=True)


def finish_node(job_id: int, name: str) -> None:
    """Mark a node COMPLETED and stamp its finish time."""
    _update_node(job_id, name, status=AgentRunStatus.COMPLETED, finished=True)


def fail_node(job_id: int, name: str, error: str) -> None:
    """Mark a node FAILED with the reason, so the UI can point at the exact step."""
    _update_node(job_id, name, status=AgentRunStatus.FAILED, finished=True, error=error)


def _update_node(
    job_id: int,
    name: str,
    *,
    status: AgentRunStatus,
    started: bool = False,
    finished: bool = False,
    error: str | None = None,
) -> None:
    with SessionLocal() as db:
        run = db.execute(
            select(AgentRun).where(AgentRun.job_id == job_id, AgentRun.name == name)
        ).scalar_one_or_none()
        if run is None:
            # A node without a pre-created row should not happen, but never let
            # bookkeeping crash the pipeline -- log and move on.
            logger.warning("No agent_run row for job %s node %s", job_id, name)
            return
        run.status = status
        if started:
            run.started_at = _now()
        if finished:
            run.finished_at = _now()
        if error is not None:
            run.error_message = error
        db.commit()


def set_job_status(job_id: int, status: JobStatus, error: str | None = None) -> None:
    """Move the overall job to a new status, optionally recording why it failed."""
    with SessionLocal() as db:
        job = db.get(Job, job_id)
        if job is None:
            logger.warning("set_job_status: no job %s", job_id)
            return
        job.status = status
        if error is not None:
            job.error_message = error
        db.commit()
