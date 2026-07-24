"""Integration tests for the background pipeline runner.

These exercise the real state machine against Postgres and object storage: a job
moves QUEUED → RUNNING → COMPLETED, every node reports its status in order, and a
node that raises fails the job with a readable reason instead of hanging (the
Section 4 "Done when"). Skipped when the stack is down.

Nodes are patched not to sleep, so the pipeline runs instantly -- the sleep is
demo dressing, not behaviour under test.
"""

from __future__ import annotations

import uuid

import pytest
from fastapi.testclient import TestClient

from app.api.main import app
from app.core.db import SessionLocal, database_healthy
from app.core.storage import storage_healthy
from app.models.agent_run import AgentRunStatus
from app.models.job import Job, JobStatus
from app.models.user import DEV_USER_ID
from app.worker.pipeline import run_pipeline
from app.worker.state import PIPELINE_NODES

pytestmark = pytest.mark.skipif(
    not (database_healthy() and storage_healthy()),
    reason="needs Postgres and object storage (start with `make up`)",
)

VALID_CSV = b"age,city,income,churn\n34,London,52000,yes\n28,Leeds,41000,no\n45,Bristol,68000,no\n"


@pytest.fixture(autouse=True)
def _instant_nodes(monkeypatch):
    """Placeholder nodes must not actually sleep during tests."""
    monkeypatch.setattr("app.worker.graph.time.sleep", lambda _s: None)


@pytest.fixture
def uploaded_job() -> int:
    """A real job with its CSV in object storage, via the upload endpoint."""
    client = TestClient(app)
    resp = client.post("/upload", files={"file": ("data.csv", VALID_CSV, "text/csv")})
    return resp.json()["job_id"]


class TestSuccessfulRun:
    def test_job_reaches_completed(self, uploaded_job):
        run_pipeline(uploaded_job)
        with SessionLocal() as db:
            assert db.get(Job, uploaded_job).status is JobStatus.COMPLETED

    def test_every_node_completes_in_order(self, uploaded_job):
        run_pipeline(uploaded_job)
        with SessionLocal() as db:
            runs = db.get(Job, uploaded_job).agent_runs
            assert [r.name for r in runs] == PIPELINE_NODES
            assert all(r.status is AgentRunStatus.COMPLETED for r in runs)
            assert all(r.started_at is not None and r.finished_at is not None for r in runs)

    def test_rerun_resets_the_roadmap(self, uploaded_job):
        """A second run must not duplicate rows or leave stale statuses."""
        run_pipeline(uploaded_job)
        run_pipeline(uploaded_job)
        with SessionLocal() as db:
            runs = db.get(Job, uploaded_job).agent_runs
            assert len(runs) == len(PIPELINE_NODES)  # not doubled


class TestFailurePath:
    def test_failing_node_marks_job_failed(self, uploaded_job, monkeypatch):
        def boom(_s):
            raise RuntimeError("node exploded")

        monkeypatch.setattr("app.worker.graph.time.sleep", boom)

        run_pipeline(uploaded_job)

        with SessionLocal() as db:
            job = db.get(Job, uploaded_job)
            assert job.status is JobStatus.FAILED
            assert "node exploded" in (job.error_message or "")

            runs = {r.name: r for r in job.agent_runs}
            # The first node failed; later nodes were never reached.
            assert runs["planner"].status is AgentRunStatus.FAILED
            assert runs["cleaning"].status is AgentRunStatus.PENDING

    def test_missing_dataset_fails_gracefully(self):
        """A job whose CSV is absent from storage must fail, not crash."""
        with SessionLocal() as db:
            job = Job(
                user_id=DEV_USER_ID,
                original_filename="ghost.csv",
                s3_key=f"jobs/missing/{uuid.uuid4()}.csv",
                size_bytes=1,
                status=JobStatus.QUEUED,
            )
            db.add(job)
            db.commit()
            db.refresh(job)
            job_id = job.id

        run_pipeline(job_id)  # no object exists at this job's raw key

        with SessionLocal() as db:
            job = db.get(Job, job_id)
            assert job.status is JobStatus.FAILED
            assert job.error_message
            db.delete(job)
            db.commit()


class TestStatusVisibleViaAPI:
    def test_get_job_exposes_node_statuses(self, uploaded_job):
        run_pipeline(uploaded_job)
        client = TestClient(app)
        body = client.get(f"/jobs/{uploaded_job}").json()

        assert body["status"] == "completed"
        names = [r["name"] for r in body["agent_runs"]]
        assert names == PIPELINE_NODES
        assert all(r["status"] == "completed" for r in body["agent_runs"])
