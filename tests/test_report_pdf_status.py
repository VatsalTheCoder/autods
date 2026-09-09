"""What ``GET /jobs/{id}/report/pdf`` says when there is no PDF to serve.

Two quite different situations used to produce the same 404, separated only by
prose in the detail string: a run that had not reached the report yet, and a run
that finished while PDF rendering failed. A client cannot act on the difference
if the status code does not carry it, and the difference is the whole question a
client is asking -- keep waiting, or stop.

The PDF is best-effort on purpose, so that a font problem cannot discard a
finished analysis. That design is what makes the second case reachable at all,
and it is why these tests build the states directly rather than by running a
pipeline: no fixture can make WeasyPrint fail on demand, and the endpoint's
behaviour is a function of the job's status and the artifact's absence, which is
exactly what is constructed here.
"""

from __future__ import annotations

import uuid

import pytest
from fastapi.testclient import TestClient

from app.api.main import app
from app.core.db import SessionLocal, database_healthy
from app.models.job import Job, JobStatus
from app.models.user import DEV_USER_ID

pytestmark = pytest.mark.skipif(not database_healthy(), reason="needs Postgres (`make up`)")


def _job(status: JobStatus) -> int:
    """A job row in ``status`` with no artifacts of any kind."""
    with SessionLocal() as db:
        job = Job(
            user_id=DEV_USER_ID,
            original_filename="listings.csv",
            # Unique per call: s3_key carries a unique constraint, and these
            # states are built more than once across the tests below.
            s3_key=f"test/pdf-status/{status.value}/{uuid.uuid4()}",
            size_bytes=1,
            status=status,
        )
        db.add(job)
        db.commit()
        return job.id


class TestARunThatHasNotGotThereYet:
    """Still going, so the answer is "wait", and 202 is how that is spelled."""

    @pytest.mark.parametrize(
        "status", [JobStatus.UPLOADED, JobStatus.CONFIRMED, JobStatus.QUEUED, JobStatus.RUNNING]
    )
    def test_an_unfinished_run_is_202_not_404(self, status):
        resp = TestClient(app).get(f"/jobs/{_job(status)}/report/pdf")
        assert resp.status_code == 202, resp.json()

    def test_the_message_names_the_state_and_points_at_the_markdown(self):
        resp = TestClient(app).get(f"/jobs/{_job(JobStatus.RUNNING)}/report/pdf")
        detail = resp.json()["detail"]
        assert "running" in detail
        assert "/report" in detail


class TestARunThatFinishedWithoutOne:
    """Terminal: rendering failed, and no amount of polling will fix it."""

    @pytest.mark.parametrize("status", [JobStatus.COMPLETED, JobStatus.FAILED])
    def test_a_finished_run_with_no_pdf_is_404(self, status):
        resp = TestClient(app).get(f"/jobs/{_job(status)}/report/pdf")
        assert resp.status_code == 404, resp.json()

    def test_the_message_says_waiting_will_not_help(self):
        resp = TestClient(app).get(f"/jobs/{_job(JobStatus.COMPLETED)}/report/pdf")
        detail = resp.json()["detail"]
        assert "waiting will not produce one" in detail
        assert "/report" in detail


class TestAJobThatDoesNotExist:
    def test_an_unknown_job_is_still_404(self):
        """The one 404 that was never ambiguous, and must not become a 202."""
        resp = TestClient(app).get("/jobs/99999999/report/pdf")
        assert resp.status_code == 404
        assert "No job" in resp.json()["detail"]
