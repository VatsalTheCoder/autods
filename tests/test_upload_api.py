"""Tests for the upload endpoint.

These run against a real Postgres and real object storage, because the
behaviour worth testing here *is* the interaction between them -- specifically
that a rejected upload leaves nothing behind in either. Mocking both would test
only that the mocks were called.

Skipped automatically when the stack is not running, so the suite stays green
outside Docker.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from app.api.main import app
from app.core.db import SessionLocal, database_healthy
from app.core.storage import object_exists, storage_healthy
from app.models.artifact import Artifact
from app.models.job import Job, JobStatus

pytestmark = pytest.mark.skipif(
    not (database_healthy() and storage_healthy()),
    reason="needs Postgres and object storage (start with `make up`)",
)

VALID_CSV = b"age,city,income\n34,London,52000\n28,Leeds,41000\n45,Bristol,68000\n"


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def _post(client: TestClient, content: bytes, filename: str = "data.csv"):
    return client.post("/upload", files={"file": (filename, content, "text/csv")})


class TestSuccessfulUpload:
    def test_returns_job_id_and_preview(self, client):
        response = _post(client, VALID_CSV)

        assert response.status_code == 201
        body = response.json()
        assert body["job_id"] > 0
        assert body["status"] == JobStatus.UPLOADED.value
        assert body["preview"]["n_rows"] == 3
        assert body["preview"]["columns"] == ["age", "city", "income"]
        assert len(body["preview"]["rows"]) == 3

    def test_persists_the_job_row(self, client):
        job_id = _post(client, VALID_CSV).json()["job_id"]

        with SessionLocal() as db:
            job = db.get(Job, job_id)
            assert job is not None
            assert job.n_rows == 3
            assert job.n_columns == 3
            assert job.status is JobStatus.UPLOADED

    def test_stores_the_file_in_object_storage(self, client):
        job_id = _post(client, VALID_CSV).json()["job_id"]

        with SessionLocal() as db:
            job = db.get(Job, job_id)
            assert object_exists(job.s3_key)

    def test_registers_the_raw_dataset_artifact(self, client):
        job_id = _post(client, VALID_CSV).json()["job_id"]

        with SessionLocal() as db:
            artifacts = (
                db.execute(select(Artifact).where(Artifact.job_id == job_id)).scalars().all()
            )

            names = {a.name for a in artifacts}
            # The raw dataset, plus the schema report detected at upload (Section 3).
            assert "dataset.csv" in names
            assert "schema_report.json" in names

    def test_each_upload_gets_its_own_storage_key(self, client):
        first = _post(client, VALID_CSV).json()["job_id"]
        second = _post(client, VALID_CSV).json()["job_id"]

        with SessionLocal() as db:
            assert db.get(Job, first).s3_key != db.get(Job, second).s3_key


class TestRejectedUpload:
    @pytest.mark.parametrize(
        ("content", "filename"),
        [
            (b"age\n1\n2\n", "data.csv"),  # single column
            (b"age,city\n", "data.csv"),  # headers only
            (b"", "data.csv"),  # empty
            (VALID_CSV, "data.xlsx"),  # wrong extension
        ],
    )
    def test_returns_422_with_a_readable_reason(self, client, content, filename):
        response = _post(client, content, filename)

        assert response.status_code == 422
        assert len(response.json()["detail"]) > 10  # a sentence, not a code

    def test_leaves_no_job_row_behind(self, client):
        """A rejected upload must not create a half-formed job."""
        with SessionLocal() as db:
            before = db.execute(select(Job)).scalars().all()
            before_ids = {job.id for job in before}

        _post(client, b"age\n1\n2\n")

        with SessionLocal() as db:
            after_ids = {job.id for job in db.execute(select(Job)).scalars().all()}

        assert after_ids == before_ids


class TestJobEndpoints:
    def test_lists_jobs_newest_first(self, client):
        _post(client, VALID_CSV)
        latest_id = _post(client, VALID_CSV).json()["job_id"]

        jobs = client.get("/jobs").json()

        assert jobs[0]["id"] == latest_id

    def test_fetches_a_single_job(self, client):
        job_id = _post(client, VALID_CSV).json()["job_id"]

        body = client.get(f"/jobs/{job_id}").json()

        assert body["id"] == job_id
        assert body["n_rows"] == 3

    def test_unknown_job_is_404(self, client):
        assert client.get("/jobs/99999999").status_code == 404

    def test_lists_artifacts(self, client):
        job_id = _post(client, VALID_CSV).json()["job_id"]

        artifacts = client.get(f"/jobs/{job_id}/artifacts").json()

        raw = next(a for a in artifacts if a["name"] == "dataset.csv")
        assert raw["kind"] == "raw_dataset"


class TestArtifactLinks:
    def test_returns_a_working_presigned_url(self, client):
        job_id = _post(client, VALID_CSV).json()["job_id"]

        body = client.get(f"/jobs/{job_id}/artifacts/dataset.csv/link").json()

        assert body["url"].startswith("http")
        assert "X-Amz-Signature" in body["url"]

    def test_unknown_artifact_is_404(self, client):
        job_id = _post(client, VALID_CSV).json()["job_id"]

        assert client.get(f"/jobs/{job_id}/artifacts/nope.png/link").status_code == 404
