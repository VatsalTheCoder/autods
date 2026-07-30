"""Tests for the schema-detection API surface.

Runs against real Postgres and object storage (the interaction is the point),
skipped when the stack is down. The LLM is never called for real: the default
path has no key and degrades to profiling, and the enrichment path is driven by
a FakeLLM injected through FastAPI's dependency override.
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from app.api.main import app
from app.core.db import SessionLocal, database_healthy
from app.core.llm import FakeLLM
from app.core.llm.factory import get_optional_llm
from app.core.storage import storage_healthy
from app.models.job import Job, JobStatus
from app.models.token_usage import TokenUsage

pytestmark = pytest.mark.skipif(
    not (database_healthy() and storage_healthy()),
    reason="needs Postgres and object storage (start with `make up`)",
)

VALID_CSV = b"age,city,income,churn\n34,London,52000,yes\n28,Leeds,41000,no\n45,Bristol,68000,no\n"


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


@pytest.fixture
def upload(client):
    def _upload():
        return client.post("/upload", files={"file": ("data.csv", VALID_CSV, "text/csv")})

    return _upload


def _fake_inference() -> str:
    return json.dumps(
        {
            "columns": [
                {"name": "age", "meaning": "age in years", "is_pii": False},
                {"name": "city", "meaning": "home city", "is_pii": False},
                {"name": "income", "meaning": "annual income", "is_pii": False},
                {"name": "churn", "meaning": "did they leave", "is_pii": False},
            ],
            "suggested_target": "churn",
            "task_type": "classification",
        }
    )


class TestUploadReturnsSchema:
    def test_response_includes_a_schema_report(self, upload):
        body = upload().json()
        report = body["schema_report"]
        assert report["n_columns"] == 4
        assert report["suggested_target"] == "churn"
        assert report["task_type"] == "classification"
        # No key in the test environment, so enrichment is skipped.
        assert report["llm_enriched"] is False

    def test_schema_report_artifact_is_persisted(self, client, upload):
        job_id = upload().json()["job_id"]
        fetched = client.get(f"/jobs/{job_id}/schema")
        assert fetched.status_code == 200
        assert fetched.json()["suggested_target"] == "churn"

    def test_schema_for_unknown_job_is_404(self, client):
        assert client.get("/jobs/99999999/schema").status_code == 404


class TestLLMEnrichedUpload:
    def test_injected_fake_llm_enriches_the_report(self, client, upload):
        app.dependency_overrides[get_optional_llm] = lambda: FakeLLM([_fake_inference()])
        try:
            report = upload().json()["schema_report"]
        finally:
            app.dependency_overrides.pop(get_optional_llm, None)

        assert report["llm_enriched"] is True
        meanings = {c["name"]: c["meaning"] for c in report["columns"]}
        assert meanings["income"] == "annual income"

    def test_enrichment_logs_token_usage(self, client, upload):
        app.dependency_overrides[get_optional_llm] = lambda: FakeLLM([_fake_inference()])
        try:
            job_id = upload().json()["job_id"]
        finally:
            app.dependency_overrides.pop(get_optional_llm, None)

        with SessionLocal() as db:
            rows = db.query(TokenUsage).filter(TokenUsage.job_id == job_id).all()
            assert len(rows) == 1
            assert rows[0].agent == "schema_detection"


class TestConfirmation:
    @pytest.fixture(autouse=True)
    def stub_enqueue(self, monkeypatch):
        """Confirming now dispatches the pipeline; stub the dispatch so the live
        worker does not asynchronously mutate status mid-assertion. Returns the
        list of job ids that would have been queued."""
        calls: list[int] = []
        monkeypatch.setattr(
            "app.api.routes.upload.enqueue_pipeline", lambda job_id: calls.append(job_id)
        )
        return calls

    def test_confirm_sets_target_task_and_queues(self, client, upload, stub_enqueue):
        job_id = upload().json()["job_id"]

        resp = client.post(
            "/jobs",
            json={
                "job_id": job_id,
                "target_column": "churn",
                "task_type": "classification",
                "columns": [
                    {"name": "income", "is_pii": False, "exclude": True},
                ],
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["target_column"] == "churn"
        assert body["task_type"] == "classification"
        # Confirming launches the pipeline, so the job is QUEUED, not just saved.
        assert body["status"] == JobStatus.QUEUED.value

        with SessionLocal() as db:
            job = db.get(Job, job_id)
            assert job.status is JobStatus.QUEUED
            assert job.target_column == "churn"

        # The pipeline was handed to the worker exactly once, for this job.
        assert stub_enqueue == [job_id]

    def test_confirmed_schema_artifact_is_saved(self, client, upload):
        job_id = upload().json()["job_id"]
        client.post(
            "/jobs",
            json={"job_id": job_id, "target_column": "churn", "task_type": "classification"},
        )
        artifacts = client.get(f"/jobs/{job_id}/artifacts").json()
        names = {a["name"] for a in artifacts}
        assert "confirmed_schema.json" in names

    def test_target_not_in_dataset_is_rejected(self, client, upload):
        job_id = upload().json()["job_id"]
        resp = client.post(
            "/jobs",
            json={
                "job_id": job_id,
                "target_column": "not_a_column",
                "task_type": "classification",
            },
        )
        assert resp.status_code == 422

    def test_confirm_unknown_job_is_404(self, client):
        resp = client.post(
            "/jobs",
            json={"job_id": 99999999, "target_column": "x", "task_type": "regression"},
        )
        assert resp.status_code == 404


class TestUnknownFieldsAreRejected:
    """A silently-ignored field here decides whether PII reaches the model.

    ``exclude`` is easy to guess wrong -- ``include`` is the obvious alternative
    and it is *inverted*, so a caller who guesses gets the opposite of what they
    asked for. Pydantic ignores unknown keys by default, which turned that
    mistake into a 200 and a model quietly trained on the column the caller
    meant to withhold. It has to be a 422.
    """

    def test_a_misnamed_column_field_is_a_422_not_a_silent_default(self, client, upload):
        job_id = upload().json()["job_id"]
        response = client.post(
            "/jobs",
            json={
                "job_id": job_id,
                "target_column": "churn",
                "task_type": "classification",
                "columns": [{"name": "city", "include": False}],
            },
        )
        assert response.status_code == 422
        # The message has to name the offending field, or it does not help.
        assert "include" in response.text

    def test_a_misnamed_top_level_field_is_also_rejected(self, client, upload):
        job_id = upload().json()["job_id"]
        response = client.post(
            "/jobs",
            json={
                "job_id": job_id,
                "target_column": "churn",
                "task_type": "classification",
                "targets": ["churn"],
            },
        )
        assert response.status_code == 422

    def test_the_correct_shape_still_works(self, client, upload):
        job_id = upload().json()["job_id"]
        response = client.post(
            "/jobs",
            json={
                "job_id": job_id,
                "target_column": "churn",
                "task_type": "classification",
                "columns": [{"name": "city", "is_pii": False, "exclude": True}],
            },
        )
        assert response.status_code == 200


PII_CSV = (
    b"email,age,city,churn\n"
    b"a@example.com,34,London,yes\n"
    b"b@example.com,28,Leeds,no\n"
    b"c@example.com,45,Bristol,no\n"
)


class TestOmittedColumnsInheritDetectedExclusions:
    """An omitted ``columns`` must not silently un-exclude the PII.

    Detection sets ``exclude=True`` on PII and ColumnProfile promises that the
    safe choice is the default. ``confirm_job`` used to take the request
    literally, so a caller who left the array out got a 200 and a model trained
    on the flagged column -- seen on a real run, where the pipeline modelled an
    email and the critic then remarked on the model's reliance on it.
    """

    @pytest.fixture
    def pii_job(self, client) -> int:
        response = client.post("/upload", files={"file": ("pii.csv", PII_CSV, "text/csv")})
        report = response.json()["schema_report"]
        # Guard the premise: without a flagged column these tests prove nothing.
        flagged = {c["name"] for c in report["columns"] if c["is_pii"]}
        assert "email" in flagged, f"expected email to be flagged as PII, got {flagged}"
        assert next(c for c in report["columns"] if c["name"] == "email")["exclude"] is True
        return response.json()["job_id"]

    def _confirmed(self, job_id: int) -> dict:
        from app.services.artifacts import CONFIRMED_SCHEMA_ARTIFACT, load_json_artifact

        with SessionLocal() as db:
            return load_json_artifact(db, job_id, CONFIRMED_SCHEMA_ARTIFACT)

    def test_an_omitted_list_keeps_the_pii_excluded(self, client, pii_job):
        response = client.post(
            "/jobs",
            json={"job_id": pii_job, "target_column": "churn", "task_type": "classification"},
        )
        assert response.status_code == 200

        confirmed = self._confirmed(pii_job)
        excluded = {c["name"] for c in confirmed["columns"] if c["exclude"]}
        assert "email" in excluded

    def test_the_non_pii_columns_are_not_excluded_by_inheriting(self, client, pii_job):
        """Inheriting must copy detection's decision, not exclude everything."""
        client.post(
            "/jobs",
            json={"job_id": pii_job, "target_column": "churn", "task_type": "classification"},
        )
        confirmed = self._confirmed(pii_job)
        excluded = {c["name"] for c in confirmed["columns"] if c["exclude"]}
        assert excluded == {"email"}

    def test_an_explicit_choice_still_wins(self, client, pii_job):
        """Opting back in stays possible -- inheritance is a default, not a lock."""
        response = client.post(
            "/jobs",
            json={
                "job_id": pii_job,
                "target_column": "churn",
                "task_type": "classification",
                "columns": [{"name": "email", "is_pii": True, "exclude": False}],
            },
        )
        assert response.status_code == 200

        confirmed = self._confirmed(pii_job)
        assert [c["name"] for c in confirmed["columns"] if c["exclude"]] == []

    def test_a_pii_target_is_never_excluded_by_inheriting(self, client):
        """Excluding the target would drop the column the run is about.

        Nothing downstream validates against an excluded target, so inheriting
        one would turn a confirmed run into a nonsensical one.
        """
        response = client.post("/upload", files={"file": ("pii.csv", PII_CSV, "text/csv")})
        job_id = response.json()["job_id"]

        confirm = client.post(
            "/jobs",
            json={"job_id": job_id, "target_column": "email", "task_type": "classification"},
        )
        assert confirm.status_code == 200

        confirmed = self._confirmed(job_id)
        email = next(c for c in confirmed["columns"] if c["name"] == "email")
        assert email["exclude"] is False
        # Still recorded as PII -- the flag is information, the exclusion is policy.
        assert email["is_pii"] is True
