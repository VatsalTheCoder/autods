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
    def test_confirm_sets_target_task_and_status(self, client, upload):
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
        assert body["status"] == JobStatus.CONFIRMED.value

        with SessionLocal() as db:
            job = db.get(Job, job_id)
            assert job.status is JobStatus.CONFIRMED
            assert job.target_column == "churn"

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
