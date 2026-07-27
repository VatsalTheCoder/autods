"""Live prediction against the saved model -- Section 8's "Done when" (spec 7.11).

Runs the real pipeline end to end against Postgres and object storage, then asks
the API for a prediction. The round trip is the point: the model is produced by
the worker, written to S3, and loaded back by a *different* process, which is the
only version of this that proves the artifact is genuinely self-contained. A test
that kept the fitted object in memory would pass without establishing anything.

The endpoint's contract is that a caller sends the columns they uploaded. So the
tests send raw values -- ``"London"``, ``"34"`` -- and never anything the recipe
produced, because a caller who had to know about ``city_London`` would be being
asked to reimplement the preprocessing.
"""

from __future__ import annotations

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.api.main import app
from app.core.db import SessionLocal, database_healthy
from app.core.storage import storage_healthy
from app.models.job import Job
from app.services.artifacts import (
    EXPLAINABILITY_ARTIFACT,
    FINAL_MODEL_ARTIFACT,
    FINAL_MODEL_INFO_ARTIFACT,
    load_artifact_bytes,
    load_json_artifact,
)
from app.worker.pipeline import run_pipeline

pytestmark = pytest.mark.skipif(
    not (database_healthy() and storage_healthy()),
    reason="needs Postgres and object storage (start with `make up`)",
)

N_ROWS = 60


def dataset_csv() -> bytes:
    """A dataset with signal in it, so a prediction is not a coin flip."""
    frame = pd.DataFrame(
        {
            "age": [22 + (i % 40) for i in range(N_ROWS)],
            "city": ["London", "Leeds", "Bristol"] * (N_ROWS // 3),
            "support_calls": [i % 8 for i in range(N_ROWS)],
            "churn": ["yes" if i % 8 > 3 else "no" for i in range(N_ROWS)],
        }
    )
    return frame.to_csv(index=False).encode("utf-8")


@pytest.fixture(autouse=True)
def _no_celery_dispatch(monkeypatch):
    """Confirming must not also hand the job to the live worker."""
    monkeypatch.setattr("app.api.routes.upload.enqueue_pipeline", lambda job_id: None)


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(app)


@pytest.fixture(scope="module")
def completed_job() -> int:
    """One real run, shared: the pipeline takes seconds, not milliseconds."""
    if not (database_healthy() and storage_healthy()):  # pragma: no cover
        pytest.skip("stack is down")

    with TestClient(app) as client:
        # The module-scoped fixture cannot use the function-scoped monkeypatch,
        # so the dispatch is stubbed directly for the one call that needs it.
        from app.api.routes import upload as upload_route

        original = upload_route.enqueue_pipeline
        upload_route.enqueue_pipeline = lambda job_id: None
        try:
            created = client.post(
                "/upload", files={"file": ("data.csv", dataset_csv(), "text/csv")}
            )
            job_id = created.json()["job_id"]
            confirmed = client.post(
                "/jobs",
                json={"job_id": job_id, "target_column": "churn", "task_type": "classification"},
            )
            assert confirmed.status_code == 200
        finally:
            upload_route.enqueue_pipeline = original

    run_pipeline(job_id)
    with SessionLocal() as db:
        job = db.get(Job, job_id)
        assert job.status.value == "completed", job.error_message
    return job_id


class TestTheModelIsSaved:
    def test_the_pickle_and_its_description_are_both_in_storage(self, completed_job):
        with SessionLocal() as db:
            assert load_artifact_bytes(db, completed_job, FINAL_MODEL_ARTIFACT)
            info = load_json_artifact(db, completed_job, FINAL_MODEL_INFO_ARTIFACT)
        assert info["model_name"]
        assert info["n_rows"] == N_ROWS

    def test_the_explainability_report_is_stored(self, completed_job):
        with SessionLocal() as db:
            report = load_json_artifact(db, completed_job, EXPLAINABILITY_ARTIFACT)
        assert report["global_importance"]
        assert report["feature_name_mapping"]

    def test_the_shap_charts_are_real_pngs(self, completed_job):
        with SessionLocal() as db:
            report = load_json_artifact(db, completed_job, EXPLAINABILITY_ARTIFACT)
            for name in report["plots"]:
                assert load_artifact_bytes(db, completed_job, name).startswith(b"\x89PNG"), name


class TestTheDescribeEndpoint:
    def test_it_lists_the_raw_columns_a_caller_should_send(self, client, completed_job):
        body = client.get(f"/jobs/{completed_job}/model").json()
        names = [column["name"] for column in body["feature_columns"]]
        assert names == ["age", "city", "support_calls"]
        assert "churn" not in names

    def test_it_carries_an_example_value_per_column(self, client, completed_job):
        body = client.get(f"/jobs/{completed_job}/model").json()
        assert all(column["example"] for column in body["feature_columns"])

    def test_the_explainability_endpoint_serves_the_report(self, client, completed_job):
        body = client.get(f"/jobs/{completed_job}/explainability").json()
        assert body["global_importance"][0]["feature"] in ("age", "city", "support_calls")
        # The names are the user's, not the recipe's.
        assert all(
            item["feature"] in ("age", "city", "support_calls")
            for item in body["global_importance"]
        )


class TestPredicting:
    def test_a_complete_row_gets_a_label_the_model_was_trained_on(self, client, completed_job):
        response = client.post(
            f"/jobs/{completed_job}/predict",
            json={"rows": [{"age": 41, "city": "London", "support_calls": 7}]},
        )
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["target_column"] == "churn"
        assert body["predictions"][0]["prediction"] in ("yes", "no")

    def test_probabilities_are_keyed_by_the_users_own_labels(self, client, completed_job):
        body = client.post(
            f"/jobs/{completed_job}/predict",
            json={"rows": [{"age": 41, "city": "London", "support_calls": 7}]},
        ).json()
        probabilities = body["predictions"][0]["probabilities"]
        assert set(probabilities) == {"yes", "no"}
        assert sum(probabilities.values()) == pytest.approx(1.0)

    def test_several_rows_come_back_in_order(self, client, completed_job):
        body = client.post(
            f"/jobs/{completed_job}/predict",
            json={
                "rows": [
                    {"age": 25, "city": "Leeds", "support_calls": 0},
                    {"age": 25, "city": "Leeds", "support_calls": 7},
                ]
            },
        ).json()
        assert len(body["predictions"]) == 2

    def test_strings_from_a_form_are_coerced_to_the_numbers_they_are(self, client, completed_job):
        """An HTML form sends "41", and the numeric branch would choke on it."""
        typed = client.post(
            f"/jobs/{completed_job}/predict",
            json={"rows": [{"age": 41, "city": "London", "support_calls": 7}]},
        ).json()
        as_strings = client.post(
            f"/jobs/{completed_job}/predict",
            json={"rows": [{"age": "41", "city": "London", "support_calls": "7"}]},
        ).json()
        assert as_strings["predictions"][0] == typed["predictions"][0]

    def test_a_missing_column_is_imputed_and_reported(self, client, completed_job):
        """A real prediction on partial information -- but never a silent one."""
        body = client.post(
            f"/jobs/{completed_job}/predict",
            json={"rows": [{"city": "London", "support_calls": 7}]},
        ).json()
        assert body["missing_columns"] == ["age"]
        assert body["predictions"][0]["prediction"] in ("yes", "no")

    def test_an_unknown_column_is_ignored_and_reported(self, client, completed_job):
        """Otherwise a misspelled name looks exactly like a correct one."""
        body = client.post(
            f"/jobs/{completed_job}/predict",
            json={"rows": [{"age": 41, "city": "London", "support_calls": 7, "aeg": 99}]},
        ).json()
        assert body["unexpected_columns"] == ["aeg"]

    def test_a_column_the_recipe_drops_is_not_demanded(self, client, completed_job):
        """A dropped column is part of the model's frame but not of its inputs.

        Reporting it as missing would send a user hunting for a value that
        changes nothing about the answer.
        """
        described = client.get(f"/jobs/{completed_job}/model").json()
        dropped = [c["name"] for c in described["feature_columns"] if not c["used"]]

        body = client.post(
            f"/jobs/{completed_job}/predict",
            json={"rows": [{"age": 41, "city": "London", "support_calls": 7}]},
        ).json()
        assert not [name for name in dropped if name in body["missing_columns"]]


class TestRefusals:
    def test_a_row_with_none_of_the_models_columns_is_rejected(self, client, completed_job):
        response = client.post(f"/jobs/{completed_job}/predict", json={"rows": [{"nonsense": 1}]})
        assert response.status_code == 400
        assert "input columns" in response.json()["detail"]

    def test_an_empty_request_is_rejected_by_the_schema(self, client, completed_job):
        assert client.post(f"/jobs/{completed_job}/predict", json={"rows": []}).status_code == 422

    def test_a_job_that_has_not_trained_yet_is_a_404(self, client):
        """An ordinary state a Results page polls into, not a fault."""
        created = client.post("/upload", files={"file": ("data.csv", dataset_csv(), "text/csv")})
        job_id = created.json()["job_id"]
        response = client.post(f"/jobs/{job_id}/predict", json={"rows": [{"age": 41}]})
        assert response.status_code == 404
        assert "no trained model yet" in response.json()["detail"]

    def test_an_unknown_job_is_a_404(self, client):
        assert client.post("/jobs/999999/predict", json={"rows": [{"age": 1}]}).status_code == 404
