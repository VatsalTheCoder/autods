"""The pipeline end to end on a *regression* task.

``test_pipeline.py`` runs the whole pipeline, and until now it only ever ran a
classification job. Regression takes a different route through most of the
stages -- ``KFold`` rather than ``StratifiedKFold``, no resampling, r²/RMSE/MAE
rather than F1, a different SHAP branch and a different report -- and none of
that had been exercised as a whole, in tests or in a real run.

The dataset here is small and synthetic but its *generative relationship is
known*, which is what makes these assertions worth more than "it did not
crash": the model is checked against the relationship it was supposed to
recover, and the SHAP directions against the signs that produced the data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.api.main import app
from app.core.db import SessionLocal, database_healthy
from app.core.storage import storage_healthy
from app.models.job import Job, JobStatus
from app.services.artifacts import (
    EVALUATION_ARTIFACT,
    EXPLAINABILITY_ARTIFACT,
    REPORT_ARTIFACT,
    load_artifact_bytes,
    load_json_artifact,
)
from app.worker.pipeline import run_pipeline

pytestmark = pytest.mark.skipif(
    not (database_healthy() and storage_healthy()),
    reason="needs Postgres and object storage (start with `make up`)",
)

N_ROWS = 90


def dataset_csv() -> bytes:
    """Rooms and area drive the rent; distance works against it.

    The coefficients are the point. ``area`` is built to dominate and
    ``distance_km`` to push the other way, so the explainability output can be
    checked against the truth rather than merely inspected.
    """
    rng = np.random.default_rng(11)
    area = rng.integers(400, 1800, N_ROWS)
    rooms = np.clip((area / 450).round(), 1, 4).astype(int)
    distance_km = rng.uniform(0.2, 9.0, N_ROWS).round(2)
    district = np.array(["central", "suburb", "outskirts"] * (N_ROWS // 3))

    rent = 900 + 1.8 * area + 140 * rooms - 55 * distance_km + rng.normal(0, 60, N_ROWS)

    frame = pd.DataFrame(
        {
            "area": area,
            "rooms": rooms,
            "distance_km": distance_km,
            "district": district,
            "rent": rent.round(2),
        }
    )
    # One gap and one duplicate, so imputation and cleaning both have work.
    frame.loc[5, "area"] = None
    frame = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    return frame.to_csv(index=False).encode("utf-8")


@pytest.fixture(autouse=True)
def _no_celery_dispatch(monkeypatch):
    monkeypatch.setattr("app.api.routes.upload.enqueue_pipeline", lambda job_id: None)


@pytest.fixture
def completed_regression_job() -> int:
    client = TestClient(app)
    upload = client.post("/upload", files={"file": ("rents.csv", dataset_csv(), "text/csv")})
    job_id = upload.json()["job_id"]

    confirm = client.post(
        "/jobs",
        json={"job_id": job_id, "target_column": "rent", "task_type": "regression"},
    )
    assert confirm.status_code == 200, confirm.text

    run_pipeline(job_id)
    return job_id


class TestItCompletes:
    def test_the_job_reaches_completed(self, completed_regression_job):
        with SessionLocal() as db:
            job = db.get(Job, completed_regression_job)
            assert job.status is JobStatus.COMPLETED, job.error_message


class TestItTookTheRegressionRoute:
    """The assertions that would pass on a classification run are not enough."""

    def test_the_cross_validator_is_not_stratified(self, completed_regression_job):
        """Stratification needs classes. On a continuous target it is wrong."""
        with SessionLocal() as db:
            evaluation = load_json_artifact(db, completed_regression_job, EVALUATION_ARTIFACT)
        assert evaluation["cv_strategy"] == "KFold"
        assert "Stratified" not in evaluation["cv_strategy"]

    def test_the_metrics_are_regression_metrics(self, completed_regression_job):
        with SessionLocal() as db:
            evaluation = load_json_artifact(db, completed_regression_job, EVALUATION_ARTIFACT)

        assert evaluation["primary_metric"] == "r2"
        assert {"r2", "mae", "rmse"} <= set(evaluation["metrics"])
        # Classification metrics must be absent, not merely unused: their
        # presence would mean something computed them on a continuous target.
        assert not {"f1_macro", "accuracy", "roc_auc"} & set(evaluation["metrics"])

    def test_the_model_recovered_the_relationship(self, completed_regression_job):
        """The data is linear with modest noise, so a weak fit means a real fault."""
        with SessionLocal() as db:
            evaluation = load_json_artifact(db, completed_regression_job, EVALUATION_ARTIFACT)
        assert evaluation["metrics"]["r2"]["mean"] > 0.8

    def test_no_resampling_happened(self, completed_regression_job):
        """SMOTE needs a minority class; there is no such thing here."""
        with SessionLocal() as db:
            markdown = load_artifact_bytes(db, completed_regression_job, REPORT_ARTIFACT).decode(
                "utf-8"
            )
        assert "SMOTE" not in markdown

    def test_the_report_leads_with_a_regression_score(self, completed_regression_job):
        with SessionLocal() as db:
            markdown = load_artifact_bytes(db, completed_regression_job, REPORT_ARTIFACT).decode(
                "utf-8"
            )
        assert "R²" in markdown
        assert "F1" not in markdown
        assert "confusion matrix" not in markdown.lower()


class TestExplainingAContinuousTarget:
    def test_shap_still_adds_up(self, completed_regression_job):
        """Additivity is the check that the decomposition means anything."""
        with SessionLocal() as db:
            report = load_json_artifact(db, completed_regression_job, EXPLAINABILITY_ARTIFACT)
        assert report["additivity_max_error"] < 0.05

    def test_the_directions_match_the_signs_that_made_the_data(self, completed_regression_job):
        """``area`` was built to raise the rent and ``distance_km`` to lower it."""
        with SessionLocal() as db:
            report = load_json_artifact(db, completed_regression_job, EXPLAINABILITY_ARTIFACT)

        directions = {item["feature"]: item["direction"] for item in report["global_importance"]}
        if directions.get("area"):
            assert "up" in directions["area"]
        if directions.get("distance_km"):
            assert "down" in directions["distance_km"]

    def test_there_are_no_classes_to_explain(self, completed_regression_job):
        """A continuous target has no class labels; claiming otherwise is a bug."""
        with SessionLocal() as db:
            report = load_json_artifact(db, completed_regression_job, EXPLAINABILITY_ARTIFACT)
        assert not report["classes"]


class TestPredictingAContinuousValue:
    def test_a_prediction_is_a_number_with_no_probabilities(self, completed_regression_job):
        """Probabilities are a classification concept and must not be invented."""
        client = TestClient(app)
        response = client.post(
            f"/jobs/{completed_regression_job}/predict",
            json={"rows": [{"area": 1200, "rooms": 3, "distance_km": 1.5, "district": "central"}]},
        )
        assert response.status_code == 200, response.text

        row = response.json()["predictions"][0]
        assert isinstance(row["prediction"], float)
        assert row["probabilities"] == {}
        # The generative relationship puts a flat of this size in this range;
        # a prediction outside it means the pipeline is not serving the model
        # it claims to be.
        assert 2_000 < row["prediction"] < 5_000
