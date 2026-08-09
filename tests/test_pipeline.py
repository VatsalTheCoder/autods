"""Integration tests for the background pipeline -- the Section 5 "Done when".

These run the real vertical slice against Postgres and object storage: a CSV is
uploaded through the API, its schema is confirmed, the pipeline runs every node
for real, and what comes out is a cross-validated score and a readable report with
every artifact in S3 and every status in Postgres. Skipped when the stack is down.

Nothing is stubbed except the Celery dispatch. Confirming a schema normally hands
the job to the worker, and the local worker is running, so it would race these
tests for the same job -- the tests call ``run_pipeline`` directly instead, which
is the same function the Celery task wraps.

No LLM is configured in the test environment, so the planner degrades to its
default plan. That is deliberate: the whole slice has to work without a key.
"""

from __future__ import annotations

import io
import json
import uuid

import joblib
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted

from app.api.main import app
from app.core.db import SessionLocal, database_healthy
from app.core.storage import storage_healthy
from app.models.agent_run import AgentRunStatus
from app.models.job import Job, JobStatus
from app.models.user import DEV_USER_ID
from app.services.artifacts import (
    CLEANED_DATASET_ARTIFACT,
    CLEANING_ARTIFACT,
    CLUSTERING_ARTIFACT,
    CRITIC_ARTIFACT,
    EDA_ARTIFACT,
    EVALUATION_ARTIFACT,
    EXPLAINABILITY_ARTIFACT,
    FINAL_MODEL_ARTIFACT,
    FINAL_MODEL_INFO_ARTIFACT,
    NARRATIVE_ARTIFACT,
    PLANNER_ARTIFACT,
    PREPROCESSING_ARTIFACT,
    PREPROCESSOR_ARTIFACT,
    REPORT_ARTIFACT,
    REPORT_PDF_ARTIFACT,
    load_artifact_bytes,
    load_json_artifact,
)
from app.worker.pipeline import run_pipeline
from app.worker.state import PIPELINE_NODES

pytestmark = [
    # Slow by nature: see the `e2e` marker in pyproject.toml. CI runs these on
    # a push to main, not on every pull request.
    pytest.mark.e2e,
    pytest.mark.skipif(
        not (database_healthy() and storage_healthy()),
        reason="needs Postgres and object storage (start with `make up`)",
    ),
]

N_ROWS = 60


def dataset_csv() -> bytes:
    """A dataset big enough for five folds, with a gap and a duplicate in it.

    Deliberately imperfect: one missing value (so the fold-fitted imputer has
    something to do and the report has something to explain) and a repeated row
    (so cleaning has something to remove).
    """
    frame = pd.DataFrame(
        {
            "age": [22 + (i % 40) for i in range(N_ROWS)],
            "city": ["London", "Leeds", "Bristol"] * (N_ROWS // 3),
            "income": [30000 + i * 500 for i in range(N_ROWS)],
            "churn": ["yes", "no"] * (N_ROWS // 2),
        }
    )
    frame.loc[3, "age"] = None
    frame = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    return frame.to_csv(index=False).encode("utf-8")


@pytest.fixture(autouse=True)
def _no_celery_dispatch(monkeypatch):
    """Confirming must not also hand the job to the live worker (see module docstring)."""
    monkeypatch.setattr("app.api.routes.upload.enqueue_pipeline", lambda job_id: None)


@pytest.fixture
def confirmed_job() -> int:
    """A job uploaded and confirmed through the real API, ready to run."""
    client = TestClient(app)
    upload = client.post("/upload", files={"file": ("data.csv", dataset_csv(), "text/csv")})
    job_id = upload.json()["job_id"]

    confirm = client.post(
        "/jobs",
        json={"job_id": job_id, "target_column": "churn", "task_type": "classification"},
    )
    assert confirm.status_code == 200
    return job_id


@pytest.fixture
def completed_job(confirmed_job) -> int:
    run_pipeline(confirmed_job)
    return confirmed_job


class TestSuccessfulRun:
    def test_job_reaches_completed(self, completed_job):
        with SessionLocal() as db:
            job = db.get(Job, completed_job)
            assert job.status is JobStatus.COMPLETED, job.error_message

    def test_every_node_reaches_a_terminal_state_in_order(self, completed_job):
        """Every node ends either completed or deliberately skipped -- none pending.

        The optional Section 7 steps are off in this fixture (a 60-row balanced
        dataset needs neither sampling nor feature selection), so the run
        legitimately routes past them. What must hold for *all* of them is that
        the roadmap is the full node list and nothing was left unreached.
        """
        with SessionLocal() as db:
            runs = db.get(Job, completed_job).agent_runs
            assert [r.name for r in runs] == PIPELINE_NODES
            assert all(r.status in (AgentRunStatus.COMPLETED, AgentRunStatus.SKIPPED) for r in runs)
            assert all(r.finished_at is not None for r in runs)

    def test_the_optional_steps_are_visibly_skipped_not_missing(self, completed_job):
        """Section 7's dynamic-orchestration claim, checked on a real run.

        A node the graph routed past has to be distinguishable from one the run
        never got to -- otherwise "the pipeline adapted to this dataset" is not
        an observable claim. It keeps its row, gets SKIPPED, and says why.
        """
        with SessionLocal() as db:
            runs = {r.name: r for r in db.get(Job, completed_job).agent_runs}

        for name in ("sampling", "feature_selection"):
            assert runs[name].status is AgentRunStatus.SKIPPED, name
            assert runs[name].error_message, f"{name} was skipped without saying why"

    def test_the_required_steps_all_completed(self, completed_job):
        with SessionLocal() as db:
            runs = {r.name: r for r in db.get(Job, completed_job).agent_runs}

        for name in (
            "planner",
            "cleaning",
            "eda",
            "preprocessing",
            "modeling",
            "final_training",
            "explainability",
            "critic",
            "report",
        ):
            assert runs[name].status is AgentRunStatus.COMPLETED, name
            assert runs[name].started_at is not None

    def test_rerun_resets_the_roadmap(self, completed_job):
        """A second run must not duplicate rows or leave stale statuses."""
        run_pipeline(completed_job)
        with SessionLocal() as db:
            runs = db.get(Job, completed_job).agent_runs
            assert len(runs) == len(PIPELINE_NODES)


class TestArtifacts:
    """Every stage's output must be in S3 with a row in Postgres (spec 13)."""

    @pytest.mark.parametrize(
        "name",
        [
            PLANNER_ARTIFACT,
            CLEANING_ARTIFACT,
            EDA_ARTIFACT,
            CLUSTERING_ARTIFACT,
            PREPROCESSING_ARTIFACT,
            EVALUATION_ARTIFACT,
            FINAL_MODEL_INFO_ARTIFACT,
            EXPLAINABILITY_ARTIFACT,
            CRITIC_ARTIFACT,
            NARRATIVE_ARTIFACT,
        ],
    )
    def test_json_artifacts_are_registered_and_readable(self, completed_job, name):
        with SessionLocal() as db:
            assert load_json_artifact(db, completed_job, name) is not None

    def test_charts_are_stored_as_real_pngs(self, completed_job):
        """Section 6's output has to survive the round trip through S3."""
        with SessionLocal() as db:
            eda = load_json_artifact(db, completed_job, EDA_ARTIFACT)
            assert eda["plots"], "no charts were produced"
            for name in eda["plots"]:
                data = load_artifact_bytes(db, completed_job, name)
                assert data.startswith(b"\x89PNG"), name

    def test_the_cleaned_dataset_has_no_cluster_column(self, completed_job):
        """The guardrail, checked on what a real run actually stored (spec 9)."""
        with SessionLocal() as db:
            data = load_artifact_bytes(db, completed_job, CLEANED_DATASET_ARTIFACT)
        columns = pd.read_csv(io.BytesIO(data)).columns
        assert not [c for c in columns if "cluster" in c.lower()]

    def test_the_cleaned_dataset_is_stored(self, completed_job):
        with SessionLocal() as db:
            data = load_artifact_bytes(db, completed_job, CLEANED_DATASET_ARTIFACT)
        frame = pd.read_csv(io.BytesIO(data))
        # The duplicate row went; the deliberate gap did not.
        assert len(frame) == N_ROWS
        assert frame["age"].isna().sum() == 1

    def test_the_report_is_stored_as_markdown(self, completed_job):
        with SessionLocal() as db:
            data = load_artifact_bytes(db, completed_job, REPORT_ARTIFACT)
        assert data.decode("utf-8").startswith("# Analysis of")

    def test_the_stored_pipeline_is_unfitted(self, completed_job):
        """The leakage claim, checked against the artifact a real run produced.

        ``test_leakage.py`` proves this on synthetic data; this proves the pickle
        the actual pipeline wrote to object storage is unfitted too.
        """
        with SessionLocal() as db:
            data = load_artifact_bytes(db, completed_job, PREPROCESSOR_ARTIFACT)
        restored = joblib.load(io.BytesIO(data))
        with pytest.raises(NotFittedError):
            check_is_fitted(restored)

    def test_the_final_model_is_fitted_and_usable(self, completed_job):
        """The counterpart to the test above, and the one exception to it.

        The recipe in S3 must be unfitted; the *final model* in S3 must be
        fitted, on every row, and must predict when loaded back into a fresh
        process. Both artifacts exist precisely so that each claim is checkable
        rather than asserted (spec 7.9).
        """
        with SessionLocal() as db:
            data = load_artifact_bytes(db, completed_job, FINAL_MODEL_ARTIFACT)
            info = load_json_artifact(db, completed_job, FINAL_MODEL_INFO_ARTIFACT)

        restored = joblib.load(io.BytesIO(data))
        check_is_fitted(restored.named_steps["model"])
        assert info["n_rows"] == N_ROWS

        row = pd.DataFrame([{"age": 41, "city": "London", "income": 42000}])
        assert restored.predict(row)[0] in ("yes", "no")

    def test_the_explanation_is_in_the_users_column_names(self, completed_job):
        """Section 8's whole point: no reader should meet ``city_London``."""
        with SessionLocal() as db:
            payload = load_json_artifact(db, completed_job, EXPLAINABILITY_ARTIFACT)

        assert payload["global_importance"]
        named = {item["feature"] for item in payload["global_importance"]}
        assert named <= {"age", "city", "income"}
        # And the encoded names it was translated from are published alongside.
        assert "city_London" in payload["feature_name_mapping"]
        assert payload["feature_name_mapping"]["city_London"] == "city"

    def test_the_report_is_also_rendered_as_a_pdf(self, completed_job):
        """Section 9's "Done when": a multi-section document, not a text file."""
        with SessionLocal() as db:
            data = load_artifact_bytes(db, completed_job, REPORT_PDF_ARTIFACT)
        assert data is not None, "no PDF was produced"
        assert data.startswith(b"%PDF-")
        assert len(data) > 5_000, "a multi-section report should not be this small"

    def test_the_review_reaches_the_report(self, completed_job):
        """The critic exists to be read, so it has to appear in what is read."""
        with SessionLocal() as db:
            critic = load_json_artifact(db, completed_job, CRITIC_ARTIFACT)
            markdown = load_artifact_bytes(db, completed_job, REPORT_ARTIFACT).decode("utf-8")

        assert critic["verdict"], "the review reached no verdict"
        assert "## What the review found" in markdown

    def test_the_narrative_is_stored_apart_from_the_report(self, completed_job):
        """So a reader can tell which sentences a model wrote (spec 7.12)."""
        with SessionLocal() as db:
            narrative = load_json_artifact(db, completed_job, NARRATIVE_ARTIFACT)
        assert narrative is not None
        # No key in the test environment, so the prose is absent and the
        # deterministic report stands -- which is the behaviour being asserted.
        assert narrative["source"] == "default"

    def test_the_report_explains_the_model_and_names_what_is_served(self, completed_job):
        with SessionLocal() as db:
            markdown = load_artifact_bytes(db, completed_job, REPORT_ARTIFACT).decode("utf-8")
        assert "## Why the model predicts what it does" in markdown
        assert "## The model that gets served" in markdown

    def test_the_evaluation_artifact_holds_real_cross_validated_scores(self, completed_job):
        with SessionLocal() as db:
            payload = load_json_artifact(db, completed_job, EVALUATION_ARTIFACT)
        assert payload["n_folds"] == 5
        assert len(payload["folds"]) == 5
        assert 0.0 <= payload["metrics"]["f1_macro"]["mean"] <= 1.0
        # 60 rows, 5 folds: each fold trains on 48 and is scored on 12.
        assert [f["n_train"] for f in payload["folds"]] == [48] * 5
        assert sum(f["n_test"] for f in payload["folds"]) == N_ROWS

    def test_the_cleaning_artifact_records_the_gap_it_left(self, completed_job):
        with SessionLocal() as db:
            payload = load_json_artifact(db, completed_job, CLEANING_ARTIFACT)
        assert payload["duplicate_rows_removed"] == 1
        assert payload["missing_values_left_to_the_pipeline"] == {"age": 1}

    def test_the_planner_fell_back_to_defaults_without_a_key(self, completed_job):
        with SessionLocal() as db:
            payload = load_json_artifact(db, completed_job, PLANNER_ARTIFACT)
        assert payload["source"] == "default"


class TestFailurePath:
    def test_a_failing_node_fails_the_job_and_names_itself(self, confirmed_job, monkeypatch):
        def boom(*_args, **_kwargs):
            raise RuntimeError("node exploded")

        monkeypatch.setattr("app.worker.graph.make_plan", boom)

        run_pipeline(confirmed_job)

        with SessionLocal() as db:
            job = db.get(Job, confirmed_job)
            assert job.status is JobStatus.FAILED
            assert "node exploded" in (job.error_message or "")

            runs = {r.name: r for r in job.agent_runs}
            # The first node failed; later nodes were never reached.
            assert runs["planner"].status is AgentRunStatus.FAILED
            assert runs["cleaning"].status is AgentRunStatus.PENDING

    def test_a_later_failure_keeps_the_earlier_artifacts(self, confirmed_job, monkeypatch):
        """Per-node commits: a failed run is still inspectable up to where it broke."""

        def boom(*_args, **_kwargs):
            raise RuntimeError("modelling exploded")

        monkeypatch.setattr("app.worker.graph.run_leaderboard", boom)

        run_pipeline(confirmed_job)

        with SessionLocal() as db:
            job = db.get(Job, confirmed_job)
            assert job.status is JobStatus.FAILED
            assert {r.name for r in job.agent_runs if r.status is AgentRunStatus.COMPLETED} == {
                "planner",
                "cleaning",
                "eda",
                "feature_strategy",
                "preprocessing",
            }
            assert load_json_artifact(db, confirmed_job, CLEANING_ARTIFACT) is not None
            assert load_json_artifact(db, confirmed_job, EVALUATION_ARTIFACT) is None

    def test_an_unusable_dataset_fails_with_a_readable_reason(self, monkeypatch):
        """A dataset cleaning empties out must explain itself, not raise from sklearn."""
        client = TestClient(app)
        # Every feature column is constant, so nothing is left to model with.
        frame = pd.DataFrame({"country": ["UK"] * 20, "churn": ["yes", "no"] * 10})
        upload = client.post(
            "/upload",
            files={"file": ("flat.csv", frame.to_csv(index=False).encode(), "text/csv")},
        )
        job_id = upload.json()["job_id"]
        client.post(
            "/jobs",
            json={"job_id": job_id, "target_column": "churn", "task_type": "classification"},
        )

        run_pipeline(job_id)

        with SessionLocal() as db:
            job = db.get(Job, job_id)
            assert job.status is JobStatus.FAILED
            assert "No usable feature columns" in (job.error_message or "")

    def test_an_unconfirmed_job_fails_before_any_node_runs(self):
        """The pipeline needs a confirmed target; it must say so, not crash later."""
        client = TestClient(app)
        upload = client.post("/upload", files={"file": ("data.csv", dataset_csv(), "text/csv")})
        job_id = upload.json()["job_id"]

        run_pipeline(job_id)  # never confirmed

        with SessionLocal() as db:
            job = db.get(Job, job_id)
            assert job.status is JobStatus.FAILED
            assert "never confirmed" in (job.error_message or "")

    def test_missing_dataset_fails_gracefully(self):
        """A job whose CSV is absent from storage must fail, not hang."""
        with SessionLocal() as db:
            job = Job(
                user_id=DEV_USER_ID,
                original_filename="ghost.csv",
                s3_key=f"jobs/missing/{uuid.uuid4()}.csv",
                size_bytes=1,
                status=JobStatus.QUEUED,
                target_column="churn",
                task_type="classification",
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


class TestVisibleViaAPI:
    """The whole point of the slice: the browser can see all of this."""

    def test_get_job_exposes_node_statuses(self, completed_job):
        body = TestClient(app).get(f"/jobs/{completed_job}").json()
        assert body["status"] == "completed"
        assert [r["name"] for r in body["agent_runs"]] == PIPELINE_NODES
        # "skipped" reaches the browser as its own status, which is what lets the
        # Progress page draw it differently from a step still waiting to run.
        assert all(r["status"] in ("completed", "skipped") for r in body["agent_runs"])
        assert {r["status"] for r in body["agent_runs"]} == {"completed", "skipped"}

    def test_the_evaluation_endpoint_serves_the_metrics(self, completed_job):
        body = TestClient(app).get(f"/jobs/{completed_job}/evaluation").json()
        assert body["target_column"] == "churn"
        # Whichever of the four candidates won -- the roster is the contract
        # here, not any particular model beating the others on this fixture.
        assert body["model_name"] in {
            "RandomForest",
            "LogisticRegression",
            "XGBoost",
            "LightGBM",
        }
        assert body["primary_metric"] == "f1_macro"
        assert len(body["folds"]) == 5

    def test_the_leaderboard_endpoint_serves_the_whole_roster(self, completed_job):
        body = TestClient(app).get(f"/jobs/{completed_job}/leaderboard").json()
        assert {e["model_name"] for e in body["entries"]} == {
            "RandomForest",
            "LogisticRegression",
            "XGBoost",
            "LightGBM",
            "Baseline (most frequent class)",
        }
        assert [e["rank"] for e in body["entries"]] == [1, 2, 3, 4, 5]

    def test_the_baseline_reaches_the_api_labelled(self, completed_job):
        """A client cannot tell the reference row from a contender without this."""
        body = TestClient(app).get(f"/jobs/{completed_job}/leaderboard").json()
        flagged = [e for e in body["entries"] if e["is_baseline"]]
        assert [e["model_name"] for e in flagged] == ["Baseline (most frequent class)"]

    def test_the_winner_is_the_model_the_evaluation_describes(self, completed_job):
        """Two endpoints, one run -- they must not disagree about who won."""
        client = TestClient(app)
        board = client.get(f"/jobs/{completed_job}/leaderboard").json()
        evaluation = client.get(f"/jobs/{completed_job}/evaluation").json()

        assert board["entries"][0]["model_name"] == evaluation["model_name"]

    def test_the_feature_strategy_endpoint_serves_the_decision(self, completed_job):
        """The decision, served apart from the recipe built out of it (spec 7.6)."""
        body = TestClient(app).get(f"/jobs/{completed_job}/features").json()
        assert {c["column"] for c in body["columns"]} == {"age", "city", "income"}
        # No LLM in the test environment, so this must not claim one chose.
        assert body["source"] == "hardcoded"
        assert body["rejected_columns"] == []

    def test_the_report_endpoint_serves_readable_markdown(self, completed_job):
        body = TestClient(app).get(f"/jobs/{completed_job}/report").json()
        assert body["job_id"] == completed_job
        assert "## How this was validated" in body["markdown"]
        assert "training rows only" in body["markdown"]

    def test_the_cleaning_endpoint_serves_the_cleaning_report(self, completed_job):
        body = TestClient(app).get(f"/jobs/{completed_job}/cleaning").json()
        assert body["duplicate_rows_removed"] == 1

    def test_the_eda_endpoint_serves_the_statistics_and_chart_index(self, completed_job):
        body = TestClient(app).get(f"/jobs/{completed_job}/eda").json()
        assert body["target_column"] == "churn"
        assert {c["name"] for c in body["columns"]}
        assert "target_distribution.png" in body["plots"]

    def test_the_clustering_endpoint_serves_the_groups(self, completed_job):
        body = TestClient(app).get(f"/jobs/{completed_job}/clustering").json()
        assert body["method"] in ("kmeans", "kprototypes")
        assert len(body["profiles"]) == body["k"]

    def test_a_chart_can_be_fetched_as_a_png(self, completed_job):
        """End of the chain: the browser can actually display an EDA chart."""
        resp = TestClient(app).get(
            f"/jobs/{completed_job}/artifacts/target_distribution.png/content"
        )
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "image/png"
        assert resp.content.startswith(b"\x89PNG")

    def test_artifacts_are_listed_for_the_results_page(self, completed_job):
        body = TestClient(app).get(f"/jobs/{completed_job}/artifacts").json()
        names = {a["name"] for a in body}
        assert {
            "dataset.csv",
            CLEANED_DATASET_ARTIFACT,
            EVALUATION_ARTIFACT,
            REPORT_ARTIFACT,
            PREPROCESSOR_ARTIFACT,
        } <= names


class TestArtifactContentEndpoint:
    """Serving bytes through the API -- what makes charts displayable (Section 6).

    A presigned URL cannot be followed from a browser against local MinIO, so the
    Results page reads artifacts through here instead. These tests cover the three
    shapes it has to handle: text, tabular and binary.
    """

    def test_markdown_is_served_with_its_own_content_type(self, completed_job):
        resp = TestClient(app).get(f"/jobs/{completed_job}/artifacts/{REPORT_ARTIFACT}/content")
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/markdown")
        assert resp.text.startswith("# Analysis of")

    def test_csv_is_served_as_csv(self, completed_job):
        resp = TestClient(app).get(
            f"/jobs/{completed_job}/artifacts/{CLEANED_DATASET_ARTIFACT}/content"
        )
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/csv")
        assert len(pd.read_csv(io.BytesIO(resp.content))) == N_ROWS

    def test_binary_survives_the_round_trip(self, completed_job):
        """A pickle served through the API must still load -- no text mangling."""
        resp = TestClient(app).get(
            f"/jobs/{completed_job}/artifacts/{PREPROCESSOR_ARTIFACT}/content"
        )
        assert resp.status_code == 200
        restored = joblib.load(io.BytesIO(resp.content))
        with pytest.raises(NotFittedError):
            check_is_fitted(restored)

    def test_the_artifacts_real_name_is_offered_for_download(self, completed_job):
        """Otherwise a browser saves the file as "content"."""
        resp = TestClient(app).get(f"/jobs/{completed_job}/artifacts/{REPORT_ARTIFACT}/content")
        assert REPORT_ARTIFACT in resp.headers["content-disposition"]

    def test_responses_are_cacheable_but_private(self, completed_job):
        """Streamlit reruns its whole script per interaction; without this every
        chart would be re-fetched on every click."""
        resp = TestClient(app).get(f"/jobs/{completed_job}/artifacts/{REPORT_ARTIFACT}/content")
        assert "private" in resp.headers["cache-control"]

    def test_an_unknown_artifact_is_404(self, completed_job):
        resp = TestClient(app).get(f"/jobs/{completed_job}/artifacts/nope.png/content")
        assert resp.status_code == 404
        assert "no artifact named" in resp.json()["detail"]

    def test_an_unknown_job_is_distinguishable_from_an_unknown_artifact(self):
        resp = TestClient(app).get("/jobs/99999999/artifacts/report.md/content")
        assert resp.status_code == 404
        assert "No job" in resp.json()["detail"]


class TestResultsEndpointsBeforeCompletion:
    def test_evaluation_is_404_until_the_pipeline_produces_it(self, confirmed_job):
        resp = TestClient(app).get(f"/jobs/{confirmed_job}/evaluation")
        assert resp.status_code == 404
        assert "has not produced" in resp.json()["detail"]

    def test_report_is_404_until_the_pipeline_produces_it(self, confirmed_job):
        resp = TestClient(app).get(f"/jobs/{confirmed_job}/report")
        assert resp.status_code == 404

    def test_an_unknown_job_is_404(self):
        resp = TestClient(app).get("/jobs/99999999/evaluation")
        assert resp.status_code == 404
        assert "No job" in resp.json()["detail"]


def test_the_evaluation_artifact_is_valid_json_on_disk(completed_job):
    """Guards against a Pydantic model that serialises to something unparseable."""
    with SessionLocal() as db:
        raw = load_artifact_bytes(db, completed_job, EVALUATION_ARTIFACT)
    assert json.loads(raw)["task_type"] == "classification"
