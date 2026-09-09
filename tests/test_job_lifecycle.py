"""Confirming a job is a one-shot operation, and the cache tracks the model.

Two defects that compound. ``POST /jobs`` accepted a job in any state, reset it
to QUEUED and dispatched a second Celery task; and the prediction cache was
keyed on the artifact's byte size, which does not change when a model of the
same shape is refitted. Together they let a job be run twice over its own
artifacts and then serve the model from before the second run.

The tests here use mocked storage, database and queue. Nothing is enqueued and
no object is written.
"""

from __future__ import annotations

import io
from unittest.mock import patch

import joblib
import numpy as np
import pytest
from sklearn.linear_model import LinearRegression

import app.api.routes.upload as upload_route
import app.services.prediction as prediction
from app.api.schemas import ConfirmJobRequest
from app.ml.contracts import FinalModelInfo
from app.models.job import Job, JobStatus


def _job(status: JobStatus) -> Job:
    return Job(
        id=77,
        user_id=1,
        original_filename="d.csv",
        s3_key="k",
        size_bytes=1,
        status=status,
        n_rows=10,
        n_columns=2,
    )


class _FakeDB:
    def __init__(self, job: Job) -> None:
        self._job = job

    def get(self, _model, _id):
        return self._job

    def commit(self):
        pass

    def refresh(self, _obj):
        pass

    def rollback(self):
        pass


def _confirm(job: Job):
    """Call the route directly with storage and the queue mocked out."""
    request = ConfirmJobRequest(job_id=77, target_column="y", task_type="classification")
    dispatched: list[int] = []
    with (
        patch.object(upload_route, "load_json_artifact", return_value=None),
        patch.object(upload_route, "register_json_artifact"),
        patch.object(upload_route, "enqueue_pipeline", side_effect=dispatched.append),
    ):
        try:
            upload_route.confirm_job(request, db=_FakeDB(job))
        except Exception as exc:  # noqa: BLE001 - the HTTPException is the assertion
            return dispatched, exc
    return dispatched, None


class TestAJobIsConfirmedOnce:
    def test_the_ordinary_path_still_works(self):
        job = _job(JobStatus.UPLOADED)
        dispatched, error = _confirm(job)
        assert error is None
        assert dispatched == [77]
        assert job.status is JobStatus.QUEUED

    @pytest.mark.parametrize(
        "status", [JobStatus.QUEUED, JobStatus.RUNNING, JobStatus.COMPLETED, JobStatus.FAILED]
    )
    def test_a_job_already_underway_is_refused(self, status):
        """The bug: any of these was accepted, reset to QUEUED and dispatched."""
        job = _job(status)
        dispatched, error = _confirm(job)
        assert error is not None and error.status_code == 409
        assert dispatched == [], "a second pipeline task was dispatched"
        assert job.status is status, "the status of a live job was overwritten"

    def test_the_refusal_says_how_to_run_it_again(self):
        """A rerun is a new job, so the message has to say so."""
        _, error = _confirm(_job(JobStatus.COMPLETED))
        assert "new run" in error.detail and "kept" in error.detail

    def test_repeated_confirmation_dispatches_exactly_once(self):
        """The double-click, which is how this happens in practice."""
        job = _job(JobStatus.UPLOADED)
        total: list[int] = []
        for _ in range(5):
            dispatched, _error = _confirm(job)
            total.extend(dispatched)
        assert total == [77]


class TestTheWorkerClaimsTheJob:
    """The API check is time-of-check-to-time-of-use; this is the real guard."""

    def test_only_one_of_two_tasks_for_the_same_job_may_run(self):
        from app.core.db import SessionLocal, database_healthy

        if not database_healthy():
            pytest.skip("needs Postgres (`make up`)")

        from app.models.user import DEV_USER_ID
        from app.worker.progress import claim_job

        with SessionLocal() as db:
            job = Job(
                user_id=DEV_USER_ID,
                original_filename="claim.csv",
                s3_key=f"test/claim/{np.random.default_rng().integers(1 << 62)}",
                size_bytes=1,
                status=JobStatus.QUEUED,
            )
            db.add(job)
            db.commit()
            job_id = job.id

        assert claim_job(job_id) is True, "the first task should win the job"
        assert claim_job(job_id) is False, "a second task claimed a job already running"

    def test_a_job_the_queue_never_produced_is_let_through(self):
        """Losing the claim and never having been queued are different things.

        A job called directly in a state the queue would not produce -- never
        confirmed, say -- must still reach the pipeline's own validation, which
        reports what is wrong with it. Declining here would turn that error into
        a silent no-op.
        """
        from app.core.db import SessionLocal, database_healthy

        if not database_healthy():
            pytest.skip("needs Postgres (`make up`)")

        from app.models.user import DEV_USER_ID
        from app.worker.progress import claim_job

        with SessionLocal() as db:
            job = Job(
                user_id=DEV_USER_ID,
                original_filename="direct.csv",
                s3_key=f"test/direct/{np.random.default_rng().integers(1 << 62)}",
                size_bytes=1,
                status=JobStatus.UPLOADED,
            )
            db.add(job)
            db.commit()
            job_id = job.id

        assert claim_job(job_id) is True

    def test_run_pipeline_stops_when_it_loses_the_claim(self):
        """Winning the claim has to gate the work, not merely be recorded.

        Added after a mutation check: breaking ``run_pipeline`` so it called
        ``claim_job`` and ignored the answer left every other test in this file
        green, because they all exercised ``claim_job`` on its own.
        """
        import app.worker.pipeline as pipeline

        with (
            patch.object(pipeline, "claim_job", return_value=False) as claim,
            patch.object(pipeline, "init_agent_runs") as init,
            patch.object(pipeline, "_load_context") as load_context,
        ):
            pipeline.run_pipeline(77)

        claim.assert_called_once_with(77)
        init.assert_not_called()
        load_context.assert_not_called()

    def test_claiming_a_job_that_does_not_exist_is_declined(self):
        from app.core.db import database_healthy
        from app.worker.progress import claim_job

        if not database_healthy():
            pytest.skip("needs Postgres (`make up`)")
        assert claim_job(999_999_999) is False


class TestTheModelCacheTracksTheStoredModel:
    """Keyed on content, not on length: two models can share a length."""

    @staticmethod
    def _blob(slope: float) -> bytes:
        model = LinearRegression().fit(
            np.array([[0.0], [1.0], [2.0]]), np.array([0.0, slope, 2 * slope])
        )
        buffer = io.BytesIO()
        joblib.dump(model, buffer)
        return buffer.getvalue()

    def _load(self, blob: bytes, version: str):
        info = FinalModelInfo(
            model_name="LinearRegression",
            task_type="regression",
            target_column="y",
            n_rows=3,
            n_features=1,
            feature_columns=[],
        )

        class _Artifact:
            size_bytes = len(blob)
            s3_key = "jobs/77/final_model.pkl"

        class _Result:
            def scalar_one_or_none(self):
                return _Artifact()

        class _DB:
            def execute(self, *_a, **_k):
                return _Result()

        with (
            patch.object(
                prediction, "load_json_artifact", return_value=info.model_dump(mode="json")
            ),
            patch.object(prediction, "object_version", return_value=version),
            patch.object(prediction, "download_bytes", return_value=blob) as download,
        ):
            model, _info = prediction.load_final_model(_DB(), 77)
        return model, download.call_count

    def test_a_replaced_model_of_the_same_length_is_not_served_from_cache(self):
        prediction._MODEL_CACHE.clear()
        first, second = self._blob(1.0), self._blob(10.0)
        assert len(first) == len(second), "fixture no longer exercises the collision"

        model_a, _ = self._load(first, version="etag-before")
        model_b, reads = self._load(second, version="etag-after")

        assert model_a.predict([[5.0]])[0] == pytest.approx(5.0)
        assert model_b.predict([[5.0]])[0] == pytest.approx(50.0), "served the stale model"
        assert reads == 1, "the replacement was never read from storage"

    def test_an_unchanged_model_is_served_from_cache(self):
        """The cache still has to earn its place."""
        prediction._MODEL_CACHE.clear()
        blob = self._blob(1.0)
        self._load(blob, version="etag-same")
        _model, reads = self._load(blob, version="etag-same")
        assert reads == 0, "the cache is no longer caching"

    def test_an_unreadable_version_is_never_cached(self):
        """Unverifiable means load it, not assume the held copy is current."""
        prediction._MODEL_CACHE.clear()
        blob = self._blob(1.0)
        self._load(blob, version=None)
        _model, reads = self._load(blob, version=None)
        assert reads == 1
        assert prediction._MODEL_CACHE == {}
