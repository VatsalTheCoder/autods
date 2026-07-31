"""The pipeline end to end on a dataset too small to run the folds it asked for.

Every run so far has had enough of every class to do what the settings said:
five stratified folds, SMOTE if the plan wanted it. This one does not. With four
members of the rarer class it trips two minimums at once -- stratification drops
to four folds, and SMOTE declines because four examples are too few to
interpolate between -- and the question these tests ask is not whether the guards
fire, which unit tests already cover, but whether **the person reading the report
is told they fired**.

That distinction is the whole point. ``modeling.py`` says of the fold reduction:
"Every reduction is recorded as a warning so the report never implies five folds
it did not run." The warning was being recorded, correctly, onto the leaderboard
-- and the report is built from the evaluation artifact, which never received it.
So the report said "the dataset was split into 4 folds" in the flat tone of
something that had always been the plan, and the sentence explaining why sat
unread in a different file. A live sweep found it; these tests hold it shut.

The numbers are chosen, not sampled: 150 rows and exactly 4 positives. Sampling
the minority count would make which guards fire a property of the seed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.api.main import app
from app.core.db import SessionLocal, database_healthy
from app.core.storage import storage_healthy
from app.services.artifacts import (
    EVALUATION_ARTIFACT,
    LEADERBOARD_ARTIFACT,
    REPORT_ARTIFACT,
    load_artifact_bytes,
    load_json_artifact,
)
from app.worker.pipeline import run_pipeline

pytestmark = pytest.mark.skipif(
    not (database_healthy() and storage_healthy()),
    reason="needs Postgres and object storage (start with `make up`)",
)

N_ROWS = 150
N_POSITIVE = 4


def dataset_csv() -> bytes:
    """A pilot study with four responders among 150 participants.

    The responders are the four highest rows on a latent index rather than four
    random ones, so the class is learnable in principle. The test is what the
    system says about scoring it from four examples, not whether it can.
    """
    rng = np.random.default_rng(4)

    age = rng.normal(58, 12, N_ROWS).clip(19, 92).round(0)
    baseline_score = rng.normal(42, 9, N_ROWS).round(1)
    dose_mg = rng.choice([5.0, 10.0, 20.0], size=N_ROWS, p=[0.4, 0.4, 0.2])
    cohort = rng.choice(["A", "B", "C"], size=N_ROWS, p=[0.45, 0.35, 0.20])

    index = (
        0.9 * (dose_mg - dose_mg.mean()) / dose_mg.std()
        + 0.6 * (baseline_score - baseline_score.mean()) / baseline_score.std()
        - 0.4 * (age - age.mean()) / age.std()
        + rng.normal(0, 0.5, N_ROWS)
    )
    responded = np.zeros(N_ROWS, dtype=int)
    responded[np.argsort(index)[-N_POSITIVE:]] = 1

    return (
        pd.DataFrame(
            {
                "cohort": cohort,
                "age": age,
                "baseline_score": baseline_score,
                "dose_mg": dose_mg,
                "responded": responded,
            }
        )
        .to_csv(index=False)
        .encode("utf-8")
    )


@pytest.fixture(autouse=True)
def _no_celery_dispatch(monkeypatch):
    monkeypatch.setattr("app.api.routes.upload.enqueue_pipeline", lambda job_id: None)


@pytest.fixture
def completed_small_job() -> int:
    client = TestClient(app)
    upload = client.post("/upload", files={"file": ("pilot.csv", dataset_csv(), "text/csv")})
    job_id = upload.json()["job_id"]

    confirm = client.post(
        "/jobs",
        json={"job_id": job_id, "target_column": "responded", "task_type": "classification"},
    )
    assert confirm.status_code == 200, confirm.text

    run_pipeline(job_id)
    return job_id


def _evaluation(job_id: int) -> dict:
    with SessionLocal() as db:
        return load_json_artifact(db, job_id, EVALUATION_ARTIFACT)


def _report(job_id: int) -> str:
    with SessionLocal() as db:
        return load_artifact_bytes(db, job_id, REPORT_ARTIFACT).decode("utf-8")


class TestTheGuardsFired:
    """Preconditions. If these fail the rest of the file is testing nothing."""

    def test_the_fold_count_came_down(self, completed_small_job):
        assert _evaluation(completed_small_job)["n_folds"] == N_POSITIVE

    def test_it_still_stratified_what_it_could(self, completed_small_job):
        assert _evaluation(completed_small_job)["cv_strategy"] == "StratifiedKFold"

    def test_the_run_completed_anyway(self, completed_small_job):
        """A dataset this thin is a reason to caveat the score, not to refuse it."""
        assert _evaluation(completed_small_job)["metrics"]


class TestTheReasonReachesTheEvaluationArtifact:
    def test_the_reduction_is_stated(self, completed_small_job):
        warnings = " ".join(_evaluation(completed_small_job)["warnings"])
        assert "Reduced to 4 folds" in warnings
        assert "rarest class has only 4 rows" in warnings

    def test_the_skipped_resampling_is_stated(self, completed_small_job):
        warnings = " ".join(_evaluation(completed_small_job)["warnings"])
        assert "SMOTE" in warnings

    def test_nothing_is_said_twice(self, completed_small_job):
        """The leaderboard absorbs each candidate's fold warnings as it scores
        them, so the winner's list is a subset of the leaderboard's. Merging the
        two without dedup would print the per-fold notes twice."""
        warnings = _evaluation(completed_small_job)["warnings"]
        assert len(warnings) == len(set(warnings))

    def test_it_carries_what_the_leaderboard_carries(self, completed_small_job):
        """The leaderboard is where these decisions are made and recorded. The
        evaluation report should not be a lossy copy of it."""
        with SessionLocal() as db:
            leaderboard = load_json_artifact(db, completed_small_job, LEADERBOARD_ARTIFACT)
        missing = set(leaderboard["warnings"]) - set(_evaluation(completed_small_job)["warnings"])
        assert not missing, f"warnings stranded on the leaderboard: {missing}"


class TestTheReasonReachesThePersonReadingIt:
    """The artifact is plumbing. This is the claim that actually matters."""

    def test_the_report_explains_the_fold_count(self, completed_small_job):
        assert "Reduced to 4 folds" in _report(completed_small_job)

    def test_the_report_does_not_present_four_folds_as_the_plan(self, completed_small_job):
        """The regression this file exists for. The report described the split
        without ever saying it was forced -- true sentence, misleading document.

        The explanation has to be the *fold* one specifically. An earlier draft
        of this test asserted only that "rarest class has only 4 rows" appeared
        somewhere in the report, and it passed against the unfixed code: the
        skipped-SMOTE warning contains that same phrase and reaches the report by
        another route entirely. The four-row minority was explained; the four
        folds were not.
        """
        report = _report(completed_small_job)
        assert "split into 4 folds" in report
        assert "Reduced to 4 folds" in report
        assert "too few for 5-fold stratification" in report

    def test_the_caveats_section_is_where_it_lands(self, completed_small_job):
        report = _report(completed_small_job)
        assert "## Caveats" in report
        caveats = report.split("## Caveats", 1)[1]
        assert "Reduced to 4 folds" in caveats
