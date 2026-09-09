"""A run that dies leaves a job that says so, and no file behind.

Job 6589 was OOM-killed while writing its cleaned dataset and sat at RUNNING
afterwards, with no error, because a job's status is written from inside the
task and the task had stopped existing.

Three things are tested here, and they fail in different ways:

* the cleaned CSV is streamed through a file rather than built in memory, and
  that file is removed whether the upload works or not;
* a task whose process dies marks its job failed;
* a worker starting up fails the jobs nobody is running -- and, the part worth
  being careful about, leaves alone the ones somebody is.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from app.core.db import SessionLocal, database_healthy
from app.models.job import Job, JobStatus
from app.models.user import DEV_USER_ID
from app.worker import tasks
from app.worker.progress import fail_job, recover_orphaned_jobs, running_job_ids

pytestmark = pytest.mark.skipif(not database_healthy(), reason="needs Postgres (`make up`)")


def _job(status: JobStatus) -> int:
    with SessionLocal() as db:
        job = Job(
            user_id=DEV_USER_ID,
            original_filename="oom.csv",
            s3_key=f"test/oom/{np.random.default_rng().integers(1 << 62)}",
            size_bytes=1,
            status=status,
        )
        db.add(job)
        db.commit()
        return job.id


def _status(job_id: int) -> tuple[JobStatus, str | None]:
    with SessionLocal() as db:
        job = db.get(Job, job_id)
        return job.status, job.error_message


class TestADeadTaskFailsItsJob:
    def test_a_lost_worker_marks_the_job_failed(self):
        job_id = _job(JobStatus.RUNNING)

        class WorkerLostError(Exception):
            pass

        tasks._fail_job_when_the_task_dies(
            task_id="t-1", exception=WorkerLostError("child exited"), args=[job_id]
        )
        status, error = _status(job_id)
        assert status is JobStatus.FAILED
        assert "out of memory" in error

    def test_an_ordinary_exception_records_itself_instead(self):
        """Not every death is the OOM killer; the message should not claim it is."""
        job_id = _job(JobStatus.RUNNING)
        tasks._fail_job_when_the_task_dies(
            task_id="t-2", exception=ValueError("bad column"), args=[job_id]
        )
        status, error = _status(job_id)
        assert status is JobStatus.FAILED
        assert "bad column" in error
        assert "out of memory" not in error

    def test_a_job_that_already_finished_is_not_dragged_backwards(self):
        """A late signal must not overwrite a completed run."""
        job_id = _job(JobStatus.COMPLETED)
        assert fail_job(job_id, "too late") is False
        assert _status(job_id)[0] is JobStatus.COMPLETED


class TestTheStartupSweepOnlyTouchesOrphans:
    def test_a_running_job_nobody_is_executing_is_failed(self):
        # Every *other* running job is declared active, so this sweep can only
        # touch the one this test made. An empty active set is a real argument
        # to this function, but passing one here would fail every unrelated job
        # in the database -- which is exactly what the first version of this
        # test did, to real jobs, on the shared development Postgres.
        job_id = _job(JobStatus.RUNNING)
        others = running_job_ids() - {job_id}
        recovered = recover_orphaned_jobs(active_job_ids=others, reason="lost")
        assert recovered == [job_id]
        assert _status(job_id)[0] is JobStatus.FAILED

    def test_a_running_job_a_worker_owns_is_left_alone(self):
        """The safety property: a new worker must not kill its neighbour's job."""
        mine = _job(JobStatus.RUNNING)
        theirs = _job(JobStatus.RUNNING)
        others = (running_job_ids() - {mine}) | {theirs}
        recovered = recover_orphaned_jobs(active_job_ids=others, reason="lost")
        assert theirs not in recovered
        assert _status(theirs)[0] is JobStatus.RUNNING
        assert recovered == [mine]

    def test_an_unknown_answer_touches_nothing(self):
        """None is not an empty set, and treating it as one fails live jobs."""
        job_id = _job(JobStatus.RUNNING)
        with patch.object(tasks, "active_pipeline_jobs", return_value=None):
            with patch.object(tasks, "recover_orphaned_jobs") as sweep:
                tasks._recover_orphans_on_startup(sender=None)
        sweep.assert_not_called()
        assert _status(job_id)[0] is JobStatus.RUNNING

    def test_a_definite_empty_answer_does_sweep(self):
        """The contrast: workers replied and are running nothing."""
        job_id = _job(JobStatus.RUNNING)
        others = running_job_ids() - {job_id}
        with patch.object(tasks, "active_pipeline_jobs", return_value=others):
            tasks._recover_orphans_on_startup(sender=None)
        assert _status(job_id)[0] is JobStatus.FAILED


class TestActiveJobsAreReadFromTheWorkers:
    def test_job_ids_come_from_the_running_tasks(self):
        replies = {
            "worker@a": [{"name": tasks.TASK_NAME, "args": [11]}],
            "worker@b": [{"name": tasks.TASK_NAME, "args": [22]}, {"name": "other", "args": [33]}],
        }
        with patch.object(tasks.celery_app.control, "inspect") as inspect:
            inspect.return_value.active.return_value = replies
            assert tasks.active_pipeline_jobs() == {11, 22}

    def test_no_reply_is_unknown_not_empty(self):
        with patch.object(tasks.celery_app.control, "inspect") as inspect:
            inspect.return_value.active.return_value = None
            assert tasks.active_pipeline_jobs() is None

    def test_a_broker_failure_is_unknown_not_empty(self):
        with patch.object(tasks.celery_app.control, "inspect", side_effect=OSError("no broker")):
            assert tasks.active_pipeline_jobs() is None


class TestTheCleanedCsvIsStreamedAndTidiedUp:
    """Written through a file, and the file does not survive the node."""

    @staticmethod
    def _frame() -> pd.DataFrame:
        return pd.DataFrame({"a": np.arange(50.0), "b": np.arange(50.0)})

    def test_the_temporary_file_is_removed_after_success(self, tmp_path, monkeypatch):
        seen: list[Path] = []
        monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
        with patch("app.worker.graph.register_file_artifact") as register:
            register.side_effect = lambda *a, **k: seen.append(Path(a[3]))
            self._run_cleaning(monkeypatch)
        assert seen, "the artifact was never registered"
        assert not seen[0].exists(), "the temporary CSV outlived the node"
        assert list(tmp_path.glob("autods-cleaned-*")) == []

    def test_the_temporary_file_is_removed_after_a_failed_upload(self, tmp_path, monkeypatch):
        """The finally block is the point: storage failing must not leak a file."""
        monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
        with patch("app.worker.graph.register_file_artifact", side_effect=OSError("storage down")):
            with pytest.raises(OSError, match="storage down"):
                self._run_cleaning(monkeypatch)
        assert list(tmp_path.glob("autods-cleaned-*")) == []

    def _run_cleaning(self, monkeypatch):
        """Drive cleaning_node with everything but the file handling stubbed."""
        from app.ml.cleaning import CleaningResult
        from app.ml.contracts import CleaningReport
        from app.worker import graph

        report = CleaningReport(
            n_rows_before=50, n_rows_after=50, n_columns_before=2, n_columns_after=2
        )
        monkeypatch.setattr(
            graph, "clean_frame", lambda *a, **k: CleaningResult(frame=self._frame(), report=report)
        )
        monkeypatch.setattr(graph, "register_json_artifact", lambda *a, **k: None)
        return graph.cleaning_node(
            {
                "job_id": 1,
                "frame": self._frame(),
                "schema": None,
                "target": "b",
                "task_type": "regression",
                "excluded": [],
                "plan": None,
            }
        )
