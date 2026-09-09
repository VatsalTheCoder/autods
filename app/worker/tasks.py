"""Celery tasks, the enqueue helper, and the two ways a dead run gets noticed.

The task is a thin wrapper: all the real logic lives in ``run_pipeline`` so it
can be tested without a broker. ``enqueue_pipeline`` is the single call sites use
to dispatch work, which keeps the API decoupled from Celery specifics and gives
tests one obvious thing to stub.

**A job's status is written from inside the task**, which is fine until the task
stops existing. A worker killed outright -- by the OOM killer, or by its
container going away -- never reaches its own error handling, so the row keeps
saying RUNNING for ever and nothing else looks again. Job 6589 sat like that
after creditcard.csv exhausted the machine.

Two handlers cover two different deaths, and neither covers the other:

* ``task_failure`` fires in the *parent* when a prefork child dies mid-task, so
  the job is failed while the worker itself is still alive. This is the OOM case.
* ``worker_ready`` sweeps at startup, for when the whole worker went down and
  there was no parent left to notice anything.

The sweep asks the live workers what they are running rather than inferring it
from timestamps, because the answer has to be evidence: a worker starting up
beside a busy one must not fail the job its neighbour is halfway through.
"""

from __future__ import annotations

import logging

from celery.signals import task_failure, worker_ready

from app.worker.celery_app import celery_app
from app.worker.pipeline import run_pipeline
from app.worker.progress import fail_job, recover_orphaned_jobs

logger = logging.getLogger(__name__)

TASK_NAME = "autods.run_pipeline"

# What a job's error says when the process running it disappeared. Phrased for
# the person reading it on the Progress page, who wants to know whether to try a
# smaller file rather than what a signal handler is.
LOST_WORKER_REASON = (
    "The worker running this job stopped unexpectedly, most often because it ran "
    "out of memory. The job did not finish. A smaller dataset, or fewer columns, "
    "is the usual way through."
)


@celery_app.task(name=TASK_NAME)
def run_pipeline_task(job_id: int) -> None:
    """Run the full pipeline for a job in the background worker."""
    run_pipeline(job_id)


def active_pipeline_jobs(app=None) -> set[int] | None:
    """Job ids the live workers say they are executing, or None if unknown.

    ``None`` and ``set()`` mean different things and the difference is the whole
    point. An empty set is a definite answer -- the workers replied and are
    running nothing -- and every RUNNING row is therefore orphaned. ``None`` is
    the absence of an answer, and a caller that treated it as "nothing is
    running" would fail every job in flight the moment the broker was slow.
    """
    app = app or celery_app
    try:
        replies = app.control.inspect(timeout=2.0).active()
    except Exception:  # noqa: BLE001 - a broker problem must not crash startup
        logger.exception("Could not ask the workers what they are running")
        return None
    if replies is None:
        logger.warning("No worker replied when asked what it is running")
        return None

    active: set[int] = set()
    for tasks in replies.values():
        for task in tasks or []:
            if task.get("name") != TASK_NAME:
                continue
            args = task.get("args") or []
            if args:
                try:
                    active.add(int(args[0]))
                except (TypeError, ValueError):
                    logger.warning("Unreadable job id in active task args: %r", args)
    return active


@task_failure.connect(sender=run_pipeline_task)
def _fail_job_when_the_task_dies(*, task_id=None, exception=None, args=None, **_kwargs) -> None:
    """Mark the job failed when its task does not survive.

    Covers the OOM kill of a prefork child: the parent notices the child is gone
    and raises WorkerLostError here, which is the only moment anything in this
    process still knows which job it was.
    """
    if not args:
        logger.error("run_pipeline task %s failed with no job id in its args", task_id)
        return
    job_id = int(args[0])
    reason = LOST_WORKER_REASON if _is_lost_worker(exception) else f"The run failed: {exception}"
    fail_job(job_id, reason)


def _is_lost_worker(exception: BaseException | None) -> bool:
    """Whether this failure is the process dying rather than the code raising."""
    return type(exception).__name__ in {"WorkerLostError", "Terminated", "TimeLimitExceeded"}


@worker_ready.connect
def _recover_orphans_on_startup(sender=None, **_kwargs) -> None:
    """Fail jobs left RUNNING by a worker that is no longer here.

    Runs once the worker is accepting work. Anything the live workers report as
    active is left alone; when they cannot be asked, nothing is touched, because
    an unknown answer is not evidence of an orphan.
    """
    app = getattr(sender, "app", None) or celery_app
    active = active_pipeline_jobs(app)
    if active is None:
        logger.warning("Skipping the orphaned-job sweep: could not confirm what is running")
        return
    recovered = recover_orphaned_jobs(active, LOST_WORKER_REASON)
    if recovered:
        logger.warning("Marked %d orphaned job(s) failed on startup: %s", len(recovered), recovered)


def enqueue_pipeline(job_id: int) -> None:
    """Hand a job to the worker queue. The one dispatch point the API calls."""
    run_pipeline_task.delay(job_id)
    logger.info("Queued pipeline for job %s", job_id)
