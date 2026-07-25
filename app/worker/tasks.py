"""Celery tasks and the enqueue helper.

The task is a thin wrapper: all the real logic lives in ``run_pipeline`` so it
can be tested without a broker. ``enqueue_pipeline`` is the single call sites use
to dispatch work, which keeps the API decoupled from Celery specifics and gives
tests one obvious thing to stub.
"""

from __future__ import annotations

import logging

from app.worker.celery_app import celery_app
from app.worker.pipeline import run_pipeline

logger = logging.getLogger(__name__)


@celery_app.task(name="autods.run_pipeline")
def run_pipeline_task(job_id: int) -> None:
    """Run the full pipeline for a job in the background worker."""
    run_pipeline(job_id)


def enqueue_pipeline(job_id: int) -> None:
    """Hand a job to the worker queue. The one dispatch point the API calls."""
    run_pipeline_task.delay(job_id)
    logger.info("Queued pipeline for job %s", job_id)
