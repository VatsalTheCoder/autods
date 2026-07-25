"""Celery application.

Section 0 defined the worker and a trivial ``ping`` task to prove the
API -> Redis -> worker path works. Section 4 adds the real pipeline task
(``autods.run_pipeline``), discovered from ``app.worker.tasks`` at worker
startup and dispatched by ``POST /jobs``.
"""

from __future__ import annotations

import logging

from celery import Celery

from app.core.config import get_settings

logger = logging.getLogger(__name__)

settings = get_settings()

celery_app = Celery(
    "autods",
    broker=str(settings.redis_url),
    backend=str(settings.redis_url),
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    # Only take a new task once the current one is finished. ML jobs are long
    # and uneven, so prefetching would leave work queued behind a busy worker
    # while another sits idle.
    worker_prefetch_multiplier=1,
    # Report a task as started, so the Progress page can distinguish
    # "queued" from "running" (Section 4).
    task_track_started=True,
    # Results outlive a demo session but do not accumulate forever.
    result_expires=86400,
)

# Import app.worker.tasks on worker startup so the pipeline task registers.
celery_app.autodiscover_tasks(["app.worker"])


@celery_app.task(name="autods.ping")
def ping() -> str:
    """Smoke test proving the worker consumes from the queue."""
    logger.info("ping task executed")
    return "pong"
