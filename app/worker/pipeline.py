"""The pipeline runner -- the plain function the Celery task wraps.

Kept as an ordinary function (not glued to Celery) so it can be called directly
in tests, no broker or worker required. The Celery task in ``tasks.py`` is a
one-line wrapper around it.

Its job is the state machine around the graph: move the job to RUNNING, lay out
the per-node roadmap, read the dataset **back from S3** (never local disk -- the
worker is a different machine from the API as far as the code is concerned), run
the graph, and land the job on COMPLETED or, on any failure, FAILED with a
readable reason instead of hanging (spec 10).
"""

from __future__ import annotations

import logging

import pandas as pd

from app.core.storage import download_bytes, raw_dataset_key
from app.models.job import JobStatus
from app.services.profiling import read_csv_frame
from app.worker.graph import build_pipeline_graph
from app.worker.progress import init_agent_runs, set_job_status
from app.worker.state import PIPELINE_NODES

logger = logging.getLogger(__name__)


def _load_dataset(job_id: int) -> pd.DataFrame:
    """Fetch the uploaded CSV from object storage. Proof the worker reads S3."""
    data = download_bytes(raw_dataset_key(job_id))
    return read_csv_frame(data)


def run_pipeline(job_id: int) -> None:
    """Execute the pipeline for one job, end to end, updating status as it goes."""
    logger.info("Pipeline starting for job %s", job_id)
    set_job_status(job_id, JobStatus.RUNNING)
    init_agent_runs(job_id, PIPELINE_NODES)

    try:
        frame = _load_dataset(job_id)
        graph = build_pipeline_graph()
        graph.invoke(
            {"job_id": job_id, "n_rows": int(frame.shape[0]), "completed": [], "notes": {}}
        )
    except Exception as exc:
        # Any failure -- a node raising, S3 unreachable, a bad graph -- lands
        # here. The job is marked FAILED with the reason; the failing node has
        # already marked itself FAILED (see graph.py), so the UI can point at
        # the exact step. Nothing is left RUNNING forever.
        logger.exception("Pipeline failed for job %s", job_id)
        set_job_status(job_id, JobStatus.FAILED, error=str(exc))
        return

    set_job_status(job_id, JobStatus.COMPLETED)
    logger.info("Pipeline completed for job %s", job_id)
