"""The pipeline runner -- the plain function the Celery task wraps.

Kept as an ordinary function (not glued to Celery) so it can be called directly
in tests, no broker or worker required. The Celery task in ``tasks.py`` is a
one-line wrapper around it.

Its job is everything around the graph: move the job to RUNNING, lay out the
per-node roadmap, gather the inputs the graph needs -- the dataset read **back
from S3** (never local disk: the worker is a different machine from the API as
far as the code is concerned) plus the schema the user confirmed at the
checkpoint -- run the graph, and land the job on COMPLETED or, on any failure,
FAILED with a readable reason instead of hanging (spec 10).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd

from app.agents.schema_models import ConfirmedSchema, SchemaReport
from app.core.db import SessionLocal
from app.core.storage import download_bytes, raw_dataset_key
from app.models.job import Job, JobStatus
from app.services.artifacts import (
    CONFIRMED_SCHEMA_ARTIFACT,
    SCHEMA_ARTIFACT,
    load_json_artifact,
)
from app.services.profiling import profile_dataset, read_csv_frame
from app.worker.graph import build_pipeline_graph
from app.worker.progress import init_agent_runs, set_job_status
from app.worker.state import PIPELINE_NODES

logger = logging.getLogger(__name__)


class PipelineInputError(RuntimeError):
    """The job cannot be run at all -- raised before any node starts."""


@dataclass(slots=True)
class JobContext:
    """Everything about a job the graph needs, gathered before it starts."""

    filename: str
    target: str
    task_type: str
    excluded: list[str]


def _load_dataset(job_id: int) -> pd.DataFrame:
    """Fetch the uploaded CSV from object storage. Proof the worker reads S3."""
    data = download_bytes(raw_dataset_key(job_id))
    return read_csv_frame(data)


def _load_context(job_id: int) -> JobContext:
    """Read the job row and the schema the user confirmed.

    The target and task type are read from the job row rather than the confirmed
    schema artifact -- they are mirrored there at confirmation time precisely so
    every later stage can get them with one column read (see ``models/job.py``).
    The exclusion list has no such mirror, so it comes from the artifact.
    """
    with SessionLocal() as db:
        job = db.get(Job, job_id)
        if job is None:
            raise PipelineInputError(f"No job {job_id}.")
        if not job.target_column or not job.task_type:
            raise PipelineInputError(
                f"Job {job_id} has no confirmed target column; the schema was never confirmed."
            )

        excluded: list[str] = []
        payload = load_json_artifact(db, job_id, CONFIRMED_SCHEMA_ARTIFACT)
        if payload is not None:
            excluded = ConfirmedSchema.model_validate(payload).excluded()

        return JobContext(
            filename=job.original_filename,
            target=job.target_column,
            task_type=job.task_type,
            excluded=excluded,
        )


def _load_schema(job_id: int, frame: pd.DataFrame) -> SchemaReport:
    """The schema report, re-profiled if the stored copy is missing.

    Cleaning uses it to decide which columns were stored as the wrong type. The
    artifact is written best-effort at upload (a storage blip there must not fail
    an upload whose file is already safe), so it can legitimately be absent --
    and profiling is deterministic, so recomputing it costs a second and gives
    the same answer.
    """
    with SessionLocal() as db:
        payload = load_json_artifact(db, job_id, SCHEMA_ARTIFACT)
    if payload is not None:
        return SchemaReport.model_validate(payload)
    logger.info("Job %s has no stored schema report; re-profiling", job_id)
    return profile_dataset(frame)


def run_pipeline(job_id: int) -> None:
    """Execute the pipeline for one job, end to end, updating status as it goes."""
    logger.info("Pipeline starting for job %s", job_id)
    set_job_status(job_id, JobStatus.RUNNING)
    init_agent_runs(job_id, PIPELINE_NODES)

    try:
        context = _load_context(job_id)
        frame = _load_dataset(job_id)
        schema = _load_schema(job_id, frame)

        graph = build_pipeline_graph()
        graph.invoke(
            {
                "job_id": job_id,
                "filename": context.filename,
                "frame": frame,
                "n_rows": int(frame.shape[0]),
                "schema": schema,
                "target": context.target,
                "task_type": context.task_type,
                "excluded": context.excluded,
                "completed": [],
                "notes": {},
            }
        )
    except Exception as exc:
        # Any failure -- a node raising, S3 unreachable, an unconfirmed job --
        # lands here. The job is marked FAILED with the reason; a failing node
        # has already marked itself FAILED (see graph.py), so the UI can point at
        # the exact step. Nothing is left RUNNING forever.
        logger.exception("Pipeline failed for job %s", job_id)
        set_job_status(job_id, JobStatus.FAILED, error=str(exc))
        return

    set_job_status(job_id, JobStatus.COMPLETED)
    logger.info("Pipeline completed for job %s", job_id)
