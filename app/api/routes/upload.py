"""Upload and job-listing endpoints.

Ordering here matters. The file is validated fully *before* anything is written
to storage or the database, so a rejected upload leaves no orphaned object and
no half-created row. Storage is written before the database row is committed,
because an object with no row is harmless garbage, whereas a row pointing at an
object that does not exist breaks every later stage.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.agents.schema_detection import AGENT_NAME as SCHEMA_AGENT
from app.agents.schema_detection import detect_schema
from app.agents.schema_models import SchemaReport
from app.api.schemas import (
    ArtifactLink,
    ArtifactSummary,
    ConfirmJobRequest,
    DatasetPreview,
    JobDetail,
    JobSummary,
    UploadResponse,
)
from app.core.config import get_settings
from app.core.db import get_db
from app.core.llm.base import LLMClient
from app.core.llm.factory import get_optional_llm
from app.core.llm.usage import make_usage_recorder
from app.core.storage import (
    StorageError,
    artifact_key,
    presigned_url,
    raw_dataset_key,
    upload_bytes,
)
from app.models.artifact import Artifact, ArtifactKind
from app.models.job import Job, JobStatus
from app.models.user import DEV_USER_ID
from app.services.artifacts import load_json_artifact, register_json_artifact
from app.services.csv_validation import (
    CSVValidationError,
    inspect_csv,
    validate_filename,
    validate_size,
)
from app.services.profiling import read_csv_frame
from app.worker.tasks import enqueue_pipeline

SCHEMA_ARTIFACT = "schema_report.json"
CONFIRMED_SCHEMA_ARTIFACT = "confirmed_schema.json"

logger = logging.getLogger(__name__)

router = APIRouter(tags=["jobs"])


@router.post(
    "/upload",
    response_model=UploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload a CSV and create a job",
)
def upload_csv(
    file: UploadFile = File(..., description="A UTF-8 encoded CSV file."),
    db: Session = Depends(get_db),
    llm: LLMClient | None = Depends(get_optional_llm),
) -> UploadResponse:
    settings = get_settings()

    # ---- Validate before touching storage or the database -------------------
    try:
        validate_filename(file.filename)
        data = file.file.read()
        validate_size(len(data), settings.max_upload_mb)
        summary = inspect_csv(data)
    except CSVValidationError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(exc)
        ) from exc
    finally:
        file.file.close()

    # ---- Reserve the job row to get an id for the storage key ---------------
    # The key contains the job id, so the row must exist first. It is flushed
    # (not committed) so that a storage failure below rolls the whole thing back.
    job = Job(
        user_id=DEV_USER_ID,
        original_filename=file.filename or "dataset.csv",
        s3_key="",  # set immediately below, once the id is known
        size_bytes=len(data),
        status=JobStatus.UPLOADED,
        n_rows=summary.n_rows,
        n_columns=summary.n_columns,
    )
    db.add(job)
    db.flush()

    key = raw_dataset_key(job.id)
    job.s3_key = key

    try:
        upload_bytes(key, data, content_type="text/csv")
    except StorageError as exc:
        db.rollback()
        logger.exception("Upload failed for job %s", job.id)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Could not store the file. Please try again.",
        ) from exc

    db.add(
        Artifact(
            job_id=job.id,
            kind=ArtifactKind.RAW_DATASET,
            name="dataset.csv",
            s3_key=key,
            content_type="text/csv",
            size_bytes=len(data),
        )
    )
    db.commit()
    db.refresh(job)

    logger.info(
        "Job %s created: %s (%d rows x %d columns)",
        job.id,
        job.original_filename,
        summary.n_rows,
        summary.n_columns,
    )

    # ---- Synchronous schema detection (the human checkpoint) ----------------
    # Runs here, in the request, while the user waits -- so confirmation happens
    # between two jobs and no running graph is ever paused (spec 7.2). Profiling
    # always succeeds; the LLM pass enriches it when available and is skipped
    # silently otherwise, so this can never fail an upload that already stored a
    # valid file.
    report = detect_schema(
        read_csv_frame(data),
        client=llm,
        on_usage=make_usage_recorder(db, job.id, SCHEMA_AGENT),
    )
    try:
        register_json_artifact(db, job.id, SCHEMA_ARTIFACT, report)
        db.commit()
    except StorageError:
        # The report survives in the response below, so the user can still
        # confirm; only the persisted copy is lost. Do not fail the upload.
        db.rollback()
        logger.exception("Could not persist schema report for job %s", job.id)

    return UploadResponse(
        job_id=job.id,
        filename=job.original_filename,
        status=job.status,
        size_bytes=job.size_bytes,
        created_at=job.created_at,
        schema_report=report,
        preview=DatasetPreview(
            n_rows=summary.n_rows,
            n_columns=summary.n_columns,
            columns=summary.columns,
            rows=summary.preview,
        ),
    )


@router.get("/jobs", response_model=list[JobSummary], summary="List jobs, newest first")
def list_jobs(limit: int = 50, db: Session = Depends(get_db)) -> list[Job]:
    return list(
        db.execute(select(Job).order_by(Job.created_at.desc()).limit(limit)).scalars().all()
    )


@router.get(
    "/jobs/{job_id}",
    response_model=JobDetail,
    summary="Fetch one job with its per-node pipeline status",
)
def get_job(job_id: int, db: Session = Depends(get_db)) -> Job:
    job = db.get(Job, job_id)
    if job is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"No job {job_id}.")
    return job


@router.get(
    "/jobs/{job_id}/schema",
    response_model=SchemaReport,
    summary="The detected schema report for a job",
)
def get_schema(job_id: int, db: Session = Depends(get_db)) -> SchemaReport:
    """Return the schema detected at upload, for a returning user to confirm."""
    if db.get(Job, job_id) is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"No job {job_id}.")
    payload = load_json_artifact(db, job_id, SCHEMA_ARTIFACT)
    if payload is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job {job_id} has no schema report.",
        )
    return SchemaReport.model_validate(payload)


@router.post(
    "/jobs",
    response_model=JobSummary,
    status_code=status.HTTP_200_OK,
    summary="Confirm a job's schema",
)
def confirm_job(request: ConfirmJobRequest, db: Session = Depends(get_db)) -> Job:
    """Store the user's confirmed schema and mark the job ready.

    This is the far side of the human checkpoint. In Section 3 it records the
    confirmed target, task type and exclusions; Section 4 extends it to launch
    the background pipeline from here. The confirmed target is validated against
    the columns actually in the dataset, so a stale or hand-crafted request
    cannot point the pipeline at a column that does not exist.
    """
    job = db.get(Job, request.job_id)
    if job is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"No job {request.job_id}."
        )

    stored = load_json_artifact(db, job.id, SCHEMA_ARTIFACT)
    if stored is not None:
        valid_columns = set(SchemaReport.model_validate(stored).column_names())
        if request.target_column not in valid_columns:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail=f"Target {request.target_column!r} is not a column in this dataset.",
            )

    confirmed = request.as_confirmed_schema()
    job.target_column = confirmed.target_column
    job.task_type = confirmed.task_type
    # Straight to QUEUED: confirming *is* launching the pipeline (spec 12.2).
    # Committing QUEUED before dispatching avoids a race where the worker moves
    # the job to RUNNING and this request then clobbers it back to QUEUED.
    job.status = JobStatus.QUEUED

    try:
        register_json_artifact(db, job.id, CONFIRMED_SCHEMA_ARTIFACT, confirmed)
        db.commit()
    except StorageError as exc:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Could not save the confirmed schema. Please try again.",
        ) from exc

    db.refresh(job)
    enqueue_pipeline(job.id)
    logger.info(
        "Job %s confirmed and queued: target=%s task=%s excluded=%s",
        job.id,
        confirmed.target_column,
        confirmed.task_type,
        confirmed.excluded(),
    )
    return job


@router.get(
    "/jobs/{job_id}/artifacts",
    response_model=list[ArtifactSummary],
    summary="List a job's artifacts",
)
def list_artifacts(job_id: int, db: Session = Depends(get_db)) -> list[Artifact]:
    if db.get(Job, job_id) is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"No job {job_id}.")
    return list(db.execute(select(Artifact).where(Artifact.job_id == job_id)).scalars().all())


@router.get(
    "/jobs/{job_id}/artifacts/{name}/link",
    response_model=ArtifactLink,
    summary="Get a temporary download link for an artifact",
)
def get_artifact_link(
    job_id: int,
    name: str,
    expires_in: int = 3600,
    db: Session = Depends(get_db),
) -> ArtifactLink:
    """Return a presigned URL.

    Lets the browser fetch plots and reports straight from object storage,
    without the API streaming bytes and without the bucket being public.

    Known limitation (local development only): the URL is signed against
    ``S3_ENDPOINT_URL``, which is ``http://minio:9000`` -- a hostname that only
    resolves inside the Docker network. It works from the API and worker, but a
    browser cannot follow it. Nothing renders artifacts in the browser yet, so
    this is not currently reachable; Section 6 is the first to display plots and
    must fix it, either by signing against a browser-reachable endpoint or by
    proxying the bytes through the API. On AWS the problem disappears, since
    real S3 URLs are publicly resolvable.
    """
    artifact = db.execute(
        select(Artifact).where(Artifact.job_id == job_id, Artifact.name == name)
    ).scalar_one_or_none()

    if artifact is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job {job_id} has no artifact named {name!r}.",
        )

    try:
        url = presigned_url(artifact.s3_key, expires_in=expires_in)
    except StorageError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Storage is unavailable.",
        ) from exc

    return ArtifactLink(name=artifact.name, url=url, expires_in_seconds=expires_in)


__all__ = ["router", "artifact_key"]
