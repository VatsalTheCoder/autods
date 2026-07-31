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
from app.agents.schema_models import ConfirmedColumn, SchemaReport
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
from app.services.artifacts import (
    CONFIRMED_SCHEMA_ARTIFACT,
    SCHEMA_ARTIFACT,
    load_json_artifact,
    register_json_artifact,
)
from app.services.csv_validation import (
    CSVValidationError,
    inspect_csv,
    validate_filename,
    validate_size,
)
from app.services.profiling import read_csv_frame
from app.worker.tasks import enqueue_pipeline

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
    detected = SchemaReport.model_validate(stored) if stored is not None else None
    if detected is not None and request.target_column not in set(detected.column_names()):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=f"Target {request.target_column!r} is not a column in this dataset.",
        )

    if request.time_column and detected is not None:
        column = detected.column(request.time_column)
        if column is None:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail=f"Time column {request.time_column!r} is not a column in this dataset.",
            )
        # Checked against what detection typed rather than accepted on trust.
        # Ordering by a categorical or free-text column produces a split that
        # looks time-aware and is not, which is worse than plain random folds
        # because the report would then claim a guarantee it does not have.
        #
        # Numeric is allowed alongside datetime because a great many datasets
        # carry time as a counter -- an hour index, epoch seconds, a day number.
        # Ordering needs values that compare, not values that parse as dates.
        if column.semantic_type not in ("datetime", "numeric"):
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail=(
                    f"Time column {request.time_column!r} was detected as "
                    f"{column.semantic_type}. Folds can only be ordered by a date "
                    "or by a number that counts time."
                ),
            )
        if request.time_column == request.target_column:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail="The time column cannot also be the target.",
            )

    confirmed = request.as_confirmed_schema()

    if not confirmed.columns and detected is not None:
        # An omitted ``columns`` means "no opinion", not "no exclusions".
        #
        # Detection sets ``exclude=True`` on every PII column, and ColumnProfile
        # says why: the safe choice is the default and the user opts back in.
        # Taking the request literally here quietly broke that promise -- a
        # caller who left the array out got a 200 and a model trained on the PII
        # that had already been flagged. Observed on a real run, where the
        # pipeline modelled an email column and the critic then remarked on the
        # model's reliance on it.
        #
        # The UI always sends the full list, so this path is for API-first
        # callers. Inheriting is the conservative reading: to model a PII column
        # you now send it explicitly with ``exclude: false``.
        confirmed.columns = [
            ConfirmedColumn(
                name=column.name,
                is_pii=column.is_pii,
                # Never inherit an exclusion onto the target. A PII-flagged
                # target -- an ``email`` column being predicted -- would
                # otherwise exclude the very column the run is about, and
                # nothing downstream validates against that.
                exclude=column.exclude and column.name != confirmed.target_column,
            )
            for column in detected.columns
        ]
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
    """Return a presigned URL for fetching an artifact straight from storage.

    For a client that can reach object storage directly, this is the efficient
    path: the bytes never pass through the API.

    **The UI does not use it, and should not.** The URL is signed against
    ``S3_ENDPOINT_URL`` -- locally ``http://minio:9000``, a hostname that resolves
    only inside the Docker network, so a browser cannot follow it. On AWS real S3
    URLs resolve fine, which is exactly what makes this a trap: it would work in
    production and fail on every developer's laptop. Section 6 settled the
    question by serving bytes through ``GET /jobs/{id}/artifacts/{name}/content``
    instead, which behaves identically in both places.

    Kept because a future non-browser consumer (a notebook, a bulk export) is
    better served by a direct link than by streaming through the API.
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
