"""Results endpoints -- what the Results page reads after a job finishes.

These serve artifact *content* through the API rather than handing the browser a
presigned URL. That is a deliberate choice, not an oversight: locally, presigned
URLs are signed against ``http://minio:9000``, a hostname that only resolves
inside the Docker network, so a browser cannot follow one (the limitation
documented on ``GET /jobs/{id}/artifacts/{name}/link``). Both payloads here are
small -- a JSON report and a few kilobytes of Markdown -- so proxying them costs
nothing and the Results page works identically on a laptop and on AWS.

Plots, which are large and numerous, are the case where presigned URLs genuinely
pay off; Section 6 introduces them and has to solve the endpoint problem properly.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.api.schemas import ReportResponse
from app.core.db import get_db
from app.core.storage import StorageError
from app.ml.contracts import CleaningReport, EvaluationReport
from app.models.job import Job
from app.services.artifacts import (
    CLEANING_ARTIFACT,
    EVALUATION_ARTIFACT,
    REPORT_ARTIFACT,
    load_artifact_bytes,
    load_json_artifact,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["results"])


def _require_job(db: Session, job_id: int) -> Job:
    job = db.get(Job, job_id)
    if job is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"No job {job_id}.")
    return job


def _load_or_404(db: Session, job_id: int, name: str) -> dict:
    """Fetch a JSON artifact, turning both "no job" and "not produced" into 404s.

    A job that is still running has no evaluation report yet, and that is a
    perfectly ordinary state -- the Results page polls into it. So the message
    distinguishes "this job does not exist" from "this job has not got there
    yet", rather than both surfacing as a bare 404.
    """
    _require_job(db, job_id)
    try:
        payload = load_json_artifact(db, job_id, name)
    except StorageError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Storage is unavailable."
        ) from exc
    if payload is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job {job_id} has not produced {name} yet.",
        )
    return payload


@router.get(
    "/jobs/{job_id}/evaluation",
    response_model=EvaluationReport,
    summary="The cross-validated metrics for a job",
)
def get_evaluation(job_id: int, db: Session = Depends(get_db)) -> EvaluationReport:
    return EvaluationReport.model_validate(_load_or_404(db, job_id, EVALUATION_ARTIFACT))


@router.get(
    "/jobs/{job_id}/cleaning",
    response_model=CleaningReport,
    summary="What cleaning did to a job's dataset",
)
def get_cleaning(job_id: int, db: Session = Depends(get_db)) -> CleaningReport:
    return CleaningReport.model_validate(_load_or_404(db, job_id, CLEANING_ARTIFACT))


@router.get(
    "/jobs/{job_id}/report",
    response_model=ReportResponse,
    summary="The Markdown report for a job",
)
def get_report(job_id: int, db: Session = Depends(get_db)) -> ReportResponse:
    """Return the report as Markdown text for the UI to render."""
    _require_job(db, job_id)
    try:
        data = load_artifact_bytes(db, job_id, REPORT_ARTIFACT)
    except StorageError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Storage is unavailable."
        ) from exc
    if data is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job {job_id} has not produced a report yet.",
        )
    return ReportResponse(job_id=job_id, markdown=data.decode("utf-8"))
