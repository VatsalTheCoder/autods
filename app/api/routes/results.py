"""Results endpoints -- what the Results page reads after a job finishes.

These serve artifact *content* through the API rather than handing the browser a
presigned URL. That is a deliberate choice, not an oversight: locally, presigned
URLs are signed against ``http://minio:9000``, a hostname that only resolves
inside the Docker network, so a browser cannot follow one (the limitation
documented on ``GET /jobs/{id}/artifacts/{name}/link``).

Section 5 proxied only JSON and Markdown and left the general case open. Section 6
closes it: ``GET /jobs/{id}/artifacts/{name}/content`` streams *any* artifact with
its recorded content type, which is what makes the EDA charts displayable in a
browser. The alternative -- signing against a browser-reachable endpoint -- would
mean the local and deployed paths differ in exactly the place that is hardest to
test, and would leave two mechanisms for "show the user a file" where one will do.

The ``/link`` endpoint stays for the case presigned URLs are actually good at:
letting a client pull a large object straight from S3 without the API in the
middle. Nothing in the UI needs that yet.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Response, status
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
    load_artifact_content,
    load_json_artifact,
)

# Artifacts are immutable for the life of a job unless it is re-run, and the
# Results page re-executes its whole script on every interaction (Streamlit's
# model), which would otherwise re-fetch every chart on each click. A short
# private cache keeps that from being visible without risking a stale image
# outliving a re-run by more than a few minutes.
_CACHE_CONTROL = "private, max-age=300"

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


@router.get(
    "/jobs/{job_id}/artifacts/{name}/content",
    summary="Stream an artifact's bytes, with its recorded content type",
    response_class=Response,
    responses={200: {"content": {"image/png": {}, "text/csv": {}, "application/json": {}}}},
)
def get_artifact_content(job_id: int, name: str, db: Session = Depends(get_db)) -> Response:
    """Serve any artifact directly, so a browser can display it.

    This is what makes the EDA charts viewable (Section 6). A presigned URL
    cannot be followed from a browser against local MinIO, so the bytes come
    through the API instead -- identical behaviour on a laptop and on AWS, which
    matters more here than saving the API a few hundred kilobytes.

    The content type is whatever the producing agent recorded, so a PNG is served
    as a PNG and a CSV as a CSV without this endpoint knowing anything about
    either.
    """
    _require_job(db, job_id)
    try:
        found = load_artifact_content(db, job_id, name)
    except StorageError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Storage is unavailable."
        ) from exc
    if found is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job {job_id} has no artifact named {name!r}.",
        )

    data, content_type = found
    return Response(
        content=data,
        media_type=content_type,
        headers={
            "Cache-Control": _CACHE_CONTROL,
            # Named so a browser "save as" produces the artifact's real name
            # rather than the URL's last path segment ("content").
            "Content-Disposition": f'inline; filename="{name}"',
        },
    )
