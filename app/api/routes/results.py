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

from app.api.schemas import (
    PredictionRequest,
    PredictionResponse,
    ReportResponse,
    RowPrediction,
)
from app.core.db import get_db
from app.core.storage import StorageError
from app.ml.contracts import (
    CleaningReport,
    ClusteringReport,
    CriticReport,
    EdaReport,
    EvaluationReport,
    ExplainabilityReport,
    FeatureStrategy,
    FinalModelInfo,
    Leaderboard,
)
from app.models.job import Job
from app.services.artifacts import (
    CLEANING_ARTIFACT,
    CLUSTERING_ARTIFACT,
    CRITIC_ARTIFACT,
    EDA_ARTIFACT,
    EVALUATION_ARTIFACT,
    EXPLAINABILITY_ARTIFACT,
    FEATURE_ARTIFACT,
    FINAL_MODEL_INFO_ARTIFACT,
    LEADERBOARD_ARTIFACT,
    REPORT_ARTIFACT,
    REPORT_PDF_ARTIFACT,
    load_artifact_bytes,
    load_artifact_content,
    load_json_artifact,
)
from app.services.prediction import ModelNotReady, PredictionError, predict_rows

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
    "/jobs/{job_id}/eda",
    response_model=EdaReport,
    summary="Descriptive statistics and the list of charts",
)
def get_eda(job_id: int, db: Session = Depends(get_db)) -> EdaReport:
    """The statistics, plus the artifact names of the charts.

    The chart *bytes* come from the content endpoint below; this returns the
    index, so the UI knows what exists before fetching any of it.
    """
    return EdaReport.model_validate(_load_or_404(db, job_id, EDA_ARTIFACT))


@router.get(
    "/jobs/{job_id}/clustering",
    response_model=ClusteringReport,
    summary="The groups found in the data, and how well separated they are",
)
def get_clustering(job_id: int, db: Session = Depends(get_db)) -> ClusteringReport:
    return ClusteringReport.model_validate(_load_or_404(db, job_id, CLUSTERING_ARTIFACT))


@router.get(
    "/jobs/{job_id}/leaderboard",
    response_model=Leaderboard,
    summary="Every model that was tried, ranked on the same folds",
)
def get_leaderboard(job_id: int, db: Session = Depends(get_db)) -> Leaderboard:
    """The ranking. ``/evaluation`` is the detailed account of the winner alone."""
    return Leaderboard.model_validate(_load_or_404(db, job_id, LEADERBOARD_ARTIFACT))


@router.get(
    "/jobs/{job_id}/features",
    response_model=FeatureStrategy,
    summary="The per-column preparation the strategy agent chose, and what was overruled",
)
def get_features(job_id: int, db: Session = Depends(get_db)) -> FeatureStrategy:
    """Deliberately served separately from the built recipe (spec 7.6).

    This is *the decision*; ``preprocessing_report.json`` is what was built from
    it. Two endpoints because a reader checking "did the LLM get overruled, and
    where" is asking a different question from "what does the pipeline do".
    """
    return FeatureStrategy.model_validate(_load_or_404(db, job_id, FEATURE_ARTIFACT))


@router.get(
    "/jobs/{job_id}/explainability",
    response_model=ExplainabilityReport,
    summary="SHAP over the saved model, in the user's own column names",
)
def get_explainability(job_id: int, db: Session = Depends(get_db)) -> ExplainabilityReport:
    """The explanation (spec 7.10).

    Includes ``feature_name_mapping`` -- every encoded feature and the column it
    came from -- so a reader can check the translation rather than trust it.
    """
    return ExplainabilityReport.model_validate(_load_or_404(db, job_id, EXPLAINABILITY_ARTIFACT))


@router.get(
    "/jobs/{job_id}/model",
    response_model=FinalModelInfo,
    summary="What the served model is, and which columns it expects",
)
def get_final_model(job_id: int, db: Session = Depends(get_db)) -> FinalModelInfo:
    """The description of ``final_model.pkl``.

    The prediction form reads this to know what to ask for, which is why the
    input columns carry a dtype and an example value: a user guessing whether
    ``tenure_months`` wants years is a user who submits a bad prediction.
    """
    return FinalModelInfo.model_validate(_load_or_404(db, job_id, FINAL_MODEL_INFO_ARTIFACT))


@router.post(
    "/jobs/{job_id}/predict",
    response_model=PredictionResponse,
    summary="Predict new rows with the model this job trained",
)
def predict(
    job_id: int, request: PredictionRequest, db: Session = Depends(get_db)
) -> PredictionResponse:
    """Load the saved pipeline and run it over the supplied rows (spec 7.11).

    The pipeline is loaded from object storage rather than held in the API
    process: the model is produced by the *worker*, on a different machine as far
    as this code is concerned, and S3 is the only thing both of them can see.

    ``ModelNotReady`` becomes a 404 and any other prediction problem a 400. The
    distinction is real -- "this job has not trained a model yet" is an ordinary
    state a Results page polls into, while "your rows do not match the model's
    columns" is a request that will never work as sent.
    """
    _require_job(db, job_id)
    try:
        outcome = predict_rows(db, job_id, request.rows)
    except ModelNotReady as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except PredictionError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except StorageError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Storage is unavailable."
        ) from exc

    return PredictionResponse(
        job_id=job_id,
        model_name=outcome.info.model_name,
        target_column=outcome.info.target_column,
        predictions=[
            RowPrediction(prediction=row.prediction, probabilities=row.probabilities)
            for row in outcome.predictions
        ],
        missing_columns=outcome.missing_columns,
        unexpected_columns=outcome.unexpected_columns,
    )


@router.get(
    "/jobs/{job_id}/critic",
    response_model=CriticReport,
    summary="The review of the whole run, worst finding first",
)
def get_critic(job_id: int, db: Session = Depends(get_db)) -> CriticReport:
    """The critique (spec 7.11).

    ``omissions`` is part of the contract, not an aside: on a wide dataset the
    reviewer saw a capped summary, and a critique that looks complete when it is
    not would be worse than none.
    """
    return CriticReport.model_validate(_load_or_404(db, job_id, CRITIC_ARTIFACT))


@router.get(
    "/jobs/{job_id}/report/pdf",
    summary="The report as a PDF",
    response_class=Response,
    responses={200: {"content": {"application/pdf": {}}}},
)
def get_report_pdf(job_id: int, db: Session = Depends(get_db)) -> Response:
    """Serve the rendered report (spec 7.12).

    Separate from ``/report`` rather than a format parameter: they return
    different media types with different caching and download behaviour, and one
    endpoint that returns either is one endpoint a client has to branch on.

    A 404 here can mean the run has not reached the report yet *or* that
    rendering failed on a run that otherwise completed -- the PDF is best-effort
    precisely so a font problem cannot discard a finished analysis. The Markdown
    at ``/report`` is the authoritative document either way.
    """
    _require_job(db, job_id)
    try:
        data = load_artifact_bytes(db, job_id, REPORT_PDF_ARTIFACT)
    except StorageError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Storage is unavailable."
        ) from exc
    if data is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                f"Job {job_id} has no PDF report. It may still be running, or "
                "rendering may have failed -- the Markdown report at /report is "
                "the authoritative version."
            ),
        )
    return Response(
        content=data,
        media_type="application/pdf",
        headers={
            "Cache-Control": _CACHE_CONTROL,
            "Content-Disposition": f'attachment; filename="autods_job_{job_id}_report.pdf"',
        },
    )


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
