"""Serving the saved model -- what ``POST /jobs/{id}/predict`` actually does.

The endpoint's promise is that a caller sends the columns they uploaded and gets
a prediction back. Everything between those two points -- imputation, encoding,
scaling, feature selection -- happens inside the pickle, because the whole
pipeline was saved as one object rather than the estimator alone. That choice is
what makes this module short, and it is also what makes serving *correct*: a
model served behind preprocessing rebuilt by hand at request time is a model
being fed a different distribution from the one it was trained on, which is one
of the commonest ways a good offline score fails to survive contact with
production.

Two smaller things this module has to get right:

* **Types.** JSON and HTML forms both hand over strings, so ``"41"`` arrives
  where the training data held ``41``. A numeric branch given the string would
  fail inside the imputer with an error about dtypes that says nothing useful, so
  columns the training data recorded as numeric are coerced back to numbers here,
  and an uncoercible value becomes missing -- which the pipeline already knows how
  to handle.
* **Absent columns.** A column the caller omits is filled with NaN and imputed,
  exactly as a gap in the training data would be. That is a real prediction on
  partial information rather than an error, but it is reported in the response,
  because the same silence would otherwise hide a misspelled column name behind a
  confident-looking answer.
"""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass, field
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.storage import download_bytes
from app.ml.contracts import FinalModelInfo
from app.models.artifact import Artifact
from app.services.artifacts import (
    FINAL_MODEL_ARTIFACT,
    FINAL_MODEL_INFO_ARTIFACT,
    load_json_artifact,
)

logger = logging.getLogger(__name__)

# Dtypes recorded by final training that mean "this column is a number". Checked
# against the string the training frame reported rather than re-inspecting the
# data, so serving agrees with training by construction.
_NUMERIC_DTYPES = ("int", "float", "uint")

# Loaded pipelines, keyed by job and by the artifact's byte size. Deserialising a
# forest for every request would dominate the response time, and the size is
# enough of a fingerprint to invalidate the entry when a job is re-run and
# overwrites its model -- a re-run that produced a byte-identical model is one
# where the cached object is correct anyway.
_MODEL_CACHE: dict[tuple[int, int], Any] = {}
_MODEL_CACHE_LIMIT = 4


class PredictionError(RuntimeError):
    """The request cannot be served, with a reason meant for the caller."""


class ModelNotReady(PredictionError):
    """The job has not produced a final model yet -- an ordinary state, not a fault."""


@dataclass(slots=True)
class Prediction:
    """One row's answer and, for a classifier, its class probabilities."""

    prediction: Any
    probabilities: dict[str, float] = field(default_factory=dict)


@dataclass(slots=True)
class PredictionOutcome:
    """Everything the endpoint returns, including what it noticed about the input."""

    predictions: list[Prediction]
    info: FinalModelInfo
    missing_columns: list[str] = field(default_factory=list)
    unexpected_columns: list[str] = field(default_factory=list)


def load_final_model(db: Session, job_id: int) -> tuple[Any, FinalModelInfo]:
    """Fetch the job's fitted pipeline and its description, from cache or S3.

    The pickle is one this pipeline wrote to its own bucket, so ``joblib.load``
    is reading data the application produced rather than anything a user
    supplied -- worth stating, since unpickling untrusted bytes would be a
    different matter entirely.
    """
    payload = load_json_artifact(db, job_id, FINAL_MODEL_INFO_ARTIFACT)
    if payload is None:
        raise ModelNotReady(
            f"Job {job_id} has no trained model yet. A model is saved when the "
            "pipeline reaches final training."
        )
    info = FinalModelInfo.model_validate(payload)

    artifact = db.execute(
        select(Artifact).where(Artifact.job_id == job_id, Artifact.name == FINAL_MODEL_ARTIFACT)
    ).scalar_one_or_none()
    if artifact is None:
        raise ModelNotReady(f"Job {job_id} has no {FINAL_MODEL_ARTIFACT} artifact.")

    key = (job_id, int(artifact.size_bytes or 0))
    cached = _MODEL_CACHE.get(key)
    if cached is None:
        logger.info("Loading %s for job %s from storage", FINAL_MODEL_ARTIFACT, job_id)
        cached = joblib.load(io.BytesIO(download_bytes(artifact.s3_key)))
        if len(_MODEL_CACHE) >= _MODEL_CACHE_LIMIT:
            # Plain FIFO eviction. The cache exists to make a burst of requests
            # against one job fast, not to serve every job on the machine.
            _MODEL_CACHE.pop(next(iter(_MODEL_CACHE)))
        _MODEL_CACHE[key] = cached

    return cached, info


def predict_rows(db: Session, job_id: int, rows: list[dict[str, Any]]) -> PredictionOutcome:
    """Run the saved pipeline over ``rows`` and describe what it was given."""
    if not rows:
        raise PredictionError("No rows to predict.")

    pipeline, info = load_final_model(db, job_id)
    known = [column.name for column in info.feature_columns]
    # Only the columns the recipe actually consumes are validated against. A
    # column the strategy dropped still has to be *present* in the frame the
    # transformer sees, but a caller who leaves it out has not omitted anything
    # the prediction depends on, and telling them otherwise would be noise.
    expected = [column.name for column in info.feature_columns if column.used]
    if not expected:
        raise PredictionError(
            f"Job {job_id}'s saved model does not record its input columns, so a "
            "prediction request cannot be validated against it."
        )

    supplied = {key for row in rows for key in row}
    missing = [name for name in expected if name not in supplied]
    unexpected = sorted(supplied - set(known))
    if len(missing) == len(expected):
        raise PredictionError(
            "None of the model's input columns were supplied. It expects: "
            + ", ".join(expected[:10])
            + ("..." if len(expected) > 10 else "")
        )

    frame = _build_frame(rows, info)
    predicted = np.asarray(pipeline.predict(frame))
    probabilities = _probabilities(pipeline, frame, info)

    predictions = [
        Prediction(
            prediction=_native(predicted[index]),
            probabilities=probabilities[index] if probabilities else {},
        )
        for index in range(len(frame))
    ]
    logger.info("Job %s served %d prediction(s) from %s", job_id, len(predictions), info.model_name)
    return PredictionOutcome(
        predictions=predictions,
        info=info,
        missing_columns=missing,
        unexpected_columns=unexpected,
    )


def _build_frame(rows: list[dict[str, Any]], info: FinalModelInfo) -> pd.DataFrame:
    """Assemble the request into the frame the recipe expects.

    Column *order* matters as much as membership: a ``ColumnTransformer`` selects
    its branches by name, but a mismatched set would fail at transform time with
    an error about columns rather than about the request. Building the frame from
    the recorded column list rather than from the request's keys means the shape
    is right by construction and any difference shows up as a reported missing or
    unexpected column instead.
    """
    data: dict[str, list[Any]] = {}
    for column in info.feature_columns:
        values = [row.get(column.name) for row in rows]
        series = pd.Series(values, dtype="object")
        if any(token in column.dtype for token in _NUMERIC_DTYPES):
            # Coerced, not rejected: "41" from a form is a 41, and "n/a" is a
            # missing value the pipeline's imputer already has a rule for.
            series = pd.to_numeric(series, errors="coerce")
        data[column.name] = series
    return pd.DataFrame(data)


def _probabilities(pipeline, frame: pd.DataFrame, info: FinalModelInfo) -> list[dict[str, float]]:
    """Per-class probabilities keyed by the labels the user uploaded.

    The estimator's columns are in its own class order, which for the XGBoost
    wrapper is the *encoded* 0..n-1 order rather than the labels. Zipping against
    ``classes_`` -- which the wrapper sets from its encoder -- is what keeps a
    probability attached to the right label instead of to the right position.
    """
    if not info.classes or not hasattr(pipeline, "predict_proba"):
        return []
    try:
        matrix = np.asarray(pipeline.predict_proba(frame), dtype=float)
    except (AttributeError, ValueError):  # pragma: no cover - estimator-specific
        return []
    if matrix.shape[1] != len(info.classes):  # pragma: no cover - defensive
        return []
    return [
        {label: float(value) for label, value in zip(info.classes, row, strict=True)}
        for row in matrix
    ]


def _native(value: Any) -> Any:
    """NumPy scalars are not JSON-serialisable; their Python equivalents are."""
    if isinstance(value, np.generic):
        return value.item()
    return value
