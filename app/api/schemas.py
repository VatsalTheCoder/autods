"""Request and response shapes for the API.

Separate from the ORM models on purpose: the database schema and the public API
contract change for different reasons, and coupling them means every internal
column rename becomes a breaking API change.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.agents.schema_models import ConfirmedSchema, SchemaReport
from app.models.job import JobStatus


class DatasetPreview(BaseModel):
    """A glimpse of the uploaded data, so the user can confirm it parsed correctly."""

    n_rows: int = Field(..., description="Total rows, excluding the header.")
    n_columns: int
    columns: list[str]
    rows: list[dict[str, Any]] = Field(..., description="First few rows.")


class UploadResponse(BaseModel):
    job_id: int
    filename: str
    status: JobStatus
    size_bytes: int
    preview: DatasetPreview
    # Detected synchronously during upload (Section 3), so the confirmation
    # screen can render immediately without a second round-trip.
    schema_report: SchemaReport
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class JobSummary(BaseModel):
    id: int
    original_filename: str
    status: JobStatus
    n_rows: int | None
    n_columns: int | None
    size_bytes: int
    target_column: str | None
    task_type: str | None
    error_message: str | None
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)


class AgentRunSummary(BaseModel):
    """One pipeline node's status, for the Progress page (Section 4)."""

    name: str
    sequence: int
    status: str
    error_message: str | None
    started_at: datetime | None
    finished_at: datetime | None

    model_config = ConfigDict(from_attributes=True)


class JobDetail(JobSummary):
    """A single job plus its per-node pipeline status.

    Returned by ``GET /jobs/{id}`` so one poll gives the Progress page both the
    overall status and where the pipeline currently is. The list view stays on
    the lighter ``JobSummary``.
    """

    agent_runs: list[AgentRunSummary] = Field(default_factory=list)


class ConfirmJobRequest(ConfirmedSchema):
    """The confirmed schema, plus which job it belongs to.

    Extends ``ConfirmedSchema`` so the target/task/exclusion validation is
    shared with the agent layer and defined once.
    """

    job_id: int

    def as_confirmed_schema(self) -> ConfirmedSchema:
        """The schema without the transport-only ``job_id``, for persistence."""
        return ConfirmedSchema(
            target_column=self.target_column,
            task_type=self.task_type,
            columns=self.columns,
        )


class ArtifactSummary(BaseModel):
    id: int
    name: str
    kind: str
    content_type: str
    size_bytes: int | None
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class ArtifactLink(BaseModel):
    """A time-limited direct link to an artifact in object storage."""

    name: str
    url: str
    expires_in_seconds: int


class ReportResponse(BaseModel):
    """The Markdown report, served as text for the UI to render (Section 5)."""

    job_id: int
    markdown: str


class PredictionRequest(BaseModel):
    """Rows to predict, in the user's own raw columns (Section 8).

    Raw columns, not encoded ones: the saved pipeline carries its own
    preprocessing, so a caller sends ``{"city": "London", "age": 41}`` and never
    has to know what the recipe did to either. That is the whole reason the
    preprocessor and the estimator were saved as one object.

    Values are ``Any`` because a column can legitimately be a number, a label, a
    date string or null -- the pipeline's imputers and encoders are what decide
    whether a given value is usable, and duplicating those rules in the request
    schema would give two places to disagree about it.
    """

    rows: list[dict[str, Any]] = Field(
        ..., min_length=1, description="One object per row, keyed by column name."
    )


class RowPrediction(BaseModel):
    """One row's answer, with the model's confidence where it has one."""

    prediction: Any
    # Probability per class, keyed by the label the user uploaded rather than by
    # the model's internal 0..n-1 encoding. Empty for regression.
    probabilities: dict[str, float] = Field(default_factory=dict)


class PredictionResponse(BaseModel):
    """Live predictions from the saved model (spec 7.11).

    ``missing_columns`` and ``unexpected_columns`` are reported rather than
    rejected. A missing column is imputed by the pipeline exactly as a missing
    value in training data would be, which is a legitimate prediction on partial
    information -- but a caller who misspelled a column name would otherwise get
    a confident answer computed from an imputed median and no hint that anything
    was wrong.
    """

    job_id: int
    model_name: str
    target_column: str
    predictions: list[RowPrediction] = Field(default_factory=list)
    missing_columns: list[str] = Field(default_factory=list)
    unexpected_columns: list[str] = Field(default_factory=list)


class ErrorResponse(BaseModel):
    detail: str
