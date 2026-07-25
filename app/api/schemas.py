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


class ErrorResponse(BaseModel):
    detail: str
