"""Request and response shapes for the API.

Separate from the ORM models on purpose: the database schema and the public API
contract change for different reasons, and coupling them means every internal
column rename becomes a breaking API change.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

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
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class JobSummary(BaseModel):
    id: int
    original_filename: str
    status: JobStatus
    n_rows: int | None
    n_columns: int | None
    size_bytes: int
    error_message: str | None
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)


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


class ErrorResponse(BaseModel):
    detail: str
