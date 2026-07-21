"""Artifact model.

An index of everything a job produces: reports, plots, fitted models, cleaned
datasets. Rows store the object-storage key, never the bytes (spec section 13),
so Postgres stays small and Streamlit fetches files directly from storage via
short-lived presigned URLs.

Section 1 registers the uploaded CSV. Later sections register their own outputs
through the same table, which is what lets the Results page enumerate a job's
outputs without knowing which agents happened to run.
"""

from __future__ import annotations

import enum
from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import DateTime, Enum, ForeignKey, Integer, String, UniqueConstraint, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.db import Base

if TYPE_CHECKING:
    from app.models.job import Job


class ArtifactKind(enum.StrEnum):
    RAW_DATASET = "raw_dataset"
    CLEANED_DATASET = "cleaned_dataset"
    REPORT = "report"
    PLOT = "plot"
    MODEL = "model"
    JSON = "json"


class Artifact(Base):
    __tablename__ = "artifacts"
    __table_args__ = (
        # A job cannot have two artifacts with the same name. Catches a
        # re-running agent silently overwriting its own previous output.
        UniqueConstraint("job_id", "name", name="uq_artifact_job_name"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    job_id: Mapped[int] = mapped_column(
        ForeignKey("jobs.id", ondelete="CASCADE"), nullable=False, index=True
    )

    kind: Mapped[ArtifactKind] = mapped_column(
        Enum(ArtifactKind, name="artifact_kind", values_callable=lambda e: [m.value for m in e]),
        nullable=False,
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    s3_key: Mapped[str] = mapped_column(String(512), nullable=False, unique=True)
    content_type: Mapped[str] = mapped_column(String(128), nullable=False)
    size_bytes: Mapped[int | None] = mapped_column(Integer, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    job: Mapped[Job] = relationship(back_populates="artifacts")

    def __repr__(self) -> str:
        return f"<Artifact id={self.id} job={self.job_id} name={self.name!r}>"
