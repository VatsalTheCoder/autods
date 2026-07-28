"""Retrievable passages from a finished run, and their embeddings (spec 7.13).

One row per passage of the run's written output -- the report, the EDA findings,
the cluster profiles, the SHAP results, the critic's review. This is what "RAG
over reports, EDA, SHAP, evaluation, and critic text" cashes out to.

**The vector store is Postgres.** The spec locks ChromaDB and documents pgvector
as the fallback; this project takes the fallback, because Section 11 has to give
ChromaDB a persistent EBS volume of its own and pgvector removes that problem by
removing the service. The trade is real but small at this scale: pgvector's
exact search over a few dozen rows is faster than an approximate index would be.

Chunks belong to a job and die with it (``ondelete=CASCADE``). A run's passages
describe that run's numbers, so keeping them after the job is gone would leave
retrievable text about a job nobody can look up.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from pgvector.sqlalchemy import Vector
from sqlalchemy import DateTime, ForeignKey, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.db import Base

if TYPE_CHECKING:
    from app.models.job import Job

# BAAI/bge-small-en-v1.5's output width. Fixed rather than configured: pgvector
# needs a declared width, and a column holding two models' vectors would return
# nonsense instead of an error.
EMBEDDING_DIMENSIONS = 384


class RunChunk(Base):
    """One retrievable passage of a run's written output."""

    __tablename__ = "run_chunks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    job_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("jobs.id", ondelete="CASCADE"), index=True, nullable=False
    )

    # The artifact this came from, so an answer can cite its source.
    source: Mapped[str] = mapped_column(String(64), nullable=False)
    heading: Mapped[str] = mapped_column(String(255), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    ordinal: Mapped[int] = mapped_column(Integer, nullable=False)

    embedding: Mapped[list[float]] = mapped_column(Vector(EMBEDDING_DIMENSIONS), nullable=False)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    job: Mapped[Job] = relationship(back_populates="chunks")

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<RunChunk job={self.job_id} source={self.source!r} heading={self.heading!r}>"
