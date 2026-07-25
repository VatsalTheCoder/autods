"""Registering and loading artifacts -- and the one list of what they are called.

Every agent from here on produces an output (schema, cleaning, evaluation,
critic...). Each one needs the same two-step dance: write the bytes to object
storage, and record a row in ``artifacts`` pointing at them (spec 13 -- Postgres
holds keys, S3 holds bytes). Doing it in one place keeps that invariant -- a row
always has a real object behind it -- from being re-implemented slightly
differently a dozen times.

The artifact *names* live here too rather than next to whichever code writes
them, because they are a shared vocabulary: the API writes ``schema_report.json``
and the worker reads it, the worker writes ``evaluation_report.json`` and the API
serves it. A name defined next to one of those two would leave the other
importing across a layer boundary just to spell a filename.

Writes are idempotent: re-running an agent overwrites its previous artifact at
the same key and updates the row, rather than colliding with the
``(job_id, name)`` uniqueness constraint.
"""

from __future__ import annotations

import json
import logging

from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.storage import artifact_key, download_bytes, upload_bytes
from app.models.artifact import Artifact, ArtifactKind

logger = logging.getLogger(__name__)

# ---- Artifact names ---------------------------------------------------------
# Written by the API during upload and the human checkpoint (Sections 1 & 3).
SCHEMA_ARTIFACT = "schema_report.json"
CONFIRMED_SCHEMA_ARTIFACT = "confirmed_schema.json"

# Written by the pipeline nodes (Section 5). ``preprocessing_pipeline.pkl`` is
# the unfitted recipe -- the spec calls it out by name (7.6) because being able
# to load it back and confirm it is unfitted is the evidence behind the whole
# leakage claim.
PLANNER_ARTIFACT = "planner_report.json"
CLEANING_ARTIFACT = "cleaning_report.json"
CLEANED_DATASET_ARTIFACT = "cleaned_dataset.csv"
PREPROCESSING_ARTIFACT = "preprocessing_report.json"
PREPROCESSOR_ARTIFACT = "preprocessing_pipeline.pkl"
EVALUATION_ARTIFACT = "evaluation_report.json"
REPORT_ARTIFACT = "report.md"


def register_bytes_artifact(
    db: Session,
    job_id: int,
    name: str,
    data: bytes,
    *,
    content_type: str,
    kind: ArtifactKind,
) -> Artifact:
    """Store raw ``data`` in S3 and upsert its ``artifacts`` row.

    The general form: CSVs, pickles and Markdown all come through here.
    Added to the session but not committed -- the caller owns the transaction,
    so the artifact is persisted or rolled back atomically with its job.
    """
    key = artifact_key(job_id, name)
    upload_bytes(key, data, content_type=content_type)

    existing = db.execute(
        select(Artifact).where(Artifact.job_id == job_id, Artifact.name == name)
    ).scalar_one_or_none()

    if existing is not None:
        existing.s3_key = key
        existing.content_type = content_type
        existing.size_bytes = len(data)
        existing.kind = kind
        artifact = existing
    else:
        artifact = Artifact(
            job_id=job_id,
            kind=kind,
            name=name,
            s3_key=key,
            content_type=content_type,
            size_bytes=len(data),
        )
        db.add(artifact)

    db.flush()
    logger.info("Registered artifact %s for job %s (%d bytes)", name, job_id, len(data))
    return artifact


def register_json_artifact(
    db: Session,
    job_id: int,
    name: str,
    payload: BaseModel | dict,
    *,
    kind: ArtifactKind = ArtifactKind.JSON,
) -> Artifact:
    """Store ``payload`` as JSON in S3 and upsert its ``artifacts`` row."""
    body = payload.model_dump_json() if isinstance(payload, BaseModel) else json.dumps(payload)
    return register_bytes_artifact(
        db,
        job_id,
        name,
        body.encode("utf-8"),
        content_type="application/json",
        kind=kind,
    )


def load_json_artifact(db: Session, job_id: int, name: str) -> dict | None:
    """Read a JSON artifact back, or None if the job never produced it."""
    data = load_artifact_bytes(db, job_id, name)
    return None if data is None else json.loads(data)


def load_artifact_bytes(db: Session, job_id: int, name: str) -> bytes | None:
    """Read any artifact's raw bytes back, or None if the job never produced it."""
    found = load_artifact_content(db, job_id, name)
    return None if found is None else found[0]


def load_artifact_content(db: Session, job_id: int, name: str) -> tuple[bytes, str] | None:
    """Read an artifact's bytes *and* its recorded content type.

    The content type comes from the row rather than being guessed from the file
    extension, so whatever an agent declared when it registered the artifact is
    what a browser is eventually told -- one source of truth for "what is this".
    """
    artifact = db.execute(
        select(Artifact).where(Artifact.job_id == job_id, Artifact.name == name)
    ).scalar_one_or_none()
    if artifact is None:
        return None
    return download_bytes(artifact.s3_key), artifact.content_type
