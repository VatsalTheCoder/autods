"""Registering and loading JSON artifacts -- one helper, used by every agent.

Every agent from here on produces a JSON report (schema, cleaning, evaluation,
critic...). Each one needs the same two-step dance: write the bytes to object
storage, and record a row in ``artifacts`` pointing at them (spec 13 -- Postgres
holds keys, S3 holds bytes). Doing it in one place keeps that invariant -- a row
always has a real object behind it -- from being re-implemented slightly
differently a dozen times.

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


def register_json_artifact(
    db: Session,
    job_id: int,
    name: str,
    payload: BaseModel | dict,
    *,
    kind: ArtifactKind = ArtifactKind.JSON,
) -> Artifact:
    """Store ``payload`` as JSON in S3 and upsert its ``artifacts`` row.

    Added to the session but not committed -- the caller owns the transaction,
    so the artifact is persisted or rolled back atomically with its job.
    """
    body = payload.model_dump_json() if isinstance(payload, BaseModel) else json.dumps(payload)
    data = body.encode("utf-8")
    key = artifact_key(job_id, name)
    upload_bytes(key, data, content_type="application/json")

    existing = db.execute(
        select(Artifact).where(Artifact.job_id == job_id, Artifact.name == name)
    ).scalar_one_or_none()

    if existing is not None:
        existing.s3_key = key
        existing.content_type = "application/json"
        existing.size_bytes = len(data)
        existing.kind = kind
        artifact = existing
    else:
        artifact = Artifact(
            job_id=job_id,
            kind=kind,
            name=name,
            s3_key=key,
            content_type="application/json",
            size_bytes=len(data),
        )
        db.add(artifact)

    db.flush()
    logger.info("Registered JSON artifact %s for job %s (%d bytes)", name, job_id, len(data))
    return artifact


def load_json_artifact(db: Session, job_id: int, name: str) -> dict | None:
    """Read a JSON artifact back, or None if the job never produced it."""
    artifact = db.execute(
        select(Artifact).where(Artifact.job_id == job_id, Artifact.name == name)
    ).scalar_one_or_none()
    if artifact is None:
        return None
    return json.loads(download_bytes(artifact.s3_key))
