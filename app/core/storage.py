"""Object storage.

The single place in the codebase allowed to touch object storage. Everything
else -- agents, API routes, Celery tasks -- goes through this module and never
writes artifacts to the local filesystem.

That rule matters because containers are ephemeral and there are several of
them: a file the API writes to its own disk simply does not exist for the
Celery worker. Routing everything through S3 from day one is what makes the
Section 11 deployment a configuration change rather than a rewrite.

Locally this talks to the MinIO container; on AWS the same code talks to real
S3. The only difference is whether ``s3_endpoint_url`` is set.
"""

from __future__ import annotations

import io
import logging
from functools import lru_cache
from typing import BinaryIO

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

from app.core.config import get_settings

logger = logging.getLogger(__name__)


class StorageError(RuntimeError):
    """Raised when an object storage operation fails."""


@lru_cache
def get_s3_client():
    """Build a boto3 S3 client from settings.

    When ``aws_access_key_id`` is unset, boto3 falls back to its default
    credential chain -- which on EC2 means the instance's IAM role, so no
    secrets need to exist on the server at all.
    """
    settings = get_settings()
    return boto3.client(
        "s3",
        endpoint_url=settings.s3_endpoint_url,
        region_name=settings.s3_region,
        aws_access_key_id=settings.aws_access_key_id,
        aws_secret_access_key=settings.aws_secret_access_key,
        config=Config(
            signature_version="s3v4",
            retries={"max_attempts": 3, "mode": "standard"},
            # Fail fast when storage is simply absent. Without these, boto3's
            # 60-second defaults make the /health check and the test suite hang
            # for a minute before admitting the obvious.
            connect_timeout=3,
            read_timeout=10,
        ),
    )


def ensure_bucket() -> None:
    """Create the configured bucket if it does not already exist.

    Convenient against MinIO on a fresh machine. On AWS the bucket is created
    by deployment, and this becomes a no-op existence check.
    """
    settings = get_settings()
    client = get_s3_client()
    try:
        client.head_bucket(Bucket=settings.s3_bucket)
        logger.debug("Bucket %s already exists", settings.s3_bucket)
    except ClientError as exc:
        code = exc.response.get("Error", {}).get("Code")
        if code not in ("404", "NoSuchBucket"):
            raise StorageError(f"Could not reach bucket {settings.s3_bucket}: {exc}") from exc
        client.create_bucket(Bucket=settings.s3_bucket)
        logger.info("Created bucket %s", settings.s3_bucket)


def upload_bytes(key: str, data: bytes, content_type: str = "application/octet-stream") -> str:
    """Store raw bytes at ``key``. Returns the key."""
    settings = get_settings()
    try:
        get_s3_client().put_object(
            Bucket=settings.s3_bucket,
            Key=key,
            Body=data,
            ContentType=content_type,
        )
    except ClientError as exc:
        raise StorageError(f"Failed to upload {key}: {exc}") from exc
    logger.info("Uploaded %s (%d bytes)", key, len(data))
    return key


def upload_fileobj(key: str, fileobj: BinaryIO, content_type: str | None = None) -> str:
    """Stream a file-like object to ``key`` without loading it fully into memory."""
    settings = get_settings()
    extra = {"ContentType": content_type} if content_type else None
    try:
        get_s3_client().upload_fileobj(fileobj, settings.s3_bucket, key, ExtraArgs=extra)
    except ClientError as exc:
        raise StorageError(f"Failed to upload {key}: {exc}") from exc
    logger.info("Uploaded %s (streamed)", key)
    return key


def download_bytes(key: str) -> bytes:
    """Fetch the object at ``key`` as bytes."""
    settings = get_settings()
    buffer = io.BytesIO()
    try:
        get_s3_client().download_fileobj(settings.s3_bucket, key, buffer)
    except ClientError as exc:
        raise StorageError(f"Failed to download {key}: {exc}") from exc
    return buffer.getvalue()


def object_exists(key: str) -> bool:
    """Return whether an object exists at ``key``."""
    settings = get_settings()
    try:
        get_s3_client().head_object(Bucket=settings.s3_bucket, Key=key)
        return True
    except ClientError:
        return False


def presigned_url(key: str, expires_in: int = 3600) -> str:
    """Return a temporary URL for reading ``key``.

    Lets Streamlit display plots and download reports straight from storage
    without proxying bytes through the API, and without the bucket being public.
    """
    settings = get_settings()
    try:
        return get_s3_client().generate_presigned_url(
            "get_object",
            Params={"Bucket": settings.s3_bucket, "Key": key},
            ExpiresIn=expires_in,
        )
    except ClientError as exc:
        raise StorageError(f"Failed to sign {key}: {exc}") from exc


def storage_healthy() -> bool:
    """Cheap reachability check, used by the /health endpoint."""
    try:
        get_s3_client().head_bucket(Bucket=get_settings().s3_bucket)
        return True
    except Exception:
        return False


# ---- Key naming -------------------------------------------------------------
# Every artifact lives under jobs/{job_id}/. Keeping key construction here means
# the layout is defined once rather than string-formatted across the codebase.


def raw_dataset_key(job_id: int) -> str:
    return f"jobs/{job_id}/raw/dataset.csv"


def artifact_key(job_id: int, name: str) -> str:
    return f"jobs/{job_id}/artifacts/{name}"


def plot_key(job_id: int, name: str) -> str:
    return f"jobs/{job_id}/plots/{name}"


def model_key(job_id: int, name: str = "final_model.pkl") -> str:
    return f"jobs/{job_id}/models/{name}"
