"""Tests for the storage layer.

The key-naming tests need no network at all, and they lock in the S3 layout
early -- key strings scattered across a dozen agents are painful to change
later. The round-trip test needs MinIO and is skipped when it is not running,
so the suite stays green outside Docker.
"""

from __future__ import annotations

import uuid

import pytest

from app.core.storage import (
    artifact_key,
    download_bytes,
    model_key,
    object_exists,
    plot_key,
    presigned_url,
    raw_dataset_key,
    storage_healthy,
    upload_bytes,
)


class TestKeyNaming:
    """Pure functions -- no storage backend required."""

    def test_every_artifact_is_namespaced_by_job(self):
        keys = [
            raw_dataset_key(42),
            artifact_key(42, "cleaning_report.json"),
            plot_key(42, "correlation.png"),
            model_key(42),
        ]
        assert all(key.startswith("jobs/42/") for key in keys)

    def test_jobs_do_not_collide(self):
        assert raw_dataset_key(1) != raw_dataset_key(2)

    def test_keys_are_categorised_by_type(self):
        assert raw_dataset_key(7) == "jobs/7/raw/dataset.csv"
        assert artifact_key(7, "eda.json") == "jobs/7/artifacts/eda.json"
        assert plot_key(7, "shap.png") == "jobs/7/plots/shap.png"
        assert model_key(7) == "jobs/7/models/final_model.pkl"


requires_storage = pytest.mark.skipif(
    not storage_healthy(),
    reason="object storage unreachable (start the stack with `make up`)",
)


@requires_storage
class TestRoundTrip:
    """Integration tests against the real MinIO container."""

    def test_upload_then_download_returns_identical_bytes(self):
        key = f"tests/{uuid.uuid4()}.txt"
        payload = b"leakage-safe by construction"

        upload_bytes(key, payload, content_type="text/plain")

        assert object_exists(key) is True
        assert download_bytes(key) == payload

    def test_missing_object_reports_absent(self):
        assert object_exists(f"tests/{uuid.uuid4()}-does-not-exist") is False

    def test_presigned_url_points_at_the_object(self):
        key = f"tests/{uuid.uuid4()}.txt"
        upload_bytes(key, b"x")

        url = presigned_url(key, expires_in=60)

        assert url.startswith("http")
        assert "X-Amz-Signature" in url
