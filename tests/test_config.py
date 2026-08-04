"""Tests for configuration loading.

These verify the 12-factor promise the whole deployment story rests on: that
every setting genuinely comes from the environment, so the same image behaves
differently on your laptop and on EC2 without a code change.
"""

from __future__ import annotations

import pytest

from app.core.config import Settings, get_settings


@pytest.fixture(autouse=True)
def _clear_settings_cache():
    """get_settings is cached, so clear it around every test."""
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def test_defaults_are_local():
    settings = Settings(_env_file=None)
    assert settings.environment == "local"
    assert settings.is_local is True


def test_environment_variables_override_defaults(monkeypatch):
    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.setenv("S3_BUCKET", "prod-bucket")
    monkeypatch.setenv("MAX_UPLOAD_MB", "500")

    settings = Settings(_env_file=None)

    assert settings.environment == "production"
    assert settings.s3_bucket == "prod-bucket"
    assert settings.max_upload_mb == 500
    assert settings.is_local is False


def test_invalid_environment_is_rejected(monkeypatch):
    """A typo in ENVIRONMENT should fail loudly at startup, not silently."""
    monkeypatch.setenv("ENVIRONMENT", "prod")  # not one of the allowed values
    with pytest.raises(ValueError):
        Settings(_env_file=None)


def test_s3_endpoint_can_be_unset_for_real_aws(monkeypatch):
    """No endpoint URL means boto3 talks to real S3 -- the AWS configuration."""
    monkeypatch.delenv("S3_ENDPOINT_URL", raising=False)
    settings = Settings(_env_file=None, s3_endpoint_url=None)
    assert settings.s3_endpoint_url is None


def test_empty_s3_endpoint_env_var_selects_real_aws(monkeypatch):
    """Selecting real S3 has to be possible *from the environment*, not just in Python.

    The test above passes ``s3_endpoint_url=None`` as a keyword, which no
    deployment can do. Through the environment the only spellings available are
    "absent" -- which falls back to the MinIO default -- and "empty", which used
    to arrive as "" and be handed to boto3's endpoint_url verbatim. So empty is
    made to mean None, and this is the test that deployment relies on.
    """
    monkeypatch.setenv("S3_ENDPOINT_URL", "")
    assert Settings(_env_file=None).s3_endpoint_url is None


def test_absent_s3_endpoint_still_defaults_to_minio(monkeypatch):
    """The local default must survive the change above."""
    monkeypatch.delenv("S3_ENDPOINT_URL", raising=False)
    assert Settings(_env_file=None).s3_endpoint_url == "http://localhost:9000"


def test_get_settings_is_cached():
    assert get_settings() is get_settings()
