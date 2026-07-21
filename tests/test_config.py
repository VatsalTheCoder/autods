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


def test_get_settings_is_cached():
    assert get_settings() is get_settings()
