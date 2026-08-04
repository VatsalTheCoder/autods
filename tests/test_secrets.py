"""Tests for the Secrets Manager settings source.

The point of this source is that it changes nothing locally and everything on
EC2, so the tests come in two halves: proof that an unconfigured process never
goes near boto3, and proof that a configured one resolves, filters and fails in
the ways the runbook promises.

boto3 is stubbed rather than mocked with moto. A fake client is enough to
exercise every branch here, and it keeps a heavyweight test-only dependency out
of the CI image -- which installs from pyproject on bare ubuntu, not from the
Dockerfile.
"""

from __future__ import annotations

import json

import pytest
from botocore.exceptions import ClientError

from app.core import secrets as secrets_module
from app.core.config import Settings, get_settings
from app.core.secrets import SecretsError


@pytest.fixture(autouse=True)
def _clean_state(monkeypatch):
    """Clear both caches and unset the trigger variable around every test."""
    monkeypatch.delenv("AWS_SECRETS_ID", raising=False)
    secrets_module.reset_cache()
    get_settings.cache_clear()
    yield
    secrets_module.reset_cache()
    get_settings.cache_clear()


class FakeSecretsClient:
    """Stands in for the Secrets Manager client, recording how it was called."""

    def __init__(self, response=None, error=None):
        self._response = response
        self._error = error
        self.calls: list[str] = []

    def get_secret_value(self, SecretId: str):  # noqa: N803 - boto3's parameter name
        self.calls.append(SecretId)
        if self._error is not None:
            raise self._error
        return self._response


def install_client(monkeypatch, client) -> dict[str, object]:
    """Patch boto3.client and record the region it was asked for."""
    seen: dict[str, object] = {}

    def fake_client(service_name: str, region_name: str | None = None, **kwargs):
        seen["service"] = service_name
        seen["region"] = region_name
        seen["constructed"] = seen.get("constructed", 0) + 1  # type: ignore[operator]
        return client

    monkeypatch.setattr(secrets_module.boto3, "client", fake_client)
    return seen


def secret_of(payload: dict) -> dict:
    return {"SecretString": json.dumps(payload)}


# ---- The local case: this must be inert --------------------------------


def test_no_secret_id_never_touches_boto3(monkeypatch):
    """The default local path must not construct a client or need credentials."""

    def explode(*args, **kwargs):
        raise AssertionError("boto3.client must not be called when AWS_SECRETS_ID is unset")

    monkeypatch.setattr(secrets_module.boto3, "client", explode)

    settings = Settings(_env_file=None)

    assert settings.aws_secrets_id is None
    assert settings.environment == "local"


def test_blank_secret_id_is_treated_as_unset(monkeypatch):
    """An empty or whitespace value is a deployment slip, not a request to fetch ''."""

    def explode(*args, **kwargs):
        raise AssertionError("a blank AWS_SECRETS_ID must not trigger a lookup")

    monkeypatch.setattr(secrets_module.boto3, "client", explode)
    monkeypatch.setenv("AWS_SECRETS_ID", "   ")

    assert Settings(_env_file=None).environment == "local"


# ---- The deployed case -------------------------------------------------


def test_secret_supplies_settings_the_environment_did_not(monkeypatch):
    client = FakeSecretsClient(secret_of({"google_api_key": "from-secret"}))
    install_client(monkeypatch, client)
    monkeypatch.setenv("AWS_SECRETS_ID", "autods/production")

    settings = Settings(_env_file=None)

    assert settings.google_api_key == "from-secret"
    assert client.calls == ["autods/production"]


def test_environment_wins_over_the_secret(monkeypatch):
    """Documented precedence: an explicit env var overrides the secret."""
    install_client(monkeypatch, FakeSecretsClient(secret_of({"google_api_key": "from-secret"})))
    monkeypatch.setenv("AWS_SECRETS_ID", "autods/production")
    monkeypatch.setenv("GOOGLE_API_KEY", "from-environment")

    assert Settings(_env_file=None).google_api_key == "from-environment"


def test_screaming_case_keys_are_accepted(monkeypatch):
    """The AWS console encourages SCREAMING_CASE; settings fields are lower case."""
    install_client(monkeypatch, FakeSecretsClient(secret_of({"GOOGLE_API_KEY": "shouty"})))
    monkeypatch.setenv("AWS_SECRETS_ID", "autods/production")

    assert Settings(_env_file=None).google_api_key == "shouty"


def test_a_password_bearing_database_url_arrives_from_the_secret(monkeypatch):
    """The other value Section 11 moves out of the environment.

    DATABASE_URL is deleted first because the development container sets it, and
    an environment variable outranks the secret by design -- without this the
    test would assert the precedence rule rather than the lookup.
    """
    monkeypatch.delenv("DATABASE_URL", raising=False)
    url = "postgresql+psycopg://autods:s3cr3t@db.internal:5432/autods"
    install_client(monkeypatch, FakeSecretsClient(secret_of({"database_url": url})))
    monkeypatch.setenv("AWS_SECRETS_ID", "autods/production")

    assert str(Settings(_env_file=None).database_url) == url


def test_keys_that_are_not_settings_are_ignored(monkeypatch):
    """A secret shared with another consumer must not break the build."""
    install_client(
        monkeypatch,
        FakeSecretsClient(secret_of({"google_api_key": "k", "grafana_token": "not-ours"})),
    )
    monkeypatch.setenv("AWS_SECRETS_ID", "autods/production")

    settings = Settings(_env_file=None)

    assert settings.google_api_key == "k"
    assert not hasattr(settings, "grafana_token")


def test_region_follows_s3_region(monkeypatch):
    install_client_seen = install_client(monkeypatch, FakeSecretsClient(secret_of({})))
    monkeypatch.setenv("AWS_SECRETS_ID", "autods/production")
    monkeypatch.setenv("S3_REGION", "eu-west-2")

    Settings(_env_file=None)

    assert install_client_seen["service"] == "secretsmanager"
    assert install_client_seen["region"] == "eu-west-2"


def test_the_secret_is_fetched_once_across_repeated_builds(monkeypatch):
    """Settings gets constructed more than once per process; the network call should not repeat."""
    client = FakeSecretsClient(secret_of({"google_api_key": "k"}))
    install_client(monkeypatch, client)
    monkeypatch.setenv("AWS_SECRETS_ID", "autods/production")

    Settings(_env_file=None)
    Settings(_env_file=None)
    Settings(_env_file=None)

    assert client.calls == ["autods/production"]


# ---- Failing loudly ----------------------------------------------------


def test_unreadable_secret_raises_rather_than_falling_back(monkeypatch):
    """Booting with the wrong database is worse than not booting."""
    error = ClientError(
        {"Error": {"Code": "AccessDeniedException", "Message": "nope"}}, "GetSecretValue"
    )
    install_client(monkeypatch, FakeSecretsClient(error=error))
    monkeypatch.setenv("AWS_SECRETS_ID", "autods/production")

    with pytest.raises(SecretsError, match="autods/production"):
        Settings(_env_file=None)


def test_malformed_json_names_the_secret(monkeypatch):
    install_client(monkeypatch, FakeSecretsClient({"SecretString": "not json{"}))
    monkeypatch.setenv("AWS_SECRETS_ID", "autods/production")

    with pytest.raises(SecretsError, match="not valid JSON"):
        Settings(_env_file=None)


def test_json_that_is_not_an_object_is_rejected(monkeypatch):
    install_client(monkeypatch, FakeSecretsClient({"SecretString": '["a", "b"]'}))
    monkeypatch.setenv("AWS_SECRETS_ID", "autods/production")

    with pytest.raises(SecretsError, match="not an object"):
        Settings(_env_file=None)


def test_binary_secret_is_rejected_with_an_explanation(monkeypatch):
    install_client(monkeypatch, FakeSecretsClient({"SecretBinary": b"\x00\x01"}))
    monkeypatch.setenv("AWS_SECRETS_ID", "autods/production")

    with pytest.raises(SecretsError, match="binary"):
        Settings(_env_file=None)
