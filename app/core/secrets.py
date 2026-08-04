"""Reading secrets from AWS Secrets Manager (spec section 14, BUILD_PLAN section 11).

The deployment story is "the same image, a different environment". That works
for ordinary settings because they are environment variables, but it breaks for
the two values that must not sit in an environment variable a `docker inspect`
can print: the Google API key and the database password.

So this adds one more settings source, below the environment rather than above
it. On EC2 you set ``AWS_SECRETS_ID`` to the name of a secret holding a JSON
object, and its keys fill in any setting the environment did not already
provide. Locally ``AWS_SECRETS_ID`` is unset, nothing here runs, and there is no
boto3 call, no network, and no credential to have.

The precedence is deliberate. An explicit environment variable wins over the
secret, so a one-off override on the box does not require editing a secret and
redeploying -- and so that a test can set an environment variable and be certain
of it. The cost is that a stale environment variable can mask a rotated secret;
that is the trade the runbook's rotation step calls out.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

import boto3
from botocore.exceptions import BotoCoreError, ClientError
from pydantic.fields import FieldInfo
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource

logger = logging.getLogger(__name__)

# The environment variable naming the secret. Not a Settings field, because this
# source has to run *before* Settings exists in order to help build it.
SECRETS_ID_ENV = "AWS_SECRETS_ID"
REGION_ENV = "S3_REGION"
DEFAULT_REGION = "us-east-1"


class SecretsError(RuntimeError):
    """Raised when a secret was asked for by name and could not be resolved.

    Deliberately fatal. If ``AWS_SECRETS_ID`` is set then someone intended this
    process to be configured from Secrets Manager, and falling back to whatever
    happens to be in the environment would start the app with the wrong database
    or no API key -- failures that surface much later and much less clearly than
    refusing to boot.
    """


# Cached because Settings can be constructed more than once per process (tests
# clear ``get_settings``' cache) and each construction would otherwise be a
# network round trip. Keyed by secret id and region so a test can point at a
# different one without poisoning the entry for the real one.
_bundle_cache: dict[tuple[str, str], dict[str, Any]] = {}


def reset_cache() -> None:
    """Forget any fetched secret. For tests; nothing in the app calls it."""
    _bundle_cache.clear()


def _fetch(secret_id: str, region: str) -> dict[str, Any]:
    """Fetch and parse the secret, or raise ``SecretsError`` explaining which step failed."""
    key = (secret_id, region)
    if key in _bundle_cache:
        return _bundle_cache[key]

    client = boto3.client("secretsmanager", region_name=region)
    try:
        response = client.get_secret_value(SecretId=secret_id)
    except (BotoCoreError, ClientError) as exc:
        # The common causes are all operator errors with distinct fixes -- wrong
        # name, no IAM permission, wrong region -- and boto3's message names
        # which, so it is worth keeping rather than flattening.
        raise SecretsError(f"Could not read secret {secret_id!r} in {region}: {exc}") from exc

    # Binary secrets are a valid Secrets Manager feature and not one this uses.
    payload = response.get("SecretString")
    if payload is None:
        raise SecretsError(
            f"Secret {secret_id!r} holds binary data; AutoDS expects a JSON object of settings."
        )

    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise SecretsError(f"Secret {secret_id!r} is not valid JSON: {exc}") from exc

    if not isinstance(parsed, dict):
        raise SecretsError(
            f"Secret {secret_id!r} is a JSON {type(parsed).__name__}, not an object of settings."
        )

    # Settings fields are lower case; the AWS console encourages SCREAMING_CASE.
    # Accept both rather than making the casing a thing that can be got wrong.
    bundle = {str(name).lower(): value for name, value in parsed.items()}
    _bundle_cache[key] = bundle
    logger.info("Loaded %d setting(s) from Secrets Manager secret %s", len(bundle), secret_id)
    return bundle


class AwsSecretsManagerSource(PydanticBaseSettingsSource):
    """A pydantic-settings source backed by one JSON secret.

    Returns the whole bundle at once from ``__call__`` rather than answering
    field by field, so the secret is fetched once per settings build instead of
    once per field.
    """

    def get_field_value(self, field: FieldInfo, field_name: str) -> tuple[Any, str, bool]:
        # Required by the base class. Unused: ``__call__`` supplies everything,
        # which is the documented way to write a bulk source.
        return None, field_name, False

    def __call__(self) -> dict[str, Any]:
        secret_id = os.environ.get(SECRETS_ID_ENV, "").strip()
        if not secret_id:
            return {}
        region = os.environ.get(REGION_ENV, "").strip() or DEFAULT_REGION
        bundle = _fetch(secret_id, region)
        # Drop anything that is not a field on the model. A secret shared with
        # another consumer may carry keys that are not settings, and
        # ``extra="ignore"`` would tolerate them anyway -- but filtering here
        # keeps the source honest about what it contributed.
        known = set(self.settings_cls.model_fields)
        return {name: value for name, value in bundle.items() if name in known}


def secrets_source(settings_cls: type[BaseSettings]) -> AwsSecretsManagerSource:
    """Build the source. A function so ``config`` does not import the class directly."""
    return AwsSecretsManagerSource(settings_cls)
