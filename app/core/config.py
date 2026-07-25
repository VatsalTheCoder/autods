"""Application configuration.

Every setting comes from an environment variable. Nothing is hardcoded and
nothing is read from a config file checked into git. This is what lets the
same Docker image run unchanged on your laptop and on EC2 (spec section 14) --
only the environment differs.
"""

from functools import lru_cache
from typing import Literal

from pydantic import Field, PostgresDsn, RedisDsn
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Settings loaded from environment variables, with .env as a local fallback."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ---- Environment ----------------------------------------------------
    environment: Literal["local", "staging", "production"] = "local"
    debug: bool = True

    # ---- Database -------------------------------------------------------
    database_url: PostgresDsn = Field(
        default="postgresql+psycopg://autods:autods@localhost:5432/autods",
        description="SQLAlchemy connection string for PostgreSQL.",
    )

    # ---- Redis / Celery -------------------------------------------------
    redis_url: RedisDsn = Field(
        default="redis://localhost:6379/0",
        description="Redis connection, used as the Celery broker and result backend.",
    )

    # ---- Object storage -------------------------------------------------
    # Locally this points at the MinIO container. On AWS, leave s3_endpoint_url
    # unset and boto3 talks to real S3 using the instance's IAM role.
    s3_bucket: str = "autods-artifacts"
    s3_endpoint_url: str | None = Field(
        default="http://localhost:9000",
        description="Set for MinIO; leave unset (None) to use real AWS S3.",
    )
    s3_region: str = "us-east-1"
    aws_access_key_id: str | None = None
    aws_secret_access_key: str | None = None

    # ---- API ------------------------------------------------------------
    api_host: str = "0.0.0.0"  # noqa: S104 - binding all interfaces is correct in a container
    api_port: int = 8000

    # ---- Streamlit ------------------------------------------------------
    # How the UI container reaches the API container. Inside Docker this is the
    # service name ("http://api:8000"); outside it is localhost.
    api_base_url: str = "http://localhost:8000"

    # ---- Uploads --------------------------------------------------------
    max_upload_mb: int = 200

    # ---- LLM (Google AI Studio) -----------------------------------------
    # Both Gemma tiers are served by the free Google AI Studio API (spec
    # section 6.3). The key lives in Secrets Manager on AWS and in .env
    # locally -- never in the image. When it is unset the real client refuses
    # to start, which is deliberate: it forces the FakeLLM path in tests.
    google_api_key: str | None = None

    # Model ids are settings, not constants, because spec section 6.3 flags the
    # exact "Gemma 4" names as UNVERIFIED. Whatever the provider actually calls
    # the two tiers, it is one env var away -- no code change. Override both
    # before pointing the real client at live traffic.
    llm_model_small: str = "gemma-3-4b-it"
    llm_model_large: str = "gemma-3-27b-it"

    # Free-tier rate limits, per tier, per the spec's stated figures. These are
    # ALSO flagged unverified there, and the input-TPM cap is the binding
    # constraint the whole prompt-shrinking strategy (Section 9) rests on, so
    # they are env-overridable rather than hardcoded into the limiter. Confirm
    # against live provider docs before sizing anything on top of them.
    llm_small_rpm: int = 30
    llm_small_input_tpm: int = 15_000
    llm_large_rpm: int = 30
    llm_large_input_tpm: int = 15_000

    # How many times structured_complete re-asks the model after malformed or
    # schema-invalid output before failing the job (spec section 10).
    llm_max_retries: int = 3
    # How many times a single call is retried through exponential backoff after
    # the provider answers HTTP 429 (rate-limited), separate from the above.
    llm_backoff_retries: int = 5
    llm_request_timeout: int = 60

    # ---- Pipeline / modelling (Section 5) -------------------------------
    # Cross-validation folds. Five is the spec's figure (7.7) and the usual
    # bias/variance compromise; exposed because a tiny demo dataset may not have
    # enough rows per class to support five stratified folds.
    cv_folds: int = 5

    # One seed for every random choice in the pipeline -- fold shuffling and the
    # model's own randomness. Fixed and configurable so a reported score is
    # reproducible, which is the first thing an examiner will try.
    random_seed: int = 42

    # Cleaning drops a column whose values are missing more often than this.
    # A column that is 90% blank carries almost no signal but forces every row
    # through imputation, so it is removed structurally rather than filled in.
    max_null_column_rate: float = 0.6

    @property
    def is_local(self) -> bool:
        return self.environment == "local"


@lru_cache
def get_settings() -> Settings:
    """Return the settings singleton.

    Cached so the environment is parsed once per process. Tests that need
    different values should call ``get_settings.cache_clear()`` first.
    """
    return Settings()
