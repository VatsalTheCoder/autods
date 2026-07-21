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
