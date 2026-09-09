"""Application configuration.

Every setting comes from an environment variable. Nothing is hardcoded and
nothing is read from a config file checked into git. This is what lets the
same Docker image run unchanged on your laptop and on EC2 (spec section 14) --
only the environment differs.
"""

from functools import lru_cache
from typing import Literal

from pydantic import Field, PostgresDsn, RedisDsn, field_validator
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict

from app.core.secrets import secrets_source


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
    # Locally this points at the MinIO container. On AWS, set S3_ENDPOINT_URL to
    # the empty string -- see the validator below for why "unset" is not the way
    # to say that -- and boto3 talks to real S3 using the instance's IAM role.
    s3_bucket: str = "autods-artifacts"
    s3_endpoint_url: str | None = Field(
        default="http://localhost:9000",
        description="Set for MinIO; set empty (or None) to use real AWS S3.",
    )

    @field_validator("s3_endpoint_url", mode="before")
    @classmethod
    def _empty_endpoint_means_real_s3(cls, value: object) -> object:
        """Treat an empty S3_ENDPOINT_URL as "unset", so real S3 is reachable from the environment.

        The default here is MinIO, which is right locally and wrong everywhere
        else. Every other setting can be changed by exporting a variable, but
        "no endpoint" had no spelling: an unset variable falls back to the MinIO
        default, and an empty one used to arrive as "" and go straight into
        boto3's ``endpoint_url``, which rejects it. That left no way to select
        real S3 without editing code -- the one thing spec section 14 promises
        deployment would not need.
        """
        if isinstance(value, str) and not value.strip():
            return None
        return value

    s3_region: str = "us-east-1"
    aws_access_key_id: str | None = None
    aws_secret_access_key: str | None = None

    # ---- Secrets Manager ------------------------------------------------
    # The name of a Secrets Manager secret holding a JSON object of settings --
    # in practice google_api_key and database_url, the two values that must not
    # be readable from `docker inspect`. Unset locally, which disables the
    # lookup entirely. Declared here (rather than only read from the
    # environment) so it is visible in the settings object and so extra="ignore"
    # does not silently swallow a typo'd spelling of it.
    aws_secrets_id: str | None = Field(
        default=None,
        description="Secrets Manager secret name; unset means configure from the environment only.",
    )

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

    # Model ids are settings, not constants, because spec 6.3 flags its own
    # names as UNVERIFIED and says to override them with whatever the provider
    # actually serves. These are the result of doing that, checked against live
    # AI Studio on 2026-07-27: the spec's gemma-3-4b-it and gemma-3-27b-it are
    # both gone (404), and the Gemma family there is now Gemma 4, whose smallest
    # member is a 26B MoE.
    #
    # That size difference is why the tiers are Gemini rather than Gemma. SMALL
    # is meant to be cheap and fast -- the spec picked a *4B* model for it -- and
    # is called once per agent on the request path. Measured on the feature
    # strategy prompt: gemini-3.1-flash-lite answered in 1.6s and succeeded 3
    # times in 3; gemma-4-26b-a4b-it took 42s and failed 4 times in 5 against the
    # 60s timeout, silently degrading every agent to its deterministic fallback.
    #
    # Re-measured 2026-07-31, and the second half of that no longer holds. On a
    # critic-sized prompt (~3k input tokens, JSON out) gemma-4-31b-it answered in
    # 13.8-16.3s and returned schema-valid JSON 4 times out of 4. The variant
    # that misbehaves is the *26B MoE*, which now answers in ~17s but returns an
    # empty body on two calls in three -- so the 2026-07-27 finding was a
    # property of that model, not of Gemma. 31B was not tried then; it is what
    # LLM_MODEL_FALLBACK points at now. It remains too slow for SMALL, which
    # sits on the request path and needs seconds, not tens of them.
    #
    # Anything here is one env var away, so a self-hosted or Gemma deployment
    # needs no code change -- which is the property spec 6.3 was protecting.
    llm_model_small: str = "gemini-3.1-flash-lite"
    # Unused until the Critic and Chat agents arrive (Sections 9, 10); verified
    # reachable now so it is not a surprise then. gemini-3.1-pro-preview was the
    # obvious alternative and is rate-limited to unusable on the free tier.
    llm_model_large: str = "gemini-3.5-flash"
    # A different model *family* on the same key, tried only when Gemini is
    # unavailable rather than merely busy. See ModelTier.FALLBACK for why a third
    # Gemini model would not have helped. Its published free-tier limits differ
    # from the Gemini ones, hence its own limiter below.
    llm_model_fallback: str = "gemma-4-31b-it"

    # Free-tier rate limits, per tier, per the spec's stated figures. These are
    # ALSO flagged unverified there, and the input-TPM cap is the binding
    # constraint the whole prompt-shrinking strategy (Section 9) rests on, so
    # they are env-overridable rather than hardcoded into the limiter. Confirm
    # against live provider docs before sizing anything on top of them.
    llm_small_rpm: int = 30
    llm_small_input_tpm: int = 15_000
    llm_large_rpm: int = 30
    llm_large_input_tpm: int = 15_000
    # These two are the only ones here that have been read off a live quota
    # dashboard rather than taken from the spec: Gemma 4 31B on AI Studio,
    # 2026-07-31. Its daily cap (14,400) is far out of reach at six calls a run
    # and is not metered here for that reason.
    llm_fallback_rpm: int = 30
    llm_fallback_input_tpm: int = 15_000

    # How many times structured_complete re-asks the model after malformed or
    # schema-invalid output before failing the job (spec section 10).
    llm_max_retries: int = 3
    # How many times a single call is retried through exponential backoff after
    # the provider answers HTTP 429 (rate-limited), separate from the above.
    llm_backoff_retries: int = 5
    llm_request_timeout: int = 60

    # The ceiling on everything a retried call may add: the sleeps *and* the
    # failed attempts. It exists because retrying a request that timed out is
    # not free the way retrying a rate-limit rejection is -- a 429 comes back in
    # milliseconds, a dead endpoint costs a full llm_request_timeout every time.
    # Without a bound, one unresponsive provider turns a five-retry policy into
    # five minutes of a pipeline node doing nothing.
    #
    # 90s against a 60s timeout means a slow failure is retried once and then
    # given up on, while a fast one (a 503 returned immediately) still gets the
    # full retry count. One number, and the *shape* of the failure decides how
    # many attempts it earns -- which is the behaviour we want in both cases.
    #
    # The cost, stated plainly: against a provider that is *entirely* down, each
    # LLM-calling node now spends up to this budget plus one final attempt before
    # falling back, so a pipeline run takes a few minutes longer than it used to
    # before reaching the same deterministic result. That is the price of
    # surviving a blip rather than degrading on one, and it is bounded -- which
    # the previous behaviour of "never retry" was not trading against anything.
    # A re-ask for malformed JSON does not multiply it: that path only runs when
    # the provider answered, and a transient failure propagates instead of being
    # re-asked (see ``llm/structured.py``).
    llm_retry_budget_seconds: int = 90

    # ---- Pipeline / modelling (Section 5) -------------------------------
    # Cross-validation folds. Five is the spec's figure (7.7) and the usual
    # bias/variance compromise; exposed because a tiny demo dataset may not have
    # enough rows per class to support five stratified folds.
    cv_folds: int = 5

    # One seed for every random choice in the pipeline -- fold shuffling and the
    # model's own randomness. Fixed and configurable so a reported score is
    # reproducible, which is the first thing an examiner will try.
    random_seed: int = 42

    # ---- The optional steps the planner switches on (Section 7) ----------
    # Above this many rows, a planned sampling step actually subsamples; at or
    # below it, every row is used. Four models times five folds is twenty fits,
    # so the ceiling is about keeping a demo responsive rather than about
    # statistics -- 20k rows is already far more than these models need to
    # separate signal from noise on a tabular dataset.
    max_modelling_rows: int = 20_000

    # How many features a planned selection step keeps. Clamped to the number
    # actually available, and if it still exceeds the encoded width scikit-learn
    # keeps everything rather than failing.
    feature_selection_k: int = 20

    # Cleaning drops a column whose values are missing more often than this.
    # A column that is 90% blank carries almost no signal but forces every row
    # through imputation, so it is removed structurally rather than filled in.
    max_null_column_rate: float = 0.6

    # ---- Final training & SHAP (Section 8) -------------------------------
    # How many rows SHAP explains. TreeExplainer is exact but its cost grows
    # with rows x features x trees, and a global importance ranking is stable
    # long before every row is used -- the mean absolute SHAP value over 500
    # rows is not meaningfully different from the one over 50,000. The rows are
    # a stratified sample where that is possible, and the report says so rather
    # than implying the whole dataset was explained.
    shap_max_rows: int = 500

    # How many source columns the importance chart and report table show. Past
    # about fifteen bars a reader is skimming, not reading.
    shap_top_features: int = 15
    # Individual predictions explained in full. One per class (or the extremes
    # of a regression) is enough to show the mechanism; a page of waterfalls is
    # not more convincing than three.
    shap_local_examples: int = 3

    # ---- Critic & report (Section 9) -------------------------------------
    # How many tokens a summarised run may occupy in one prompt. Derived, not
    # picked: the free tier's binding limit is 15,000 *input tokens per minute*,
    # and the Critic and Report both run inside the same minute. Two prompts at
    # this size leave room for the system prompts, the schema hints and the
    # critic's findings being passed on to the report -- and still sit under the
    # cap with margin, which matters because the estimate is a 4-chars-per-token
    # approximation rather than the provider's own count.
    #
    # Measured for scale: the raw artifacts of a 121-column run come to ~24,000
    # tokens, so this is not a comfortable ceiling that never binds. It is the
    # reason `agents/summaries.py` exists.
    llm_prompt_budget_tokens: int = 6000

    # ---- EDA & clustering (Section 6) -----------------------------------
    # The range of k the silhouette search tries. Two is the smallest number of
    # groups that means anything; beyond about eight, "natural groupings" stop
    # being something an LLM can describe usefully or a scatter plot can show.
    cluster_k_min: int = 2
    cluster_k_max: int = 8

    # Clustering every row of a large dataset to pick k is wasted work -- the
    # silhouette score stabilises long before then. Above this, k selection runs
    # on a random sample and the chosen k is applied to everything.
    cluster_sample_size: int = 5_000

    # Charts are rendered headlessly to PNG. 110 DPI is legible on a laptop
    # screen without producing megabyte images the API has to stream.
    plot_dpi: int = 110
    # A correlation heatmap of 80 columns is an unreadable smear; cap what is
    # drawn and say so in the report rather than emitting something useless.
    max_heatmap_columns: int = 25
    # How many histograms/boxplots to draw before stopping. A 200-column dataset
    # does not need 200 charts on a results page.
    max_distribution_plots: int = 12

    @property
    def is_local(self) -> bool:
        return self.environment == "local"

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """Add Secrets Manager as the lowest-priority source.

        Order is highest priority first. The secret sits below the environment
        and below .env deliberately -- see ``app/core/secrets`` for why an
        explicit environment variable is allowed to win. When AWS_SECRETS_ID is
        unset the extra source returns an empty dict without touching boto3, so
        this costs local runs nothing.
        """
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            secrets_source(settings_cls),
            file_secret_settings,
        )


@lru_cache
def get_settings() -> Settings:
    """Return the settings singleton.

    Cached so the environment is parsed once per process. Tests that need
    different values should call ``get_settings.cache_clear()`` first.
    """
    return Settings()
