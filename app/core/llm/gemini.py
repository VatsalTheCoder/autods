"""The real client: the two model tiers served by the free Google AI Studio API.

Everything provider-specific lives here and nowhere else. Agents depend on the
``LLMClient`` interface; only this file knows the request shape, the token-usage
field names, or that some models have no system role. Swapping to a self-hosted
Ollama endpoint (the spec's portability fallback) would be a second file
implementing the same interface, touching no agent.

Which models sit behind the tiers is a setting, not a fact about this file --
see ``config.py``, which explains why the defaults are Gemini rather than the
Gemma names the spec used. The quirks below are kept because they are what makes
that swap safe: the client works against either family.

Two open-model quirks it absorbs so callers never think about them:

* **No system role.** Gemma rejects a dedicated system turn, so any system
  messages are folded into the first user turn. Harmless for models that do
  support one, which is why it is done unconditionally.
* **Free-tier caps.** Every call clears this tier's ``RateLimiter`` first
  (proactive) and is wrapped in ``retry_with_backoff`` (reactive), so the two
  free-tier limits are respected without the caller doing anything.

**Failures are classified before they leave this file**, and that is the point of
the classification: an agent decides whether to fall back by catching
``LLMError``, so anything reaching it as a bare exception gets recorded as an
"unexpected error" and reads like a bug in this codebase. A 503 is not. So the
two retryable families -- rate limiting and transient provider failure -- become
``RateLimitError`` and ``TransientLLMError`` once their retry budget is spent,
and everything else propagates untouched for the graph's failure path.

The SDK is imported lazily in the constructor, not at module load. That keeps
``app.core.llm`` importable -- and the FakeLLM-based tests runnable -- on a
machine that never installed ``google-genai``, matching the project's "green
outside the container" rule.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Sequence

from app.core.config import Settings, get_settings
from app.core.llm.base import (
    ChatMessage,
    GenerationParams,
    LLMClient,
    LLMConfigError,
    LLMResponse,
    ModelTier,
    RateLimitError,
    Role,
    TransientLLMError,
    UsageCallback,
)
from app.core.llm.rate_limit import RateLimiter, estimate_input_tokens, retry_with_backoff

logger = logging.getLogger(__name__)

# HTTP statuses that mean the provider failed, not the request. 429 is handled
# separately: it is retryable too, but it has its own type, its own proactive
# defence and its own message.
_RETRYABLE_STATUS = frozenset({500, 502, 503, 504})

# Transport failures, matched by class name so this module does not take a direct
# dependency on httpx -- see ``_is_transient``.
_RETRYABLE_TRANSPORT_ERRORS = frozenset(
    {
        "TimeoutException",
        "ConnectTimeout",
        "ReadTimeout",
        "WriteTimeout",
        "PoolTimeout",
        "ConnectError",
        "ReadError",
        "RemoteProtocolError",
    }
)


class GeminiLLM(LLMClient):
    """Talks to both Gemma tiers over the Google AI Studio API."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        settings: Settings | None = None,
        sleep_fn: Callable[[float], None] = time.sleep,
        time_fn: Callable[[], float] = time.monotonic,
    ) -> None:
        settings = settings or get_settings()
        api_key = api_key or settings.google_api_key
        if not api_key:
            raise LLMConfigError(
                "GOOGLE_API_KEY is not set. The real client cannot start without it; "
                "use FakeLLM in tests, or set the key in .env / Secrets Manager."
            )

        try:
            from google import genai
            from google.genai import errors, types
        except ImportError as exc:  # pragma: no cover - dependency is declared
            raise LLMConfigError(
                "google-genai is not installed. Run `pip install -e .` (it is a "
                "declared dependency) or use FakeLLM."
            ) from exc

        self._types = types
        self._errors = errors
        self._client = genai.Client(api_key=api_key)
        self._models = {
            ModelTier.SMALL: settings.llm_model_small,
            ModelTier.LARGE: settings.llm_model_large,
        }
        # One limiter per tier: the two models meter on separate buckets
        # (spec section 6.3), so sharing one would throttle needlessly.
        self._limiters = {
            ModelTier.SMALL: RateLimiter(
                requests_per_minute=settings.llm_small_rpm,
                input_tokens_per_minute=settings.llm_small_input_tpm,
                time_fn=time_fn,
                sleep_fn=sleep_fn,
            ),
            ModelTier.LARGE: RateLimiter(
                requests_per_minute=settings.llm_large_rpm,
                input_tokens_per_minute=settings.llm_large_input_tpm,
                time_fn=time_fn,
                sleep_fn=sleep_fn,
            ),
        }
        self._backoff_retries = settings.llm_backoff_retries
        self._timeout_ms = settings.llm_request_timeout * 1000
        self._retry_budget = float(settings.llm_retry_budget_seconds)
        self._sleep_fn = sleep_fn
        self._time_fn = time_fn

    def complete(
        self,
        messages: Sequence[ChatMessage],
        *,
        tier: ModelTier = ModelTier.SMALL,
        params: GenerationParams | None = None,
        on_usage: UsageCallback | None = None,
    ) -> LLMResponse:
        params = params or GenerationParams()
        model = self._models[tier]
        contents = self._to_contents(messages)

        # Throttle proactively on the input-token estimate before sending.
        estimated = estimate_input_tokens("\n".join(m.content for m in messages))
        self._limiters[tier].acquire(estimated)

        config = self._types.GenerateContentConfig(
            temperature=params.temperature,
            max_output_tokens=params.max_output_tokens,
            http_options=self._types.HttpOptions(timeout=self._timeout_ms),
        )

        def _call():
            return self._client.models.generate_content(
                model=model, contents=contents, config=config
            )

        try:
            raw = retry_with_backoff(
                _call,
                retries=self._backoff_retries,
                is_retryable=self._is_retryable,
                sleep_fn=self._sleep_fn,
                time_fn=self._time_fn,
                budget_seconds=self._retry_budget,
            )
        except Exception as exc:
            # Both retryable categories are translated into this layer's own
            # exception types on the way out. That is what lets every agent's
            # `except LLMError` branch report *why* it fell back: without it a
            # provider outage arrives as a bare exception and is recorded as an
            # "unexpected error", which reads like a defect in this codebase.
            if self._is_rate_limited(exc):
                logger.warning("Rate limited by %s after %d retries", model, self._backoff_retries)
                raise RateLimitError(
                    f"rate limited after {self._backoff_retries} backoff retries"
                ) from exc
            if self._is_transient(exc):
                logger.warning(
                    "%s was unavailable after retries within a %.0fs budget: %s",
                    model,
                    self._retry_budget,
                    exc,
                )
                raise TransientLLMError(
                    f"{model} was unavailable ({type(exc).__name__}: {exc}). Retries "
                    f"were exhausted within the {self._retry_budget:.0f}s budget; the "
                    "caller will fall back."
                ) from exc
            raise

        usage = getattr(raw, "usage_metadata", None)
        response = LLMResponse(
            text=raw.text or "",
            model=model,
            tier=tier,
            input_tokens=getattr(usage, "prompt_token_count", 0) or 0,
            output_tokens=getattr(usage, "candidates_token_count", 0) or 0,
            estimated=False,
        )
        if on_usage is not None:
            on_usage(response)
        return response

    def _is_rate_limited(self, exc: Exception) -> bool:
        return isinstance(exc, self._errors.APIError) and getattr(exc, "code", None) == 429

    def _is_transient(self, exc: Exception) -> bool:
        """Failures that say "the provider had a moment", not "your request is wrong".

        Two families, and the second is the one that actually bites here:

        * **Server-side 5xx.** 500, 502, 503 and 504 are the provider's own
          statement that the failure is on its side. A 4xx is not: retrying a 400
          re-sends the same malformed request, and retrying a 403 re-presents the
          same rejected key.
        * **Timeouts and dropped connections.** These surface from the transport
          rather than the SDK -- ``google-genai`` runs on httpx, so a request that
          outruns ``llm_request_timeout`` raises ``httpx.ReadTimeout``, not an
          ``APIError`` with a code. Matched by class name rather than by importing
          httpx, which keeps this module's only hard dependency the SDK it already
          imports lazily, and keeps it working if the SDK's transport ever changes.
        """
        if isinstance(exc, self._errors.APIError):
            return getattr(exc, "code", None) in _RETRYABLE_STATUS
        # TimeoutException covers Read/Write/Connect/Pool; ConnectError and
        # RemoteProtocolError cover a connection refused or cut mid-response.
        return type(exc).__name__ in _RETRYABLE_TRANSPORT_ERRORS

    def _is_retryable(self, exc: Exception) -> bool:
        return self._is_rate_limited(exc) or self._is_transient(exc)

    def _to_contents(self, messages: Sequence[ChatMessage]) -> list:
        """Translate provider-neutral messages into Gemma's content list.

        Gemma has no system role, so system text is prepended to the first user
        turn (creating one if the conversation opens with none). Assistant turns
        map to the "model" role the API expects.
        """
        system_text = "\n\n".join(m.content for m in messages if m.role is Role.SYSTEM)
        contents: list = []
        system_pending = system_text

        for message in messages:
            if message.role is Role.SYSTEM:
                continue
            text = message.content
            role = "model" if message.role is Role.ASSISTANT else "user"
            if role == "user" and system_pending:
                text = f"{system_pending}\n\n{text}"
                system_pending = ""
            contents.append(
                self._types.Content(role=role, parts=[self._types.Part.from_text(text=text)])
            )

        # A conversation that was *only* system messages still needs one user
        # turn to carry that instruction to the model.
        if system_pending:
            contents.append(
                self._types.Content(
                    role="user", parts=[self._types.Part.from_text(text=system_pending)]
                )
            )
        return contents
