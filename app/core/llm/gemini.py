"""The real client: Gemma tiers served by the free Google AI Studio API.

Everything provider-specific lives here and nowhere else. Agents depend on the
``LLMClient`` interface; only this file knows the request shape, the token-usage
field names, or that Gemma has no system role. Swapping to a self-hosted Ollama
endpoint (the spec's portability fallback) would be a second file implementing
the same interface, touching no agent.

Two open-model quirks it absorbs so callers never think about them:

* **No system role.** Gemma rejects a dedicated system turn, so any system
  messages are folded into the first user turn.
* **Free-tier caps.** Every call clears this tier's ``RateLimiter`` first
  (proactive) and is wrapped in ``retry_on_rate_limit`` (reactive), so the two
  free-tier limits are respected without the caller doing anything.

The SDK is imported lazily in the constructor, not at module load. That keeps
``app.core.llm`` importable -- and the FakeLLM-based tests runnable -- on a
machine that never installed ``google-genai``, matching the project's "green
outside the container" rule.
"""

from __future__ import annotations

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
    UsageCallback,
)
from app.core.llm.rate_limit import RateLimiter, estimate_input_tokens, retry_on_rate_limit


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
        self._sleep_fn = sleep_fn

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
            raw = retry_on_rate_limit(
                _call,
                retries=self._backoff_retries,
                is_rate_limited=self._is_rate_limited,
                sleep_fn=self._sleep_fn,
            )
        except self._errors.APIError as exc:
            # Only rate-limit errors reach here after exhausting backoff; other
            # API errors propagate unwrapped for the graph's failure path.
            if self._is_rate_limited(exc):
                raise RateLimitError(
                    f"rate limited after {self._backoff_retries} backoff retries"
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
