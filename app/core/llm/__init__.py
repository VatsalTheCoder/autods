"""The shared LLM layer every agent depends on.

Import from here, never from the submodules, so an agent is coupled to the
interface and not to a provider:

    from app.core.llm import FakeLLM, ModelTier, structured_complete, system, user

``GeminiLLM`` is intentionally re-exported lazily via ``__getattr__``: importing
it pulls in the ``google-genai`` SDK, and we do not want ``import app.core.llm``
to require that SDK just so a FakeLLM-based test can run. Access the name and it
loads; ignore it and it never does.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from app.core.llm.base import (
    ChatMessage,
    GenerationParams,
    LLMClient,
    LLMConfigError,
    LLMError,
    LLMResponse,
    ModelTier,
    RateLimitError,
    Role,
    StructuredOutputError,
    TransientLLMError,
    UsageCallback,
    assistant,
    system,
    user,
)
from app.core.llm.fake import FakeLLM, RecordedCall
from app.core.llm.rate_limit import (
    RateLimiter,
    TokenBucket,
    estimate_input_tokens,
    retry_with_backoff,
)
from app.core.llm.structured import StructuredResult, extract_json, structured_complete
from app.core.llm.usage import estimate_cost, job_token_totals, make_usage_recorder

if TYPE_CHECKING:
    from app.core.llm.gemini import GeminiLLM

__all__ = [
    "ChatMessage",
    "FakeLLM",
    "GeminiLLM",
    "GenerationParams",
    "LLMClient",
    "LLMConfigError",
    "LLMError",
    "LLMResponse",
    "ModelTier",
    "RateLimitError",
    "RateLimiter",
    "RecordedCall",
    "Role",
    "StructuredOutputError",
    "StructuredResult",
    "TokenBucket",
    "TransientLLMError",
    "UsageCallback",
    "assistant",
    "estimate_cost",
    "estimate_input_tokens",
    "extract_json",
    "job_token_totals",
    "make_usage_recorder",
    "retry_with_backoff",
    "structured_complete",
    "system",
    "user",
]


def __getattr__(name: str):
    """Load the provider client only when it is actually asked for."""
    if name == "GeminiLLM":
        from app.core.llm.gemini import GeminiLLM

        return GeminiLLM
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
