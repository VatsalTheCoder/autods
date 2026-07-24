"""Constructing the LLM client, and the FastAPI dependency that injects it.

The real client needs an API key; much of the system must run without one
(tests, a laptop with no key, a degraded upload path). So construction is
wrapped in a factory that returns ``None`` rather than raising when the client
cannot be built -- callers that can degrade (schema detection) check for None,
and tests override the dependency with a ``FakeLLM``.

The real client is cached: it holds the per-tier rate limiters, whose token
buckets must persist across requests to actually throttle. A fresh client per
request would reset the buckets every time and enforce nothing.
"""

from __future__ import annotations

import logging
from functools import lru_cache

from app.core.llm.base import LLMClient, LLMError

logger = logging.getLogger(__name__)


@lru_cache
def _cached_client() -> LLMClient | None:
    try:
        from app.core.llm.gemini import GeminiLLM

        return GeminiLLM()
    except LLMError as exc:
        # No key, or SDK missing. Not fatal: the caller decides what to do.
        logger.info("LLM client unavailable, running without it: %s", exc)
        return None


def get_optional_llm() -> LLMClient | None:
    """FastAPI dependency yielding the shared client, or None if unconfigured.

    Override in tests via ``app.dependency_overrides[get_optional_llm]`` to
    inject a ``FakeLLM`` and exercise the enrichment path offline.
    """
    return _cached_client()
