"""Test-suite-wide guarantees. Chiefly: **the tests never call a real LLM.**

Every LLM-touching test injects a ``FakeLLM`` explicitly, so the suite has always
been offline *in intent*. It was not offline *by construction*, and the
difference bites the moment a real key exists.

``get_optional_llm()`` reads ``GOOGLE_API_KEY`` from the environment. Once that
is set for the containers -- exactly what you do to try the pipeline against a
live model -- ``docker compose run api pytest`` inherits it, and the integration
tests stop being hermetic: they run the real pipeline, whose nodes call
``get_optional_llm()`` directly, and start issuing live requests. Measured on
this suite, that took it from 1m43s to **11m28s** and failed three tests, two of
them the ones asserting the deterministic fallback. All of it silent: no test
says "I just called Google". A suite that passes in CI and fails on the machine
of whoever configured a key is the worst way round for this to break.

So the key is moved out of the environment **at import time**, before pytest
collects anything. That timing is the load-bearing part: ``test_llm_gemini.py``
decides whether to skip its live round-trip with a module-level ``skipif`` that
reads the environment during collection. Stripping the key in a fixture instead
runs too late -- the test is collected as runnable, then fails at call time
because the key it was promised has gone.

The live round-trip is still available, deliberately: the key is preserved under
``AUTODS_LIVE_API_KEY``, which nothing but that test reads. Opting in stays
possible; opting in by accident does not.
"""

from __future__ import annotations

import os

import pytest

# The variable the live round-trip test opts in on. Named so that no production
# code path could read it by mistake -- the app knows only GOOGLE_API_KEY.
LIVE_KEY_VAR = "AUTODS_LIVE_API_KEY"

# Module scope, not a fixture: this must happen before test modules are imported.
_real_key = os.environ.pop("GOOGLE_API_KEY", None)
if _real_key:
    os.environ.setdefault(LIVE_KEY_VAR, _real_key)


@pytest.fixture(autouse=True, scope="session")
def _never_call_a_real_llm():
    """Drop any client or settings cached before the key was removed."""
    from app.core.config import get_settings
    from app.core.llm.factory import _cached_client

    _cached_client.cache_clear()
    get_settings.cache_clear()
    yield
    _cached_client.cache_clear()
    get_settings.cache_clear()


def test_the_guard_actually_holds() -> None:
    """The guard is load-bearing, so it gets its own assertion.

    Lives here rather than in a test module because it is about the suite's own
    environment: if this fails, every other test's offline claim is void.
    """
    from app.core.llm.factory import get_optional_llm

    assert not os.environ.get("GOOGLE_API_KEY")
    assert get_optional_llm() is None, "a live LLM client is reachable from the tests"
