"""Tests for the real Google AI Studio client.

The message-translation and config-guard tests need no network. The live
round-trip needs a real key and is skipped without one, so the suite stays green
offline -- the same skip discipline as the storage and DB tests. Set
GOOGLE_API_KEY in the environment to exercise it against the live free tier.
"""

from __future__ import annotations

import os
from typing import Literal

import pytest
from pydantic import BaseModel

from app.core.config import Settings
from app.core.llm import LLMConfigError, ModelTier, structured_complete, system, user


def test_missing_key_raises_config_error():
    """No key must fail fast at construction -- never a confusing 401 mid-run."""
    from app.core.llm import GeminiLLM

    settings = Settings(_env_file=None, google_api_key=None)
    with pytest.raises(LLMConfigError, match="GOOGLE_API_KEY"):
        GeminiLLM(api_key=None, settings=settings)


# --- Translation tests: need the SDK installed, but no network ----------------

pytest.importorskip("google.genai")


def _client_with_dummy_key():
    from app.core.llm import GeminiLLM

    # A dummy key constructs the client without any network call; we only reach
    # the wire when generate_content runs, which these tests never do.
    return GeminiLLM(api_key="dummy-key-not-used", settings=Settings(_env_file=None))


class TestMessageTranslation:
    def test_system_is_folded_into_the_first_user_turn(self):
        """Gemma has no system role, so system text must ride the first user turn."""
        client = _client_with_dummy_key()
        contents = client._to_contents([system("be terse"), user("hello")])

        assert len(contents) == 1
        assert contents[0].role == "user"
        text = contents[0].parts[0].text
        assert "be terse" in text and "hello" in text

    def test_assistant_maps_to_model_role(self):
        from app.core.llm.base import assistant

        client = _client_with_dummy_key()
        contents = client._to_contents([user("hi"), assistant("hello back")])

        assert [c.role for c in contents] == ["user", "model"]

    def test_system_only_conversation_still_yields_a_user_turn(self):
        client = _client_with_dummy_key()
        contents = client._to_contents([system("just do it")])

        assert len(contents) == 1
        assert contents[0].role == "user"
        assert "just do it" in contents[0].parts[0].text

    def test_tier_selects_the_configured_model(self):
        settings = Settings(
            _env_file=None,
            google_api_key="x",
            llm_model_small="small-model",
            llm_model_large="large-model",
        )
        from app.core.llm import GeminiLLM

        client = GeminiLLM(api_key="x", settings=settings)
        assert client._models[ModelTier.SMALL] == "small-model"
        assert client._models[ModelTier.LARGE] == "large-model"


# --- Live round-trip: needs a real key ---------------------------------------

# Opts in on AUTODS_LIVE_API_KEY rather than GOOGLE_API_KEY. conftest.py moves
# the real key there at import time so the rest of the suite cannot reach the
# network by accident; this is the one test that is *supposed* to, so it reads
# the key from where the guard put it. See tests/conftest.py.
requires_live_key = pytest.mark.skipif(
    not os.getenv("AUTODS_LIVE_API_KEY"),
    reason="set GOOGLE_API_KEY to run the live round-trip",
)


class Sentiment(BaseModel):
    label: Literal["positive", "negative", "neutral"]


@requires_live_key
class TestLiveRoundTrip:
    def test_structured_schema_round_trips_through_the_real_model(self):
        from app.core.llm import GeminiLLM

        # Passed explicitly: the guard has removed GOOGLE_API_KEY from the
        # environment, so the client cannot pick it up implicitly here.
        client = GeminiLLM(api_key=os.environ["AUTODS_LIVE_API_KEY"])
        result = structured_complete(
            client,
            [user("Classify the sentiment of: 'I love this product, it works great!'")],
            Sentiment,
            tier=ModelTier.SMALL,
        )
        assert result.data.label == "positive"
        # Real usage metadata, not estimated.
        assert result.responses[0].estimated is False
        assert result.responses[0].input_tokens > 0


# --- Failure classification: no network, no key, no real provider -------------


class TestTransientFailures:
    """A provider blip must be retried, then reported as a provider blip.

    Both halves matter. Without the retry, one 503 costs an agent its LLM answer
    for the whole run. Without the classification, the failure reaches the agent
    as a bare exception and is recorded as "fell back after an unexpected error"
    -- which reads like a defect in this codebase and sends the next person
    looking for one.
    """

    @staticmethod
    def _client(monkeypatch, responses, *, budget=90.0, timeout=60):
        """A client whose provider call yields ``responses`` in order.

        Time is simulated: each attempt advances the clock by the request
        timeout, so a hanging provider costs what it would really cost without a
        test taking minutes to prove it.
        """
        from app.core.llm import GeminiLLM

        clock = {"t": 0.0}
        settings = Settings(
            _env_file=None,
            google_api_key="x",
            llm_backoff_retries=5,
            llm_request_timeout=timeout,
            llm_retry_budget_seconds=budget,
        )
        client = GeminiLLM(
            api_key="x",
            settings=settings,
            sleep_fn=lambda s: clock.__setitem__("t", clock["t"] + s),
            time_fn=lambda: clock["t"],
        )

        attempts = {"n": 0}
        queue = list(responses)

        def fake_generate(**_kwargs):
            attempts["n"] += 1
            outcome = queue.pop(0) if queue else responses[-1]
            if isinstance(outcome, Exception):
                # A failed request still burns its timeout before raising.
                clock["t"] += timeout
                raise outcome
            return outcome

        monkeypatch.setattr(client._client.models, "generate_content", fake_generate)
        return client, attempts, clock

    @staticmethod
    def _ok():
        class _Raw:
            text = "hello"
            usage_metadata = None

        return _Raw()

    @staticmethod
    def _api_error(code: int):
        from google.genai import errors

        return errors.APIError(code, {"message": "upstream", "status": "UNAVAILABLE"})

    def test_a_503_is_retried_and_can_recover(self, monkeypatch):
        client, attempts, _ = self._client(monkeypatch, [self._api_error(503), self._ok()])

        response = client.complete([user("hi")])

        assert response.text == "hello"
        assert attempts["n"] == 2

    @pytest.mark.parametrize("code", [500, 502, 503, 504])
    def test_every_server_side_status_is_treated_as_transient(self, monkeypatch, code):
        client, attempts, _ = self._client(monkeypatch, [self._api_error(code), self._ok()])
        assert client.complete([user("hi")]).text == "hello"
        assert attempts["n"] == 2

    @pytest.mark.parametrize("code", [400, 403, 404])
    def test_client_side_errors_are_not_retried(self, monkeypatch, code):
        """Retrying a malformed request re-sends the same malformed request."""
        from google.genai import errors

        client, attempts, _ = self._client(monkeypatch, [self._api_error(code)])

        with pytest.raises(errors.APIError):
            client.complete([user("hi")])
        assert attempts["n"] == 1

    def test_a_transport_timeout_is_transient_too(self, monkeypatch):
        """The failure that actually bites: httpx raises, not the SDK.

        A request that outruns ``llm_request_timeout`` never becomes an
        ``APIError`` with a status code, so classifying on code alone would miss
        the single most likely transient failure in this project.
        """
        import httpx

        client, attempts, _ = self._client(monkeypatch, [httpx.ReadTimeout("too slow"), self._ok()])

        assert client.complete([user("hi")]).text == "hello"
        assert attempts["n"] == 2

    def test_an_exhausted_transient_failure_becomes_an_llm_error(self, monkeypatch):
        """The type is what routes an agent to its "say why" branch."""
        from app.core.llm import LLMError, TransientLLMError

        client, _, _ = self._client(monkeypatch, [self._api_error(503)])

        with pytest.raises(TransientLLMError) as caught:
            client.complete([user("hi")])
        # Agents catch LLMError; TransientLLMError has to be one or the whole
        # point of the classification is lost.
        assert isinstance(caught.value, LLMError)
        assert "unavailable" in str(caught.value)

    def test_the_message_names_the_cause_not_just_the_failure(self, monkeypatch):
        """ "ReadTimeout" in an artifact is worth more than "an error occurred"."""
        import httpx

        client, _, _ = self._client(monkeypatch, [httpx.ReadTimeout("too slow")])

        with pytest.raises(Exception, match="ReadTimeout"):
            client.complete([user("hi")])

    def test_a_hanging_provider_is_bounded_by_the_budget(self, monkeypatch):
        """Five retries against a 60s timeout would stall a node for minutes."""
        from app.core.llm import TransientLLMError

        client, attempts, clock = self._client(monkeypatch, [self._api_error(503)], budget=90.0)

        with pytest.raises(TransientLLMError):
            client.complete([user("hi")])

        assert attempts["n"] == 2  # one retry fits in 90s; a second does not
        assert clock["t"] < 130.0

    def test_rate_limiting_still_has_its_own_type(self, monkeypatch):
        """A regression guard: 429 must not be swallowed by the new branch."""
        from app.core.llm import RateLimitError

        client, _, _ = self._client(monkeypatch, [self._api_error(429)])

        with pytest.raises(RateLimitError):
            client.complete([user("hi")])
