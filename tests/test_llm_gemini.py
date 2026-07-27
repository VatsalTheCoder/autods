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
