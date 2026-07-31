"""Tests for the structured-output layer -- the retry path in particular.

The whole point of FakeLLM is that this can be tested for real, offline: script
a malformed reply then a good one and watch structured_complete re-ask and
recover, exactly as it would against a flaky open model. No network, no key, no
flakiness.
"""

from __future__ import annotations

from typing import Literal

import pytest
from pydantic import BaseModel, field_validator

from app.core.llm import (
    FakeLLM,
    ModelTier,
    StructuredOutputError,
    extract_json,
    structured_complete,
    user,
)
from app.core.llm.base import RateLimitError, TransientLLMError
from app.core.llm.structured import structured_complete_tiered


class _Simple(BaseModel):
    """The smallest thing that can come back, for tests about *which model* ran."""

    value: str


class Decision(BaseModel):
    target: str
    task_type: Literal["classification", "regression"]


# Mimics Section 7's real guard: the model may only name columns that exist.
KNOWN_COLUMNS = {"age", "city", "income", "churn"}


class ColumnChoice(BaseModel):
    column: str
    strategy: Literal["standard_scale", "onehot_encode", "median_impute"]

    @field_validator("column")
    @classmethod
    def _must_exist(cls, v: str) -> str:
        if v not in KNOWN_COLUMNS:
            raise ValueError(f"unknown column {v!r}; must be one of {sorted(KNOWN_COLUMNS)}")
        return v


class TestExtractJson:
    def test_plain_json(self):
        assert extract_json('{"a": 1}') == '{"a": 1}'

    def test_fenced_json(self):
        assert extract_json('```json\n{"a": 1}\n```') == '{"a": 1}'

    def test_bare_fence(self):
        assert extract_json('```\n{"a": 1}\n```') == '{"a": 1}'

    def test_json_buried_in_prose(self):
        assert extract_json('Sure! Here it is: {"a": 1} hope that helps') == '{"a": 1}'

    def test_array_payload(self):
        assert extract_json("[1, 2, 3]") == "[1, 2, 3]"

    def test_no_json_raises(self):
        with pytest.raises(ValueError, match="no JSON"):
            extract_json("I could not help with that.")


class TestHappyPath:
    def test_valid_first_try(self):
        llm = FakeLLM(['{"target": "churn", "task_type": "classification"}'])

        result = structured_complete(llm, [user("what is the target?")], Decision)

        assert result.data.target == "churn"
        assert result.data.task_type == "classification"
        assert result.attempts == 1
        assert llm.call_count == 1

    def test_tolerates_a_fenced_reply(self):
        llm = FakeLLM(['```json\n{"target": "price", "task_type": "regression"}\n```'])

        result = structured_complete(llm, [user("go")], Decision)

        assert result.data.task_type == "regression"


class TestRetryPath:
    def test_malformed_then_valid_recovers(self):
        llm = FakeLLM(
            [
                "not json at all, sorry",  # attempt 1: unparseable
                '{"target": "churn", "task_type": "classification"}',  # attempt 2: good
            ]
        )

        result = structured_complete(llm, [user("target?")], Decision)

        assert result.data.target == "churn"
        assert result.attempts == 2
        assert llm.call_count == 2

    def test_schema_violation_triggers_a_reask(self):
        llm = FakeLLM(
            [
                '{"target": "churn", "task_type": "clustering"}',  # invalid enum value
                '{"target": "churn", "task_type": "classification"}',
            ]
        )

        result = structured_complete(llm, [user("target?")], Decision)

        assert result.data.task_type == "classification"
        assert result.attempts == 2

    def test_invented_column_is_rejected_then_corrected(self):
        """A hallucinated column name fails the validator and forces a re-ask."""
        llm = FakeLLM(
            [
                '{"column": "postcode", "strategy": "onehot_encode"}',  # not a real column
                '{"column": "city", "strategy": "onehot_encode"}',
            ]
        )

        result = structured_complete(llm, [user("how to encode?")], ColumnChoice)

        assert result.data.column == "city"
        assert result.attempts == 2

    def test_correction_message_shows_the_model_its_error(self):
        llm = FakeLLM(["garbage", '{"target": "churn", "task_type": "classification"}'])

        structured_complete(llm, [user("target?")], Decision)

        # The second call's prompt must contain the corrective feedback.
        second_prompt = llm.last_prompt
        assert "could not be parsed" in second_prompt
        assert "garbage" in second_prompt  # its own bad output is echoed back

    def test_exhausting_retries_raises_with_context(self):
        llm = FakeLLM(default="never valid json")

        with pytest.raises(StructuredOutputError) as excinfo:
            structured_complete(llm, [user("target?")], Decision, max_retries=2)

        err = excinfo.value
        assert err.attempts == 3  # 1 initial + 2 retries
        assert err.raw == "never valid json"
        assert llm.call_count == 3


class TestUsageAccounting:
    def test_callback_fires_once_per_attempt_including_retries(self):
        seen = []
        llm = FakeLLM(["bad", '{"target": "churn", "task_type": "classification"}'])

        structured_complete(llm, [user("target?")], Decision, on_usage=seen.append)

        # Two real requests were made, so cost tracking must see two responses.
        assert len(seen) == 2
        assert all(r.estimated for r in seen)

    def test_tier_is_passed_through_to_the_client(self):
        llm = FakeLLM(['{"target": "churn", "task_type": "classification"}'])

        structured_complete(llm, [user("go")], Decision, tier=ModelTier.LARGE)

        assert llm.calls[0].tier is ModelTier.LARGE


class TestSteppingDownATier:
    """``structured_complete_tiered`` (spec 6.1, and the free tier's per-model quotas).

    The critic and the report writer are the two large-tier prompts in the
    project, so the large model's quota is the first to run out. When it does,
    the choice is between a smaller model's answer and the deterministic
    fallback -- and the deterministic fallback is precisely the output a reader
    least wants: a report with no prose, a review that is only threshold checks.
    """

    def test_the_first_tier_is_used_when_it_works(self):
        client = FakeLLM(['{"value": "from the large model"}'])
        result = structured_complete_tiered(
            client, [user("hi")], _Simple, tiers=(ModelTier.LARGE, ModelTier.SMALL)
        )
        assert result.data.value == "from the large model"
        assert client.calls[0].tier is ModelTier.LARGE

    def test_a_rate_limit_falls_through_to_the_next_tier(self):
        client = FakeLLM([RateLimitError("429"), '{"value": "from the small model"}'])
        result = structured_complete_tiered(
            client, [user("hi")], _Simple, tiers=(ModelTier.LARGE, ModelTier.SMALL)
        )
        assert result.data.value == "from the small model"
        assert client.calls[-1].tier is ModelTier.SMALL

    def test_a_rate_limit_on_every_tier_raises(self):
        """So the caller can still degrade to its deterministic fallback."""
        client = FakeLLM([RateLimitError("429"), RateLimitError("429")])
        with pytest.raises(RateLimitError):
            structured_complete_tiered(
                client, [user("hi")], _Simple, tiers=(ModelTier.LARGE, ModelTier.SMALL)
            )

    def test_an_outage_steps_down_too(self):
        """It did not, until a sweep showed what that cost.

        The original rule was rate-limits-only, reasoning that a different model
        cannot fix an outage. True while both tiers were Gemini. But on
        2026-07-31 a five-dataset sweep took eight 503s and two 504s against two
        rate limits, so the step-down sat out ten of twelve failures and the
        critic and report writer produced deterministic templates on three runs
        in four. With a Gemma tier on the end of the chain there is now somewhere
        for a Gemini outage to go.
        """
        client = FakeLLM([TransientLLMError("503"), '{"value": "from the fallback"}'])
        result = structured_complete_tiered(
            client, [user("hi")], _Simple, tiers=(ModelTier.LARGE, ModelTier.FALLBACK)
        )
        assert result.data.value == "from the fallback"
        assert client.calls[-1].tier is ModelTier.FALLBACK

    def test_the_whole_chain_is_walked_before_giving_up(self):
        """Large is busy, small is out, and Gemma answers -- the real sequence."""
        client = FakeLLM([RateLimitError("429"), TransientLLMError("503"), '{"value": "gemma"}'])
        result = structured_complete_tiered(
            client,
            [user("hi")],
            _Simple,
            tiers=(ModelTier.LARGE, ModelTier.SMALL, ModelTier.FALLBACK),
        )
        assert result.data.value == "gemma"
        # FakeLLM raises before recording, so a failed attempt leaves no
        # RecordedCall. Exactly one recorded call, on the last tier, is therefore
        # the proof that the two before it were reached and both raised.
        assert [c.tier for c in client.calls] == [ModelTier.FALLBACK]

    def test_a_schema_failure_still_does_not_step_down(self):
        """The half of the original reasoning that survives.

        A model that misread the schema has already been re-asked by
        ``structured_complete``. Carrying the same confusion to another model
        doubles the latency of a failure to buy very little -- which is exactly
        the argument that used to cover outages too, and still holds here.
        """
        # More junk than the re-ask budget, so the tier genuinely gives up rather
        # than stumbling onto a valid reply on its last attempt.
        client = FakeLLM(["not json"] * 8)
        with pytest.raises(StructuredOutputError):
            structured_complete_tiered(
                client, [user("hi")], _Simple, tiers=(ModelTier.LARGE, ModelTier.FALLBACK)
            )
        # Junk *is* recorded -- it is a reply, not an exception -- so every
        # attempt shows up here, and all of them being LARGE is the assertion.
        assert {c.tier for c in client.calls} == {ModelTier.LARGE}

    def test_usage_is_recorded_for_every_tier_attempted(self):
        """A run that paid for a rate-limited attempt should show that it did."""
        recorded = []
        client = FakeLLM([RateLimitError("429"), '{"value": "ok"}'])
        structured_complete_tiered(
            client,
            [user("hi")],
            _Simple,
            tiers=(ModelTier.LARGE, ModelTier.SMALL),
            on_usage=recorded.append,
        )
        assert recorded, "no usage was recorded at all"

    def test_no_tiers_is_a_programming_error(self):
        with pytest.raises(ValueError, match="at least one tier"):
            structured_complete_tiered(FakeLLM([]), [user("hi")], _Simple, tiers=())
