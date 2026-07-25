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
