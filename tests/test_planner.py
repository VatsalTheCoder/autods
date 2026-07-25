"""Tests for the planner agent.

Two things matter here. First, that a plan is *always* produced -- a missing key,
a rate limit or a malformed reply must degrade to defaults rather than fail a job,
because the plan is an optimisation and the pipeline runs correctly without one.
Second, that the artifact never claims the model decided something it did not:
``source`` is stamped by the code that knows the truth, not by the model.

All offline, against ``FakeLLM``.
"""

from __future__ import annotations

import pandas as pd
import pytest

from app.agents.planner import make_plan
from app.core.llm.base import LLMConfigError, RateLimitError
from app.core.llm.fake import FakeLLM
from app.services.profiling import profile_dataset

VALID_PLAN = (
    '{"drop_duplicate_rows": true, "drop_high_null_columns": false, '
    '"rationale": "Rows look like distinct records."}'
)


@pytest.fixture
def schema():
    frame = pd.DataFrame(
        {
            "age": [34, 28, 45, 51],
            "city": ["London", "Leeds", "Bristol", "London"],
            "churn": ["yes", "no", "no", "yes"],
        }
    )
    return profile_dataset(frame)


class TestWithAWorkingModel:
    def test_the_models_flags_are_obeyed(self, schema):
        plan = make_plan(schema, client=FakeLLM([VALID_PLAN]))
        assert plan.drop_duplicate_rows is True
        assert plan.drop_high_null_columns is False

    def test_the_rationale_is_kept(self, schema):
        plan = make_plan(schema, client=FakeLLM([VALID_PLAN]))
        assert "distinct records" in plan.rationale

    def test_the_plan_is_marked_as_coming_from_the_llm(self, schema):
        assert make_plan(schema, client=FakeLLM([VALID_PLAN])).source == "llm"

    def test_a_malformed_reply_is_retried_then_accepted(self, schema):
        """The structured-output re-ask path, exercised for real without a network."""
        client = FakeLLM(["not json at all", VALID_PLAN])
        plan = make_plan(schema, client=client)
        assert plan.source == "llm"
        assert client.call_count == 2

    def test_the_model_cannot_claim_a_source_it_did_not_earn(self, schema):
        """A hallucinated ``source`` must not turn a default into an LLM decision."""
        client = FakeLLM(['{"drop_duplicate_rows": true, "source": "default"}'])
        assert make_plan(schema, client=client).source == "llm"

    def test_the_prompt_describes_the_columns_without_sending_values(self, schema):
        """Dataset shape is enough to plan with; values would just cost tokens."""
        client = FakeLLM([VALID_PLAN])
        make_plan(schema, client=client)
        prompt = client.last_prompt
        assert "age" in prompt and "city" in prompt
        assert "London" not in prompt

    def test_usage_is_recorded_through_the_callback(self, schema):
        recorded = []
        make_plan(schema, client=FakeLLM([VALID_PLAN]), on_usage=recorded.append)
        assert len(recorded) == 1
        assert recorded[0].total_tokens > 0


class TestDegradingToDefaults:
    def test_no_client_gives_a_working_default_plan(self, schema):
        plan = make_plan(schema, client=None)
        assert plan.source == "default"
        assert plan.drop_duplicate_rows is True
        assert plan.drop_high_null_columns is True

    def test_a_missing_api_key_does_not_fail_the_job(self, schema):
        plan = make_plan(schema, client=FakeLLM([LLMConfigError("no key")]))
        assert plan.source == "default"

    def test_a_rate_limit_does_not_fail_the_job(self, schema):
        plan = make_plan(schema, client=FakeLLM([RateLimitError("429")]))
        assert plan.source == "default"

    def test_persistently_malformed_output_falls_back(self, schema):
        """Structured output gives up eventually; the pipeline still runs."""
        client = FakeLLM(["nonsense"] * 10, default="still nonsense")
        plan = make_plan(schema, client=client)
        assert plan.source == "default"

    def test_the_fallback_says_why(self, schema):
        plan = make_plan(schema, client=FakeLLM([RateLimitError("429")]))
        assert plan.rationale
