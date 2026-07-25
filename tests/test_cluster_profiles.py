"""Tests for the LLM cluster-description agent.

The agent is handed measured differences and asked to phrase them -- it never
sees data. These tests cover that contract: the prompt carries the measurements
and nothing else, a description for a group that does not exist is discarded, and
losing the model costs the prose but not the clustering.
"""

from __future__ import annotations

import pytest

from app.agents.cluster_profiles import describe_clusters
from app.core.llm.base import LLMConfigError, RateLimitError
from app.core.llm.fake import FakeLLM
from app.ml.contracts import ClusterProfile

REPLY = (
    '{"clusters": ['
    '{"cluster": 0, "description": "Younger customers on lower incomes."},'
    '{"cluster": 1, "description": "Long-tenured, higher-earning customers."}'
    "]}"
)


@pytest.fixture
def profiles() -> list[ClusterProfile]:
    return [
        ClusterProfile(
            cluster=0,
            size=120,
            share=0.4,
            distinguishing_features={"age": "lower than average (24.3)"},
        ),
        ClusterProfile(
            cluster=1,
            size=180,
            share=0.6,
            distinguishing_features={"income": "higher than average (91,010)"},
        ),
    ]


class TestWithAWorkingModel:
    def test_descriptions_are_attached_to_the_right_groups(self, profiles):
        described = describe_clusters(profiles, client=FakeLLM([REPLY]))
        assert "Younger" in described[0].description
        assert "Long-tenured" in described[1].description

    def test_the_measured_differences_survive(self, profiles):
        """Prose is added alongside the numbers, never in place of them."""
        described = describe_clusters(profiles, client=FakeLLM([REPLY]))
        assert described[0].distinguishing_features == {"age": "lower than average (24.3)"}

    def test_a_description_for_a_nonexistent_group_is_ignored(self, profiles):
        client = FakeLLM(['{"clusters": [{"cluster": 99, "description": "Invented."}]}'])
        described = describe_clusters(profiles, client=client)
        assert all(p.description == "" for p in described)

    def test_usage_is_recorded(self, profiles):
        recorded = []
        describe_clusters(profiles, client=FakeLLM([REPLY]), on_usage=recorded.append)
        assert len(recorded) == 1


class TestThePromptIsGroundedInMeasurements:
    def test_it_carries_the_differences_code_computed(self, profiles):
        client = FakeLLM([REPLY])
        describe_clusters(profiles, client=client)
        prompt = client.last_prompt
        assert "lower than average" in prompt
        assert "higher than average" in prompt

    def test_it_carries_group_sizes(self, profiles):
        client = FakeLLM([REPLY])
        describe_clusters(profiles, client=client)
        assert "120" in client.last_prompt

    def test_a_featureless_group_is_described_as_such(self):
        """Rather than inviting the model to invent something for an empty list."""
        client = FakeLLM([REPLY])
        describe_clusters([ClusterProfile(cluster=0, size=10, share=1.0)], client=client)
        assert "no notable differences" in client.last_prompt


class TestDegradingWithoutAModel:
    def test_no_client_leaves_the_measurements_intact(self, profiles):
        described = describe_clusters(profiles, client=None)
        assert all(p.description == "" for p in described)
        assert described[0].distinguishing_features

    def test_a_missing_key_does_not_raise(self, profiles):
        assert describe_clusters(profiles, client=FakeLLM([LLMConfigError("no key")]))

    def test_a_rate_limit_does_not_raise(self, profiles):
        assert describe_clusters(profiles, client=FakeLLM([RateLimitError("429")]))

    def test_persistently_bad_output_does_not_raise(self, profiles):
        described = describe_clusters(
            profiles, client=FakeLLM(["nonsense"] * 10, default="nonsense")
        )
        assert all(p.description == "" for p in described)

    def test_no_profiles_is_a_no_op(self):
        assert describe_clusters([], client=FakeLLM([REPLY])) == []
