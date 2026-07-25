"""Tests for the schema-detection agent.

Two paths matter: the deterministic-only path (no client), and the LLM-enriched
path driven by FakeLLM. The second is exercised entirely offline -- the whole
reason Section 2 built a fake -- including the guarantee that a broken or
hostile LLM reply degrades to the profile rather than failing.
"""

from __future__ import annotations

import json

import pandas as pd

from app.agents.schema_detection import detect_schema
from app.core.llm import FakeLLM


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "customer_name": ["Alice Smith", "Bob Jones", "Carol White", "Dan Brown"],
            "age": [34, 28, 45, 52],
            "spend": [120.5, 80.0, 200.0, 55.5],
            "churned": ["yes", "no", "no", "yes"],
        }
    )


def _inference(**overrides) -> str:
    """A well-formed LLM reply, JSON-encoded, with optional field overrides."""
    payload = {
        "columns": [
            {
                "name": "customer_name",
                "meaning": "the customer's full name",
                "is_pii": True,
                "pii_type": "name",
            },
            {"name": "age", "meaning": "customer age in years", "is_pii": False},
            {"name": "spend", "meaning": "total spend", "is_pii": False},
            {"name": "churned", "meaning": "whether they left", "is_pii": False},
        ],
        "suggested_target": "churned",
        "task_type": "classification",
    }
    payload.update(overrides)
    return json.dumps(payload)


class TestDeterministicOnly:
    def test_no_client_returns_profile_unenriched(self):
        report = detect_schema(_frame(), client=None)
        assert report.llm_enriched is False
        assert all(c.meaning is None for c in report.columns)
        # Profiling still found the target and a sensible task type.
        assert report.suggested_target == "churned"
        assert report.task_type == "classification"


class TestLLMEnrichment:
    def test_meanings_are_attached(self):
        llm = FakeLLM([_inference()])
        report = detect_schema(_frame(), client=llm)

        assert report.llm_enriched is True
        assert report.column("age").meaning == "customer age in years"

    def test_semantic_pii_is_caught_and_excluded(self):
        """The regex misses a name column; the LLM catches it and it defaults out."""
        # Confirm profiling alone would NOT have flagged the name column.
        assert detect_schema(_frame(), client=None).column("customer_name").is_pii is False

        llm = FakeLLM([_inference()])
        report = detect_schema(_frame(), client=llm)
        name = report.column("customer_name")
        assert name.is_pii is True
        assert name.pii_type == "name"
        assert name.exclude is True

    def test_usage_callback_is_invoked(self):
        seen = []
        llm = FakeLLM([_inference()])
        detect_schema(_frame(), client=llm, on_usage=seen.append)
        assert len(seen) == 1


class TestGracefulDegradation:
    def test_invented_target_falls_back_to_the_heuristic(self):
        llm = FakeLLM([_inference(suggested_target="not_a_real_column")])
        report = detect_schema(_frame(), client=llm)
        # The bogus target is ignored; the profiling guess stands.
        assert report.suggested_target == "churned"

    def test_insight_for_unknown_column_is_ignored(self):
        payload = json.loads(_inference())
        payload["columns"].append(
            {"name": "ghost_column", "meaning": "does not exist", "is_pii": True}
        )
        llm = FakeLLM([json.dumps(payload)])
        report = detect_schema(_frame(), client=llm)
        # No phantom column appears in the report.
        assert "ghost_column" not in report.column_names()

    def test_malformed_llm_output_degrades_to_profile(self):
        # Never valid JSON: structured_complete exhausts retries and raises,
        # which the agent swallows, returning the deterministic report.
        llm = FakeLLM(default="the model is having a bad day")
        report = detect_schema(_frame(), client=llm)
        assert report.llm_enriched is False
        assert report.suggested_target == "churned"  # profiling still works
