"""Routing, grounding and refusal in the chat agent (spec 7.13).

The build plan says routing is the interesting part of this section, so most of
these tests are about the *decision* rather than the prose: that arithmetic does
not go to retrieval, that a question about meaning does not go to pandas, and
that each outcome is recorded as what it was.

The model is faked throughout. What is being tested is the wiring around it --
which tool runs, what it is given, what happens when it fails -- and a real model
would make these tests slow, expensive and non-deterministic without testing any
of that better.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from app.agents.chat import NO_MODEL_ANSWER, answer_question
from app.core.llm.base import ModelTier, RateLimitError, TransientLLMError
from app.core.llm.fake import FakeLLM
from app.models.chat_message import ChatRoute
from app.services.retrieval import Retrieved

# One scripted failure per tier. Derived from the enum rather than written as a
# literal so that adding a tier does not quietly turn these degradation tests
# into success tests: with too few failures scripted, FakeLLM falls through to
# its default reply and the agent succeeds, which is what happened when
# ModelTier.FALLBACK was added.
TIERS = len(ModelTier)


@pytest.fixture
def frame() -> pd.DataFrame:
    return pd.DataFrame({"age": [20, 30, 40, 50], "churn": [0, 1, 0, 1]})


@pytest.fixture
def passages() -> list[Retrieved]:
    return [
        Retrieved(
            chunk_id=7,
            source="explainability_report.json",
            heading="Why support_calls matters to the model",
            content=(
                "support_calls is the number 1 most important column, carrying "
                "37.5% of the model's total explained influence."
            ),
            distance=0.18,
        )
    ]


def route_json(route: str, expression: str = "") -> str:
    return json.dumps({"route": route, "reasoning": "because", "expression": expression})


def answer_json(answer: str = "Because it is the strongest driver.", answered: bool = True) -> str:
    return json.dumps({"answer": answer, "answered": answered, "cited_chunks": [7]})


class TestRouting:
    def test_arithmetic_goes_to_pandas_and_computes(self, frame, passages):
        client = FakeLLM(
            [route_json("pandas", "df['age'].mean()"), answer_json("The mean age is 35.")]
        )
        result = answer_question(
            "What is the average age?", passages=passages, frame=frame, client=client
        )
        assert result.route == ChatRoute.PANDAS
        assert result.grounding == "query: df['age'].mean()"

    def test_meaning_goes_to_retrieval(self, frame, passages):
        client = FakeLLM([route_json("rag"), answer_json()])
        result = answer_question(
            "Why was support_calls important?", passages=passages, frame=frame, client=client
        )
        assert result.route == ChatRoute.RAG
        assert "7" in result.grounding

    def test_out_of_scope_is_refused_before_any_tool_runs(self, frame, passages):
        """Refusing at the router means no retrieval and no calculation happen."""
        client = FakeLLM([route_json("refused")])
        result = answer_question(
            "Who won the world cup?", passages=passages, frame=frame, client=client
        )
        assert result.route == ChatRoute.REFUSED
        # One call: the router. The answering model was never reached.
        assert len(client.calls) == 1

    def test_an_unrecognised_route_falls_back_to_retrieval(self, frame, passages):
        """RAG can refuse for itself, so it is the safe default."""
        client = FakeLLM([route_json("something_else"), answer_json()])
        result = answer_question("A question", passages=passages, frame=frame, client=client)
        assert result.route == ChatRoute.RAG

    def test_the_router_is_told_the_column_names(self, frame, passages):
        """Without them it writes expressions against columns that do not exist."""
        client = FakeLLM([route_json("pandas", "df['age'].mean()"), answer_json()])
        answer_question(
            "average age?", passages=passages, frame=frame, columns=["age", "churn"], client=client
        )
        sent = "\n".join(m.content for m in client.calls[0].messages)
        assert "age" in sent and "churn" in sent


class TestGroundingAndRefusal:
    def test_a_model_that_cannot_answer_is_recorded_as_a_refusal(self, frame, passages):
        """Distinguishes 'answered from the run' from 'could not be'."""
        client = FakeLLM(
            [route_json("rag"), answer_json("The passages do not say.", answered=False)]
        )
        result = answer_question("Something unasked", passages=passages, frame=frame, client=client)
        assert result.route == ChatRoute.REFUSED

    def test_no_passages_refuses_rather_than_inventing(self, frame):
        client = FakeLLM([route_json("rag")])
        result = answer_question("A question", passages=[], frame=frame, client=client)
        assert "Nothing in this run's output" in result.answer
        # The answering model was never called: there was nothing to answer from.
        assert len(client.calls) == 1

    def test_the_passages_are_actually_put_in_the_prompt(self, frame, passages):
        client = FakeLLM([route_json("rag"), answer_json()])
        answer_question("Why?", passages=passages, frame=frame, client=client)
        sent = "\n".join(m.content for m in client.calls[1].messages)
        assert "37.5%" in sent
        assert "support_calls" in sent


class TestTheSandboxIsEnforcedHereToo:
    def test_a_dangerous_expression_is_refused_not_run(self, frame, passages):
        """The agent is the caller the sandbox exists to protect against."""
        client = FakeLLM([route_json("pandas", "__import__('os').system('id')")])
        result = answer_question("do something bad", passages=passages, frame=frame, client=client)
        assert result.route == ChatRoute.PANDAS
        assert result.grounding.startswith("rejected:")
        assert "will not run" in result.answer

    def test_an_empty_expression_does_not_crash(self, frame, passages):
        client = FakeLLM([route_json("pandas", "")])
        result = answer_question(
            "average of nothing", passages=passages, frame=frame, client=client
        )
        assert result.route == ChatRoute.PANDAS
        assert "no query could be written" in result.answer


class TestWhenTheModelIsUnavailable:
    def test_no_client_says_so_plainly(self, frame, passages):
        result = answer_question("Anything", passages=passages, frame=frame, client=None)
        assert result.answer == NO_MODEL_ANSWER
        assert result.route == ChatRoute.REFUSED

    def test_a_routing_failure_does_not_raise(self, frame, passages):
        """A chat that raises on a bad question loses the conversation."""
        result = answer_question(
            "Anything",
            passages=passages,
            frame=frame,
            client=FakeLLM([TransientLLMError("503")] * TIERS),
        )
        assert result.route == ChatRoute.REFUSED
        assert "could not be routed" in result.answer

    def test_the_answer_falls_back_a_tier_when_rate_limited(self, frame, passages):
        """The large tier's quota runs out first on the free tier."""
        client = FakeLLM(
            [route_json("rag"), RateLimitError("429"), answer_json("A smaller answer.")]
        )
        result = answer_question("Why?", passages=passages, frame=frame, client=client)
        assert result.route == ChatRoute.RAG
        assert result.answer == "A smaller answer."

    def test_rate_limited_on_every_tier_says_so(self, frame, passages):
        client = FakeLLM([route_json("rag"), *([RateLimitError("429")] * TIERS)])
        result = answer_question("Why?", passages=passages, frame=frame, client=client)
        assert "rate limited" in result.answer
        assert result.grounding.startswith("rate-limited:")

    def test_a_calculation_survives_the_phrasing_model_failing(self, frame, passages):
        """The number is the answer; the sentence around it is a convenience."""
        client = FakeLLM(
            [route_json("pandas", "df['age'].mean()"), *([TransientLLMError("503")] * TIERS)]
        )
        result = answer_question("average age", passages=passages, frame=frame, client=client)
        assert result.route == ChatRoute.PANDAS
        assert "35" in result.answer
