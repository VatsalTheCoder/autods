"""The dataset chat agent: route a question, then answer it (spec 7.13).

Two kinds of question need two different tools, and the build plan is explicit
that "getting that routing right is the interesting part":

- *"Why was transaction amount important?"* is a question about meaning. It is
  answered from the run's written output, by retrieval.
- *"What's the average age?"* is arithmetic. No passage contains that number, so
  searching text for it produces either a near-miss or an invention. It goes to
  pandas.

**Routing is a decision made once, explicitly, and recorded.** It is not inferred
from whether retrieval happened to return something, and it is not left implicit
in a single prompt that "does its best". The route is stored on every message
(``models/chat_message.py``) precisely so a wrong answer can be diagnosed as the
wrong *tool* rather than guessed at.

The two answering paths have opposite failure modes, which is why they are kept
apart:

- Retrieval can be *fluent and wrong*: given passages that do not contain the
  answer, a model will happily assemble one from the nearest-sounding text. So
  the RAG path is instructed to refuse, and refusal is a first-class outcome
  rather than an error. It has to be the model's job -- ``services/retrieval.py``
  documents the measurement showing a distance threshold cannot do it.
- The pandas path can be *precisely wrong*: a valid expression that answers a
  different question. So the expression it ran is always reported alongside the
  number, which makes the mistake visible instead of authoritative.

Without an API key there is no chat at all. Every other agent in this project
degrades to a deterministic fallback, and this one cannot: routing a question and
phrasing an answer are the whole task, and a keyword-matching stand-in would
answer confidently and badly. Saying so is the honest degradation.
"""

from __future__ import annotations

import logging

import pandas as pd
from pydantic import BaseModel, Field

from app.core.llm.base import (
    LLMClient,
    LLMError,
    ModelTier,
    RateLimitError,
    UsageCallback,
    system,
    user,
)
from app.core.llm.structured import structured_complete, structured_complete_tiered
from app.ml.pandas_tool import UnsafeExpression, run_query
from app.models.chat_message import ChatRoute
from app.services.retrieval import Retrieved

logger = logging.getLogger(__name__)

AGENT_NAME = "dataset_chat"

# The answer given when there is no model configured. Stated plainly rather than
# dressed up: the feature is unavailable, not broken.
NO_MODEL_ANSWER = (
    "Chat needs a language model, and none is configured for this deployment. "
    "The report, the review and every artifact are still available on the "
    "Results page — they are produced without a model too."
)

NOT_INDEXED_ANSWER = (
    "This run has no searchable text, so there is nothing to answer from. That "
    "usually means indexing failed or the run predates the chat feature; "
    "re-running the job will index it."
)


class RoutingDecision(BaseModel):
    """Which tool should answer, and why."""

    # Constrained to the two tools plus a refusal. A free-text route would be a
    # routing decision that has to be parsed, which is a second place to be wrong.
    route: str = Field(
        description=(
            "'rag' for questions about meaning, findings, methods or explanations. "
            "'pandas' for arithmetic over the raw data — averages, counts, sums, "
            "distributions, filters. 'refused' if the question is not about this "
            "dataset or this analysis at all."
        )
    )
    reasoning: str = Field(default="", description="One sentence on why.")
    # Only meaningful for the pandas route. Asked for in the same call because a
    # second round trip to write the expression would double the latency of every
    # arithmetic question for no gain -- the model deciding "this is arithmetic"
    # is the model best placed to say what the arithmetic is.
    expression: str = Field(
        default="",
        description=(
            "For the 'pandas' route only: a single pandas expression over a "
            "DataFrame named df. No assignments, no imports, no lambdas."
        ),
    )


class GroundedAnswer(BaseModel):
    """An answer written from retrieved passages."""

    answer: str = Field(description="The answer, in plain English, or a refusal.")
    answered: bool = Field(
        default=True,
        description=(
            "False if the passages provided do not contain the answer. Say so "
            "rather than assembling something plausible from the nearest text."
        ),
    )
    cited_chunks: list[int] = Field(
        default_factory=list, description="Ids of the passages the answer relies on."
    )


class ChatAnswer(BaseModel):
    """What the chat produced, including how it got there."""

    answer: str
    route: ChatRoute
    grounding: str = ""


_ROUTER_SYSTEM = (
    "You route questions about a finished data-science run to one of two tools.\n\n"
    "Choose 'pandas' when the question asks for a number computed from the raw "
    "data: averages, medians, counts, sums, totals, minimums, maximums, "
    "distributions, or 'how many rows where X'. These cannot be answered from "
    "written text, because no text contains them.\n\n"
    "Choose 'rag' when the question asks what something means, why it happened, "
    "how the analysis was done, what was found, what is wrong with it, or what to "
    "do next. These are answered from the run's own written output.\n\n"
    "Choose 'refused' only when the question is not about this dataset or this "
    "analysis at all — general knowledge, current events, or a request to do "
    "something other than answer a question about the run.\n\n"
    "A question mentioning a column name is not automatically 'pandas'. "
    "'Why does age matter to the model?' is about meaning and routes to 'rag'; "
    "'what is the average age?' is arithmetic and routes to 'pandas'.\n\n"
    "For 'pandas', also write the expression. It must be a single expression over "
    "a DataFrame called df, using only column names given to you. Prefer the "
    "simplest form that answers the question."
)

_ANSWER_SYSTEM = (
    "You answer questions about a finished data-science run, using only the "
    "passages you are given.\n\n"
    "Rules:\n"
    "- Use only what the passages say. Do not add facts from general knowledge, "
    "and do not compute new numbers — you may quote figures that appear in the "
    "passages, but you may not derive them.\n"
    "- If the passages do not contain the answer, set answered to false and say "
    "plainly what is missing. This is the correct response, not a failure. An "
    "assembled answer that sounds right is worse than admitting the gap.\n"
    "- Cite the passage ids you used.\n"
    "- Plain English, two to five sentences. The reader is not a specialist."
)

_PANDAS_SYSTEM = (
    "You phrase the result of a calculation as a sentence.\n\n"
    "You are given a question, the pandas expression that was run, and its "
    "result. State the answer in plain English, quoting the number exactly as "
    "given — do not round it further, recompute it, or add figures that are not "
    "in the result.\n\n"
    "One or two sentences. If the result looks like it does not answer the "
    "question, say so rather than presenting it as though it does."
)


def answer_question(
    question: str,
    *,
    passages: list[Retrieved] | None = None,
    frame: pd.DataFrame | None = None,
    columns: list[str] | None = None,
    client: LLMClient | None = None,
    on_usage: UsageCallback | None = None,
) -> ChatAnswer:
    """Route the question and answer it. Never raises.

    A chat that raises on a bad question is a chat that loses the conversation,
    so every failure here becomes an answer that says what went wrong.
    """
    if client is None:
        return ChatAnswer(answer=NO_MODEL_ANSWER, route=ChatRoute.REFUSED, grounding="no-model")

    try:
        decision = _route(question, columns or [], client, on_usage)
    except LLMError as exc:
        logger.warning("Chat routing failed: %s", exc)
        return ChatAnswer(
            answer="The question could not be routed just now. Please try again.",
            route=ChatRoute.REFUSED,
            grounding=f"routing-error: {exc}",
        )

    if decision.route == "pandas" and frame is not None:
        return _answer_with_pandas(question, decision, frame, client, on_usage)

    if decision.route == "refused":
        return ChatAnswer(
            answer=(
                "That is outside what this analysis covers. Ask about the "
                "dataset, the model, the findings, or the review."
            ),
            route=ChatRoute.REFUSED,
            grounding=decision.reasoning,
        )

    return _answer_with_passages(question, passages or [], client, on_usage)


def _route(
    question: str,
    columns: list[str],
    client: LLMClient,
    on_usage: UsageCallback | None,
) -> RoutingDecision:
    """Decide which tool answers. The interesting part of this section."""
    available = (
        f"\n\nThe dataset's columns are: {', '.join(columns)}."
        if columns
        else "\n\nThe dataset's columns are not available."
    )
    result = structured_complete(
        client,
        [system(_ROUTER_SYSTEM), user(f"Question: {question}{available}")],
        RoutingDecision,
        # The small tier: this is a classification with a fixed vocabulary, not
        # a reasoning task, and the large tier would cost latency on every single
        # question to decide between three options.
        tier=ModelTier.SMALL,
        on_usage=on_usage,
    )
    decision = result.data
    if decision.route not in {"rag", "pandas", "refused"}:
        # An unrecognised route falls back to retrieval rather than to an error:
        # the RAG path can refuse for itself, so the safe default is the one that
        # is allowed to say "I cannot answer that".
        logger.info("Chat router returned %r; falling back to rag", decision.route)
        decision.route = "rag"
    logger.info("Chat routed to %s: %s", decision.route, decision.reasoning)
    return decision


def _answer_with_passages(
    question: str,
    passages: list[Retrieved],
    client: LLMClient,
    on_usage: UsageCallback | None,
) -> ChatAnswer:
    """Answer from retrieved text, or refuse."""
    if not passages:
        return ChatAnswer(
            answer=(
                "Nothing in this run's output addresses that. It may be a "
                "question the analysis did not examine."
            ),
            route=ChatRoute.RAG,
            grounding="no-passages",
        )

    context = "\n\n".join(passage.as_context() for passage in passages)
    messages = [system(_ANSWER_SYSTEM), user(f"Passages:\n\n{context}\n\nQuestion: {question}")]

    # The large tier first (spec 6.1 puts dataset chat at mid-large), then the
    # small one if it is rate limited. Every other agent in this project degrades
    # to a deterministic fallback when the model is unavailable; this one has
    # none, because routing and phrasing *are* the task -- so the fallback is a
    # smaller model rather than no answer. On the free tier the spec mandates,
    # the large tier's quota is the first thing to run out, and a chat that stops
    # working for the rest of the day is worse than one that answers slightly
    # less fluently from the same passages.
    try:
        result = structured_complete_tiered(
            client,
            messages,
            GroundedAnswer,
            tiers=(ModelTier.LARGE, ModelTier.SMALL),
            on_usage=on_usage,
        )
    except RateLimitError as exc:
        logger.warning("Chat answering rate limited on every tier: %s", exc)
        return ChatAnswer(
            answer=(
                "The model is rate limited at the moment, so this answer "
                "could not be written. Please try again shortly."
            ),
            route=ChatRoute.RAG,
            grounding=f"rate-limited: {exc}",
        )
    except LLMError as exc:
        logger.warning("Chat answering failed: %s", exc)
        return ChatAnswer(
            answer="The answer could not be written just now. Please try again.",
            route=ChatRoute.RAG,
            grounding=f"answer-error: {exc}",
        )

    grounded = result.data
    cited = grounded.cited_chunks or [passage.chunk_id for passage in passages]
    return ChatAnswer(
        answer=grounded.answer,
        # A refusal is recorded as a refusal even though retrieval ran, so the
        # transcript distinguishes "answered from the run" from "could not be".
        route=ChatRoute.RAG if grounded.answered else ChatRoute.REFUSED,
        grounding="chunks: " + ", ".join(str(chunk_id) for chunk_id in cited),
    )


def _answer_with_pandas(
    question: str,
    decision: RoutingDecision,
    frame: pd.DataFrame,
    client: LLMClient,
    on_usage: UsageCallback | None,
) -> ChatAnswer:
    """Run the expression, then have the model phrase the result.

    The expression is always reported. A calculation that answered a subtly
    different question is the failure mode here, and it is only catchable if the
    reader can see what was actually computed.
    """
    if not decision.expression.strip():
        return ChatAnswer(
            answer="That looked like a calculation, but no query could be written for it.",
            route=ChatRoute.PANDAS,
            grounding="no-expression",
        )

    try:
        result = run_query(frame, decision.expression)
    except UnsafeExpression as exc:
        logger.info("Chat rejected an expression: %s", exc)
        return ChatAnswer(
            answer=(
                f"That question needs a calculation this tool will not run: {exc} "
                "Try asking for a simple statistic, such as an average or a count."
            ),
            route=ChatRoute.PANDAS,
            grounding=f"rejected: {decision.expression}",
        )

    try:
        phrased = structured_complete(
            client,
            [
                system(_PANDAS_SYSTEM),
                user(
                    f"Question: {question}\n"
                    f"Expression run: {result.expression}\n"
                    f"Result:\n{result.value}"
                ),
            ],
            GroundedAnswer,
            tier=ModelTier.SMALL,
            on_usage=on_usage,
        )
        answer = phrased.data.answer
    except LLMError:
        # The number is the answer; the sentence around it is a convenience. A
        # provider failure should not lose a calculation that already succeeded.
        answer = f"{result.value}"

    return ChatAnswer(
        answer=answer,
        route=ChatRoute.PANDAS,
        grounding=f"query: {result.expression}",
    )
