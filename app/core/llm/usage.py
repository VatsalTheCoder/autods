"""Turn LLM responses into ``token_usage`` rows -- the cost-tracking callback.

Section 2's job is the *wiring*: give any agent a one-line way to have every
request it makes logged per agent per job (spec section 13). An agent does not
write SQL or think about cost; it asks for a recorder and passes it as the
``on_usage`` callback:

    recorder = make_usage_recorder(db, job_id, agent="planner")
    result = structured_complete(client, msgs, PlannerPlan, on_usage=recorder)

Every underlying call -- retries included -- then lands its own row.

The LLM client stays database-agnostic: it emits an ``LLMResponse`` and calls
the callback. All knowledge of Postgres lives here, on the far side of that
boundary, which is what let the client be tested with no database at all.
"""

from __future__ import annotations

from decimal import Decimal

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.core.llm.base import LLMResponse, ModelTier, UsageCallback
from app.models.token_usage import TokenUsage

# Price per one million tokens, in USD, keyed by model tier. The free Google AI
# Studio tier is genuinely $0 (spec section 6.3), so these default to zero --
# but the column and arithmetic exist so that switching to a paid endpoint or a
# self-hosted GPU with an amortised cost is a change to *this table*, nothing
# else. Fill in real rates here if the serving story changes.
PRICE_PER_MILLION: dict[ModelTier, tuple[Decimal, Decimal]] = {
    # tier: (input_usd_per_1M, output_usd_per_1M)
    ModelTier.SMALL: (Decimal("0"), Decimal("0")),
    ModelTier.LARGE: (Decimal("0"), Decimal("0")),
}

_MILLION = Decimal("1000000")


def estimate_cost(tier: ModelTier, input_tokens: int, output_tokens: int) -> Decimal:
    """USD cost of one request, from the tier's per-million rates."""
    input_rate, output_rate = PRICE_PER_MILLION.get(tier, (Decimal("0"), Decimal("0")))
    cost = (Decimal(input_tokens) * input_rate + Decimal(output_tokens) * output_rate) / _MILLION
    # Quantise to the column's 6 decimal places so what we store equals what we
    # computed, with no surprise rounding on the way into Postgres.
    return cost.quantize(Decimal("0.000001"))


def make_usage_recorder(db: Session, job_id: int, agent: str) -> UsageCallback:
    """Return an ``on_usage`` callback that logs each response for this agent.

    The row is added to the session but not committed -- the caller owns the
    transaction boundary (a Section 4 worker commits per node), so usage is
    persisted or rolled back atomically with the work that produced it.
    """

    def _record(response: LLMResponse) -> None:
        db.add(
            TokenUsage(
                job_id=job_id,
                agent=agent,
                model=response.model,
                tier=response.tier.value,
                input_tokens=response.input_tokens,
                output_tokens=response.output_tokens,
                cost_usd=estimate_cost(
                    response.tier, response.input_tokens, response.output_tokens
                ),
                estimated=response.estimated,
            )
        )

    return _record


def job_token_totals(db: Session, job_id: int) -> dict[str, int | Decimal]:
    """Aggregate a job's spend -- powers the cost line in the report and UI."""
    row = db.execute(
        select(
            func.coalesce(func.sum(TokenUsage.input_tokens), 0),
            func.coalesce(func.sum(TokenUsage.output_tokens), 0),
            func.coalesce(func.sum(TokenUsage.cost_usd), Decimal("0")),
            func.count(TokenUsage.id),
        ).where(TokenUsage.job_id == job_id)
    ).one()
    return {
        "input_tokens": int(row[0]),
        "output_tokens": int(row[1]),
        "cost_usd": Decimal(row[2]),
        "requests": int(row[3]),
    }
