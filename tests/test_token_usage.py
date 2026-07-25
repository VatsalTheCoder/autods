"""Tests for cost tracking.

Cost arithmetic is pure and always runs. The recording + aggregation tests need
a real Postgres (they assert rows actually land), so they skip when the stack is
down, keeping the suite green outside Docker -- the same pattern as the upload
and storage tests.
"""

from __future__ import annotations

import uuid
from decimal import Decimal

import pytest

from app.core.llm import ModelTier
from app.core.llm.base import LLMResponse
from app.core.llm.usage import (
    PRICE_PER_MILLION,
    estimate_cost,
    job_token_totals,
    make_usage_recorder,
)


class TestCostArithmetic:
    def test_free_tier_costs_nothing(self):
        assert estimate_cost(ModelTier.SMALL, 10_000, 5_000) == Decimal("0")
        assert estimate_cost(ModelTier.LARGE, 10_000, 5_000) == Decimal("0")

    def test_priced_tier_computes_per_million(self, monkeypatch):
        # Pretend the large tier is billed $2/1M input, $6/1M output.
        monkeypatch.setitem(PRICE_PER_MILLION, ModelTier.LARGE, (Decimal("2"), Decimal("6")))
        # 1,000,000 input + 1,000,000 output -> 2 + 6 = 8.
        assert estimate_cost(ModelTier.LARGE, 1_000_000, 1_000_000) == Decimal("8.000000")

    def test_cost_is_quantised_to_six_places(self, monkeypatch):
        monkeypatch.setitem(PRICE_PER_MILLION, ModelTier.SMALL, (Decimal("1"), Decimal("0")))
        # 1 token at $1/1M = 0.000001 exactly, at the column's precision.
        assert estimate_cost(ModelTier.SMALL, 1, 0) == Decimal("0.000001")


# --- DB-backed tests ---------------------------------------------------------

pytest.importorskip("sqlalchemy")
from app.core.db import SessionLocal, database_healthy  # noqa: E402
from app.models.job import Job, JobStatus  # noqa: E402
from app.models.token_usage import TokenUsage  # noqa: E402
from app.models.user import DEV_USER_ID  # noqa: E402

requires_db = pytest.mark.skipif(
    not database_healthy(),
    reason="needs Postgres (start the stack with `make up`)",
)


def _make_job(db) -> Job:
    job = Job(
        user_id=DEV_USER_ID,
        original_filename="usage_test.csv",
        s3_key=f"tests/{uuid.uuid4()}.csv",
        size_bytes=1,
        status=JobStatus.UPLOADED,
    )
    db.add(job)
    db.commit()
    db.refresh(job)
    return job


@requires_db
class TestUsageRecording:
    def test_recorder_logs_a_row_per_response(self):
        with SessionLocal() as db:
            job = _make_job(db)
            record = make_usage_recorder(db, job.id, agent="planner")

            record(
                LLMResponse(
                    text="x",
                    model="gemma-small",
                    tier=ModelTier.SMALL,
                    input_tokens=120,
                    output_tokens=30,
                    estimated=False,
                )
            )
            record(
                LLMResponse(
                    text="y",
                    model="gemma-small",
                    tier=ModelTier.SMALL,
                    input_tokens=80,
                    output_tokens=20,
                    estimated=False,
                )
            )
            db.commit()

            rows = (
                db.query(TokenUsage)
                .filter(TokenUsage.job_id == job.id)
                .order_by(TokenUsage.id)
                .all()
            )
            assert len(rows) == 2
            assert rows[0].agent == "planner"
            assert rows[0].input_tokens == 120

            db.delete(job)  # cascade removes the usage rows
            db.commit()

    def test_totals_aggregate_across_agents(self):
        with SessionLocal() as db:
            job = _make_job(db)
            for agent, (i, o) in {"planner": (100, 10), "critic": (200, 40)}.items():
                make_usage_recorder(db, job.id, agent=agent)(
                    LLMResponse(
                        text="x",
                        model="m",
                        tier=ModelTier.LARGE,
                        input_tokens=i,
                        output_tokens=o,
                        estimated=False,
                    )
                )
            db.commit()

            totals = job_token_totals(db, job.id)
            assert totals["input_tokens"] == 300
            assert totals["output_tokens"] == 50
            assert totals["requests"] == 2
            assert totals["cost_usd"] == Decimal("0")  # free tier

            db.delete(job)
            db.commit()
