"""Token-usage model -- one row per LLM request.

This is the "cost-aware AI" talking point made real (spec section 13). Every
call any agent makes is logged here: which agent, which model tier, how many
input and output tokens, and the money it would cost. On the free tier that
cost is zero, but the machinery is what matters -- switching to a paid tier or a
self-hosted GPU is then a pricing-table change, not new plumbing.

Retries count. The structured-output layer re-asks on malformed output and each
re-ask is a real request that lands its own row here, so a prompt that needed
three tries shows three rows -- the true spend, not a flattering one.
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import TYPE_CHECKING

from sqlalchemy import Boolean, DateTime, ForeignKey, Integer, Numeric, String, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.db import Base

if TYPE_CHECKING:
    from app.models.job import Job


class TokenUsage(Base):
    __tablename__ = "token_usage"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    job_id: Mapped[int] = mapped_column(
        ForeignKey("jobs.id", ondelete="CASCADE"), nullable=False, index=True
    )

    # Which agent spent the tokens: "schema_detection", "planner", "critic", ...
    agent: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    model: Mapped[str] = mapped_column(String(128), nullable=False)
    tier: Mapped[str] = mapped_column(String(16), nullable=False)

    input_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    output_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    # Numeric, not float: this is money, and floats accumulate rounding error.
    cost_usd: Mapped[Decimal] = mapped_column(Numeric(12, 6), nullable=False, default=Decimal("0"))

    # True when the token counts are heuristic (FakeLLM, or a provider that
    # returned no usage metadata) rather than provider-reported.
    estimated: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    job: Mapped[Job] = relationship(back_populates="token_usage")

    def __repr__(self) -> str:
        return (
            f"<TokenUsage job={self.job_id} agent={self.agent!r} "
            f"in={self.input_tokens} out={self.output_tokens}>"
        )
