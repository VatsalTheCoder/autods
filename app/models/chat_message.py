"""The chat transcript for a run (spec 7.13, 12.2).

One row per exchange: the question, the answer, and -- the part worth having --
**which tool answered and what it was grounded in**.

Routing between semantic retrieval and a pandas query is the interesting problem
in Section 10, and a transcript that recorded only the prose would make the
router's behaviour unauditable after the fact. Storing the route and the
grounding means a wrong answer can be diagnosed as *the wrong tool* rather than
guessed at.
"""

from __future__ import annotations

import enum
from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import DateTime, ForeignKey, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.db import Base

if TYPE_CHECKING:
    from app.models.job import Job


class ChatRoute(enum.StrEnum):
    """Which tool produced the answer."""

    # Semantic retrieval over the run's written output: "why was X important?"
    RAG = "rag"
    # Arithmetic over the cleaned dataset: "what is the average age?"
    PANDAS = "pandas"
    # Neither could be attempted -- an unanswerable or out-of-scope question.
    # A first-class outcome rather than an error: refusing to answer is the
    # correct response to "what will the stock market do tomorrow?", and it
    # should appear in the transcript as a decision rather than a failure.
    REFUSED = "refused"


class ChatMessage(Base):
    """One question and its answer."""

    __tablename__ = "chat_history"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    job_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("jobs.id", ondelete="CASCADE"), index=True, nullable=False
    )

    question: Mapped[str] = mapped_column(Text, nullable=False)
    answer: Mapped[str] = mapped_column(Text, nullable=False)
    route: Mapped[str] = mapped_column(String(16), nullable=False)

    # Retrieved chunk ids, or the pandas expression that was evaluated. Free
    # text because the two tools ground their answers in different kinds of
    # thing, and forcing them into one shape would lose what each is saying.
    grounding: Mapped[str] = mapped_column(Text, nullable=False, default="")

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    job: Mapped[Job] = relationship(back_populates="chat_messages")

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<ChatMessage job={self.job_id} route={self.route!r}>"
