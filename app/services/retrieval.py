"""Indexing a finished run, and searching it (spec 7.13).

Storage is Postgres via pgvector, so a search is an ordinary SQL query with an
ordinary ``ORDER BY`` -- which means retrieval participates in the same session
and the same transaction as everything else, and a job's passages are deleted by
the same cascade that deletes the job.

**Search is exact, not approximate.** A run produces a few dozen passages, and on
that scale a sequential scan with a distance operator beats any ANN index: IVFFlat
needs training data proportional to what it indexes, and would return approximate
neighbours in exchange for nothing. The migration says the same thing next to the
index that is deliberately absent.

Distance is cosine. The vectors come out of BGE L2-normalised, so cosine and inner
product rank identically here -- cosine is chosen because it stays correct if a
future model returns unnormalised vectors, which is the sort of change that
otherwise degrades silently.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from app.ml.chunking import Chunk
from app.ml.embedding import embed_passages, embed_query
from app.models.run_chunk import RunChunk

logger = logging.getLogger(__name__)

# How many passages a question is answered from. Enough that a question spanning
# two sections still sees both; small enough that the model is not handed a page
# of near-misses to average into something vague.
DEFAULT_TOP_K = 6

# A loose sanity cut, and deliberately *not* the mechanism that refuses
# out-of-scope questions.
#
# The obvious design is to refuse when the nearest passage is too far away. It
# does not work, and the numbers say so. Measured over a real run's 33 passages:
# in-scope questions produced best-hit distances from 0.160 to 0.501, and
# out-of-scope ones ("who won the world cup?", "how do I bake bread?") from 0.488
# to 0.591. **The two bands overlap.** Any threshold that refuses the world cup
# also refuses "what should I do next?", which is a question the run answers.
#
# The reason is structural: cosine distance measures how alike two things are,
# and every passage here is prose about a dataset, so an unrelated question is
# still roughly as far from all of them as a vague relevant question is. There is
# no gap to put a line in.
#
# So refusal belongs to the model that reads the passages, which can tell that
# none of them mention the stock market -- see ``agents/chat.py``. This cut only
# stops genuinely absurd matches from being retrieved at all.
MAX_DISTANCE = 0.85


@dataclass(frozen=True)
class Retrieved:
    """One passage, and how far it was from the question."""

    chunk_id: int
    source: str
    heading: str
    content: str
    distance: float

    def as_context(self) -> str:
        """How the passage is presented to the model answering from it."""
        return f"[{self.chunk_id}] {self.heading}\n{self.content}"


def index_run(db: Session, job_id: int, chunks: list[Chunk]) -> int:
    """Embed a run's passages and store them. Returns how many were stored.

    Replaces anything already indexed for the job, so re-running is safe and
    idempotent -- a job re-indexed after a fix should end up with one copy of
    each passage, not two sets competing in the same search.
    """
    db.execute(delete(RunChunk).where(RunChunk.job_id == job_id))

    if not chunks:
        logger.info("[job %s] nothing to index", job_id)
        return 0

    vectors = embed_passages([chunk.embedding_text() for chunk in chunks])
    db.add_all(
        [
            RunChunk(
                job_id=job_id,
                source=chunk.source,
                heading=chunk.heading,
                content=chunk.content,
                ordinal=chunk.ordinal,
                embedding=vector,
            )
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]
    )
    logger.info("[job %s] indexed %d passages", job_id, len(chunks))
    return len(chunks)


def search(
    db: Session,
    job_id: int,
    question: str,
    *,
    top_k: int = DEFAULT_TOP_K,
    max_distance: float = MAX_DISTANCE,
) -> list[Retrieved]:
    """The passages most like the question, nearest first.

    Scoped to one job. Passages from another run describe another dataset's
    numbers, so a search that crossed jobs would produce answers that are
    fluent, plausible and about the wrong data.
    """
    vector = embed_query(question)
    distance = RunChunk.embedding.cosine_distance(vector)

    rows = db.execute(
        select(RunChunk, distance.label("distance"))
        .where(RunChunk.job_id == job_id)
        .order_by(distance)
        .limit(top_k)
    ).all()

    return [
        Retrieved(
            chunk_id=chunk.id,
            source=chunk.source,
            heading=chunk.heading,
            content=chunk.content,
            distance=float(score),
        )
        for chunk, score in rows
        if float(score) <= max_distance
    ]


def indexed_count(db: Session, job_id: int) -> int:
    """How many passages a job has. Zero means the chat has nothing to answer from."""
    return len(db.execute(select(RunChunk.id).where(RunChunk.job_id == job_id)).all())
