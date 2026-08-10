"""The dataset chat endpoint (spec 7.13, 12.1).

One POST per question. Deliberately not a streaming or socket interface: an
answer here is two model calls and a database read, the whole thing takes a
couple of seconds, and a streaming transport would add a second failure mode to
a feature whose interesting problem is routing rather than delivery.

The transcript is persisted per job, so a conversation survives a page reload and
so the routing decisions can be reviewed after the fact.
"""

from __future__ import annotations

import logging

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.agents.chat import AGENT_NAME, NOT_INDEXED_ANSWER, answer_question
from app.core.db import get_db
from app.core.llm.factory import get_optional_llm
from app.core.llm.usage import make_usage_recorder
from app.models.chat_message import ChatMessage, ChatRoute
from app.models.job import Job
from app.services.artifacts import CLEANED_DATASET_ARTIFACT, load_artifact_bytes
from app.services.csv_validation import read_frame
from app.services.retrieval import indexed_count, search

logger = logging.getLogger(__name__)

router = APIRouter(tags=["chat"])


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)


class ChatResponse(BaseModel):
    question: str
    answer: str
    # Which tool answered, surfaced to the client rather than kept internal: the
    # UI shows it, because "this came from a calculation" and "this came from the
    # report" are different kinds of claim and a reader should know which.
    route: str
    grounding: str = ""


class ChatMessageOut(BaseModel):
    id: int
    question: str
    answer: str
    route: str
    grounding: str = ""


def _require_job(db: Session, job_id: int) -> Job:
    job = db.get(Job, job_id)
    if job is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"No job {job_id}.")
    return job


def _cleaned_frame(db: Session, job_id: int) -> pd.DataFrame | None:
    """The dataset the pandas tool computes over.

    The *cleaned* dataset, not the upload. Answers should agree with the report,
    and the report describes the cleaned data -- an average computed over rows
    that cleaning dropped would contradict it for no good reason.
    """
    try:
        data = load_artifact_bytes(db, job_id, CLEANED_DATASET_ARTIFACT)
    except Exception:
        logger.exception("[job %s] the cleaned dataset could not be loaded for chat", job_id)
        return None
    if data is None:
        return None
    # The same reader the pipeline used, not a bare ``pd.read_csv``. This frame
    # is the cleaned dataset *we* wrote, so a column whose value is the string
    # "NA" -- no pool, no alley -- would otherwise come back as a gap here and
    # the chat agent would answer "1,453 missing" about a column with none.
    return read_frame(data)


@router.post(
    "/jobs/{job_id}/chat",
    response_model=ChatResponse,
    summary="Ask a question about a finished run",
)
def ask(job_id: int, request: ChatRequest, db: Session = Depends(get_db)) -> ChatResponse:
    """Route a question to retrieval or pandas, answer it, and record both.

    A run that was never indexed answers with an explanation rather than a 404:
    the job exists and its results are readable, it is only the chat that has
    nothing to work from, and a 404 would suggest the job itself was missing.
    """
    _require_job(db, job_id)
    question = request.question.strip()

    if indexed_count(db, job_id) == 0:
        return _record(db, job_id, question, NOT_INDEXED_ANSWER, ChatRoute.REFUSED, "not-indexed")

    frame = _cleaned_frame(db, job_id)
    passages = search(db, job_id, question)

    answer = answer_question(
        question,
        passages=passages,
        frame=frame,
        columns=list(frame.columns) if frame is not None else [],
        client=get_optional_llm(),
        on_usage=make_usage_recorder(db, job_id, AGENT_NAME),
    )
    return _record(db, job_id, question, answer.answer, answer.route, answer.grounding)


def _record(
    db: Session,
    job_id: int,
    question: str,
    answer: str,
    route: ChatRoute,
    grounding: str,
) -> ChatResponse:
    db.add(
        ChatMessage(
            job_id=job_id, question=question, answer=answer, route=route, grounding=grounding
        )
    )
    db.commit()
    return ChatResponse(question=question, answer=answer, route=route, grounding=grounding)


@router.get(
    "/jobs/{job_id}/chat",
    response_model=list[ChatMessageOut],
    summary="The conversation so far",
)
def history(job_id: int, db: Session = Depends(get_db)) -> list[ChatMessageOut]:
    """The transcript, oldest first, so a reloaded page reads in order."""
    _require_job(db, job_id)
    rows = db.execute(
        select(ChatMessage).where(ChatMessage.job_id == job_id).order_by(ChatMessage.id)
    ).scalars()
    return [
        ChatMessageOut(
            id=row.id,
            question=row.question,
            answer=row.answer,
            route=row.route,
            grounding=row.grounding,
        )
        for row in rows
    ]


@router.get(
    "/jobs/{job_id}/chat/status",
    summary="Whether this run can be asked about",
)
def status_(job_id: int, db: Session = Depends(get_db)) -> dict:
    """How many passages the run has, so the UI can explain an empty chat."""
    _require_job(db, job_id)
    count = indexed_count(db, job_id)
    return {"job_id": job_id, "indexed_passages": count, "ready": count > 0}
