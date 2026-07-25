"""Planner -- the LLM agent that decides which optional steps run (spec 7.3).

Minimal on purpose. In Section 5 the plan is two booleans consumed by cleaning,
which is enough to prove the shape of the thing: an LLM makes a decision, that
decision is validated into a Pydantic object, and deterministic code downstream
obeys it (spec 6.2 -- "LLM decides, code executes"). Sections 7 and 8 widen the
plan to SMOTE and the model roster, once there is more than one option to pick.

Like schema detection, the LLM pass is **best-effort** -- but for a different
reason. Schema detection degrades because it runs in the request path; this
degrades because a plan is an *optimisation*, not a requirement. Every flag has
a sane default, so a missing API key or a malformed reply produces a plan that
still runs a correct pipeline. A dataset does not become unanalysable because a
free-tier model was rate-limited.
"""

from __future__ import annotations

import logging

from app.agents.schema_models import SchemaReport
from app.core.llm.base import LLMClient, LLMError, ModelTier, UsageCallback, system, user
from app.core.llm.structured import structured_complete
from app.ml.contracts import PlannerPlan

logger = logging.getLogger(__name__)

AGENT_NAME = "planner"

_SYSTEM = (
    "You are planning the preparation of a tabular dataset before a model is "
    "trained on it. You will be given a summary of the dataset's columns. "
    "Decide two things. First, whether exactly-repeated rows should be removed "
    "-- normally yes, but say no if repeated rows are plausibly meaningful "
    "records rather than duplicates. Second, whether columns that are mostly "
    "empty should be dropped rather than filled in -- normally yes. Give a one "
    "or two sentence rationale. Do not suggest any other steps."
)


def make_plan(
    report: SchemaReport,
    *,
    client: LLMClient | None = None,
    on_usage: UsageCallback | None = None,
) -> PlannerPlan:
    """Ask the LLM for a plan, falling back to defaults it cannot be obtained.

    Never raises. The returned plan's ``source`` says whether the model was
    actually consulted, so the report can distinguish a decision from a default.
    """
    if client is None:
        logger.info("Planner: no LLM configured, using default plan")
        return PlannerPlan(rationale="No LLM available; used default preparation steps.")

    try:
        result = structured_complete(
            client,
            [system(_SYSTEM), user(_dataset_prompt(report))],
            PlannerPlan,
            tier=ModelTier.SMALL,
            on_usage=on_usage,
        )
    except LLMError as exc:
        logger.warning("Planner LLM call failed, using default plan: %s", exc)
        return PlannerPlan(rationale=f"Planning fell back to defaults: {exc}")
    except Exception:  # pragma: no cover - defensive; a plan must never fail a job
        logger.exception("Unexpected error while planning; using default plan")
        return PlannerPlan(rationale="Planning fell back to defaults after an unexpected error.")

    # The model does not get to set ``source`` -- if it hallucinates the field we
    # would be recording a default plan as an LLM decision. Stamped here, where
    # the truth is known.
    plan = result.data.model_copy(update={"source": "llm"})
    logger.info(
        "Planner: duplicates=%s high_null=%s (%d attempt(s))",
        plan.drop_duplicate_rows,
        plan.drop_high_null_columns,
        result.attempts,
    )
    return plan


def _dataset_prompt(report: SchemaReport) -> str:
    """A compact dataset summary -- the same terseness discipline as schema detection.

    The planner needs shape and data quality, not values: how many rows, what the
    target is, and which columns are sparse. Sample values would add input tokens
    (spec 10's binding constraint) without informing either decision.
    """
    lines = [
        f"Dataset: {report.n_rows} rows, {report.n_columns} columns.",
        f"Target: {report.suggested_target or 'unknown'} ({report.task_type or 'unknown'} task).",
        "",
        "Columns:",
    ]
    for col in report.columns:
        lines.append(
            f"- {col.name} (type={col.semantic_type}, unique={col.n_unique}, "
            f"missing={col.null_rate:.0%})"
        )
    return "\n".join(lines)
