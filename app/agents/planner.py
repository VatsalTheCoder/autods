"""Planner -- the LLM agent that decides which optional steps run (spec 7.3).

Section 5 started this as two booleans consumed by cleaning, which was enough to
prove the shape of the thing: an LLM makes a decision, that decision is validated
into a Pydantic object, and deterministic code downstream obeys it (spec 6.2 --
"LLM decides, code executes").

Section 7 is where the plan starts to *change the shape of the run*. Three of its
flags -- oversampling, feature selection, sampling -- are read by LangGraph's
conditional edges, so a plan can route the graph past a node entirely. That is
the spec's "dynamic orchestration" (11), and it is why a skipped node is marked
SKIPPED rather than left pending: the difference between "this run did not need
that step" and "this run has not got there yet" is the whole claim.

Every field still has a workable default, so the routing degrades to a sensible
fixed pipeline rather than a broken one.

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
    "You are planning how a tabular dataset should be prepared and explored "
    "before a model is trained on it. You will be given a summary of its "
    "columns. Decide eight things.\n"
    "1. Whether exactly-repeated rows should be removed -- normally yes, but say "
    "no if repeated rows are plausibly meaningful records rather than duplicates.\n"
    "2. Whether columns that are mostly empty should be dropped rather than "
    "filled in -- normally yes.\n"
    "3. Whether looking for natural groupings of rows is worthwhile here -- "
    "normally yes, but say no for a dataset so small or narrow that groups would "
    "be meaningless.\n"
    "4. Which clustering method suits the data: 'kmeans' when every feature is "
    "numeric, 'kprototypes' when there are categorical columns as well.\n"
    "5. Whether the rare outcome should be oversampled during training -- yes "
    "when one class of the target is much rarer than the others, no for a "
    "balanced target or a numeric one.\n"
    "6. Whether only the strongest features should be kept -- yes for a dataset "
    "with many columns relative to its rows, no when there are few columns and "
    "each is likely to matter.\n"
    "7. Whether to train on a random sample rather than every row -- yes only "
    "for a very large dataset, no otherwise.\n"
    "8. Whether to drop rows in the extreme tail of a numeric target. Only for a "
    "numeric target, never for a class one. Say yes when the target's largest "
    "values are implausible or many times its typical value -- a nightly rate of "
    "$10,000 against a median of $106, a zero price, a duration of a million "
    "seconds -- because such rows dominate squared-error training while "
    "representing a different population. Say no when the spread is genuine and "
    "the largest values are exactly what the model is wanted for, such as "
    "predicting insurance losses or peak demand. At most the outermost half a "
    "percent at each end is ever removed.\n"
    "Give a one or two sentence rationale. Do not suggest any other steps."
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
        return _default_plan(report, "No LLM available; used default preparation steps.")

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
        return _default_plan(report, f"Planning fell back to defaults: {exc}")
    except Exception:  # pragma: no cover - defensive; a plan must never fail a job
        logger.exception("Unexpected error while planning; using default plan")
        return _default_plan(report, "Planning fell back to defaults after an unexpected error.")

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


def _default_plan(report: SchemaReport, rationale: str) -> PlannerPlan:
    """The plan used when the model was not consulted or could not answer.

    Not simply ``PlannerPlan()``. Schema detection already measured whether the
    target is skewed (spec 7.1), and that measurement is a better basis for the
    SMOTE decision than a constant is -- so the deterministic path reaches the
    same conclusion an LLM would on the one flag where the data settles it.

    The other two optional steps stay off by default. Both are judgement calls
    about whether a dataset is *large* or *wide*, with no threshold that is right
    across datasets, and switching them on by default would mean quietly
    discarding rows or columns on a run where nothing asked for it.

    Trimming the target's tail stays off for a sharper version of that reason.
    Profiling *has* measured the skew, so the deterministic path could switch it
    on the way it does SMOTE -- but skew alone cannot tell a data-entry artefact
    from a genuine extreme, and on an insurance-loss or peak-demand target the
    largest values are the entire point. Deleting them because a number crossed a
    threshold would answer a different question than the user asked. So this one
    flag waits for either the LLM's judgement or the user's.
    """
    imbalanced = bool(report.class_balance and report.class_balance.imbalanced)
    if imbalanced:
        rationale = f"{rationale} The target is imbalanced, so oversampling is on."
    return PlannerPlan(use_smote=imbalanced, rationale=rationale)


def _dataset_prompt(report: SchemaReport) -> str:
    """A compact dataset summary -- the same terseness discipline as schema detection.

    The planner needs shape and data quality, not values: how many rows, what the
    target is, and which columns are sparse. Sample values would add input tokens
    (spec 10's binding constraint) without informing either decision.
    """
    lines = [
        f"Dataset: {report.n_rows} rows, {report.n_columns} columns.",
        f"Target: {report.suggested_target or 'unknown'} ({report.task_type or 'unknown'} task).",
    ]
    # The imbalance measurement drives the oversampling decision, so it is the
    # one number here that is not merely shape. Without it the model would be
    # guessing at the very thing schema detection already established.
    if report.class_balance:
        balance = report.class_balance
        lines.append(
            f"Target balance: {balance.imbalance_ratio:.1f}:1 between the commonest "
            f"and rarest class ({'imbalanced' if balance.imbalanced else 'balanced'})."
        )
    # The regression counterpart, and for the same reason: decision 8 is about
    # the shape of the target's tail, and asking for it while showing only
    # column names would be asking the model to guess.
    if report.target_distribution:
        dist = report.target_distribution
        lines.append(
            f"Target distribution: min {dist.minimum:,.4g}, median {dist.median:,.4g}, "
            f"max {dist.maximum:,.4g}, skew {dist.skew:.1f}"
            f"{' (heavy right tail)' if dist.heavy_tailed else ''}."
        )
    lines += ["", "Columns:"]
    for col in report.columns:
        lines.append(
            f"- {col.name} (type={col.semantic_type}, unique={col.n_unique}, "
            f"missing={col.null_rate:.0%})"
        )
    return "\n".join(lines)
