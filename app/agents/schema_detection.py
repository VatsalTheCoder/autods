"""Schema Detection -- the first real agent (spec 7.1, build-plan Section 3).

It reads the uploaded CSV and works out what each column is, which column is
probably the target, whether the problem is classification or regression, and
which columns hold personal data. Then the pipeline *stops and asks the user* to
confirm -- but that pause is not implemented as a frozen graph. Schema detection
runs synchronously during upload, the user confirms, and only then does the
heavy job launch. The human checkpoint lives in the gap between two jobs.

Structure follows the project's core rule (spec 6.2): deterministic code does
the measurable work (dtypes, cardinality, nulls, regex PII, class balance), and
the LLM only adds judgement code cannot -- what a column *means*, and semantic
PII like a name column that matches no regex.

Crucially, the LLM pass is **best-effort**. This runs in the request path, so a
missing API key, a rate-limit, or a malformed reply must never fail the upload:
on any such problem we return the deterministic report and let the user fill the
gaps at the confirmation screen. That is the whole reason profiling is exact and
standalone.
"""

from __future__ import annotations

import logging

import pandas as pd

from app.agents.schema_models import (
    LLMSchemaInference,
    SchemaReport,
)
from app.core.llm.base import LLMClient, LLMError, ModelTier, UsageCallback, system, user
from app.core.llm.structured import structured_complete
from app.services.profiling import profile_dataset

logger = logging.getLogger(__name__)

AGENT_NAME = "schema_detection"

_SYSTEM = (
    "You are a data scientist inspecting a tabular dataset before modelling. "
    "You will be given each column's name, inferred type, and a few sample "
    "values. For every column, state in one short sentence what it represents, "
    "and whether it is personally identifiable information (a person's name, "
    "email, phone, address, or similar). Then pick the single column that is "
    "most likely the prediction target, and whether predicting it is a "
    "classification or a regression problem. Only ever use column names exactly "
    "as given; never invent one."
)


def detect_schema(
    frame: pd.DataFrame,
    *,
    client: LLMClient | None = None,
    on_usage: UsageCallback | None = None,
) -> SchemaReport:
    """Profile the dataset and, if a client is given, enrich it with the LLM.

    Passing ``client=None`` (or any LLM failure) yields the deterministic report
    unchanged -- a complete, editable schema either way.
    """
    report = profile_dataset(frame)
    if client is None:
        return report
    try:
        return _enrich(report, client, on_usage=on_usage)
    except LLMError as exc:
        # Includes LLMConfigError (no key), RateLimitError, StructuredOutputError.
        logger.warning("Schema LLM enrichment skipped: %s", exc)
        return report
    except Exception:  # pragma: no cover - defensive; upload must not die here
        logger.exception("Unexpected error during schema LLM enrichment; using profile only")
        return report


def _enrich(
    report: SchemaReport, client: LLMClient, *, on_usage: UsageCallback | None
) -> SchemaReport:
    """Fold the LLM's judgement into the deterministic report.

    Everything the model returns is validated against the *actual* columns, and
    anything that does not line up is ignored rather than trusted -- an invented
    target falls back to the heuristic guess, an insight for a non-existent
    column is dropped. The user is the final authority regardless.
    """
    known = set(report.column_names())
    result = structured_complete(
        client,
        [system(_SYSTEM), user(_columns_prompt(report))],
        LLMSchemaInference,
        tier=ModelTier.SMALL,
        on_usage=on_usage,
    )
    inference = result.data

    # Merge per-column meanings and semantic-PII flags by name.
    insights = {ins.name: ins for ins in inference.columns if ins.name in known}
    for col in report.columns:
        ins = insights.get(col.name)
        if ins is None:
            continue
        col.meaning = ins.meaning
        if ins.is_pii and not col.is_pii:
            # LLM caught PII the regex could not (e.g. a name column). Default
            # it to excluded too, matching how regex PII is treated.
            col.is_pii = True
            col.pii_type = ins.pii_type or "semantic"
            col.exclude = True

    # Trust the LLM's target/task only if the target names a real column.
    if inference.suggested_target in known:
        report.suggested_target = inference.suggested_target
        report.task_type = inference.task_type

    report.llm_enriched = True
    return report


def _columns_prompt(report: SchemaReport) -> str:
    """A compact, one-line-per-column description for the model.

    Kept terse on purpose (spec 10): this is the whole prompt body, and padding
    it with raw data would eat into the input-token budget for no benefit -- a
    handful of sample values per column is enough to reason about meaning.
    """
    lines = [f"Dataset: {report.n_rows} rows, {report.n_columns} columns.", "", "Columns:"]
    for col in report.columns:
        samples = ", ".join(str(v) for v in col.sample_values[:3])
        lines.append(
            f"- {col.name} (type={col.semantic_type}, unique={col.n_unique}, "
            f"nulls={col.null_rate:.0%}) e.g. {samples}"
        )
    return "\n".join(lines)
