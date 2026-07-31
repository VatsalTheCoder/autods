"""The report agent -- it writes the sentences, code writes the numbers (spec 7.12).

Section 5 assembled the whole report by string formatting and said so at the
time: "a report that hallucinates a number is worse than no report, and until
there is a critic in place the safest generator is one that can only restate what
the artifacts say." Section 9 adds the model this file promised, and keeps that
sentence true by giving it a narrower job than "write the report".

**The model never sees a request for a figure and never produces one.** It is
given the same summarised run the critic reads and asked for four things: an
executive summary, a paragraph on the data, a paragraph on the model, and a list
of recommendations. Every metric, every fold count, every table in the finished
document is still formatted by ``ml/report.py`` from the artifacts. So the worst
a bad generation can do is read poorly -- it cannot make the report *wrong*,
which is the failure that would actually matter.

That is also why the prose is stored as its own artifact rather than as a
finished document. ``narrative_report.json`` is the model's contribution alone,
which means a reader can see exactly which sentences came from a model and which
came from the data, and a regenerated report cannot quietly acquire numbers.

Without a key, the deterministic report of Section 5 is what renders -- unchanged
and complete, just less readable. That is the same degradation every agent here
has, and the reason the whole pipeline still works with no LLM at all.
"""

from __future__ import annotations

import logging

from app.agents.summaries import RunSummary
from app.core.llm.base import LLMClient, LLMError, ModelTier, UsageCallback, system, user
from app.core.llm.structured import structured_complete_tiered
from app.ml.contracts import CriticReport, NarrativeReport

logger = logging.getLogger(__name__)

AGENT_NAME = "report_writer"

_SYSTEM = (
    "You are writing the prose for an automated data-science report that a "
    "non-specialist stakeholder will read. You are given a JSON summary of what "
    "the pipeline did and, where one exists, the findings of a reviewer.\n\n"
    "You write sentences. You do not write numbers: every metric, count and table "
    "is inserted around your text by code, from the artifacts. State what things "
    "mean, not what they measure.\n\n"
    "Rules:\n"
    "- Never state a figure. Not a score, not a row count, not a percentage. If a "
    "number is essential to a sentence, refer to it in words ('the headline "
    "score', 'most of the rows') and let the table beside it carry the value.\n"
    "- Never claim the model was validated in a way the summary does not show, "
    "and never call a result significant, proven or production-ready.\n"
    "- If the reviewer raised concerns, reflect them honestly in the summary "
    "rather than writing around them. A report that reads well and omits the "
    "caveat is the failure mode here.\n"
    "- Plain English. No 'leverage', no 'utilize', no 'robust' unless you mean "
    "the statistical sense and can justify it.\n"
    "- Two to four sentences per section. This sits above a page of tables; it is "
    "an orientation, not a replacement for them."
)


def write_narrative(
    summary: RunSummary,
    *,
    critic: CriticReport | None = None,
    client: LLMClient | None = None,
    on_usage: UsageCallback | None = None,
) -> NarrativeReport:
    """Write the report's prose. Never raises -- a report must not fail a job.

    Returns an empty narrative when no model is available, which
    ``ml/report.py`` renders as the Section 5 document: complete, correct and
    plainer.
    """
    if client is None:
        logger.info("Report writer: no LLM configured, the deterministic report stands")
        return NarrativeReport(source="default")

    try:
        result = structured_complete_tiered(
            client,
            [system(_SYSTEM), user(_prompt(summary, critic))],
            NarrativeReport,
            # The large tier first, per spec 6.1: this and the critic are the
            # two reasoning tasks in the pipeline. Stepping down to the small one
            # when rate limited, because the alternative is the Section 5 report
            # with no prose at all -- and these two agents are why the large
            # model's quota runs out first.
            tiers=(ModelTier.LARGE, ModelTier.SMALL, ModelTier.FALLBACK),
            on_usage=on_usage,
        )
    except LLMError as exc:
        logger.warning("Report writer failed, falling back to the plain report: %s", exc)
        return NarrativeReport(source="default")
    except Exception:  # pragma: no cover - defensive; a report must not fail a job
        logger.exception("Unexpected error while writing the report narrative")
        return NarrativeReport(source="default")

    narrative = result.data
    # Stamped here, never by the model: a narrative that fell back must not be
    # recorded as a written one.
    narrative.source = "llm"
    logger.info(
        "Report writer: %d chars of prose, %d recommendation(s)",
        len(narrative.executive_summary) + len(narrative.data_story) + len(narrative.model_story),
        len(narrative.recommendations),
    )
    return narrative


def _prompt(summary: RunSummary, critic: CriticReport | None) -> str:
    parts = [
        "Here is the summary of the run:",
        "",
        summary.as_prompt_block(),
    ]

    if critic is not None and critic.findings:
        # The critic's output is given verbatim rather than summarised again:
        # it is already short, and the report's job is to reflect it faithfully.
        raised = "\n".join(f"- [{f.severity}] {f.finding}" for f in critic.by_severity()[:10])
        parts += [
            "",
            "A reviewer examined this run and raised the following. The executive "
            "summary must not read as though these do not exist:",
            raised,
        ]
        if critic.verdict:
            parts += ["", f"The reviewer's overall verdict: {critic.verdict}"]

    parts += ["", "Write the report's prose."]
    return "\n".join(parts)
