"""The critic -- the one agent whose job is to argue with the pipeline (spec 7.11).

Every other agent here decides something the pipeline then carries out. This one
decides nothing: it reads what the run produced and says what is wrong with it.
That makes it the easiest agent in the project to build badly, because a critique
is prose and prose always looks like output. A model handed "review this machine
learning run" will reliably produce four paragraphs of plausible-sounding
methodology advice whether or not it read anything.

Two things stop that here.

**The measurements come first.** ``measured_findings`` derives what code can
check from thresholds -- a lead inside the fold-to-fold spread, a target so
imbalanced that accuracy flatters, clusters too weak to present as findings, a
SHAP decomposition that does not reconstruct the model's output. Those findings
are established before the model is called, handed to it as fact, and returned
whether or not it answers. The model's job is to add judgement to a review that
already exists, which is the same arrangement that keeps the cluster profiler
honest.

**It is told what it cannot see.** On a wide dataset the summary it reads is
capped, and a reviewer who does not know that will criticise the twenty feature
decisions in front of it as though they were all of them. The omissions travel
into the prompt and into the artifact (``agents/summaries.py``).

The measured findings are also the deterministic fallback. A run without a key
still gets a critic report -- a shorter, duller, entirely checkable one -- which
is the behaviour every agent in this project has.
"""

from __future__ import annotations

import logging

from app.agents.summaries import RunSummary
from app.core.llm.base import LLMClient, LLMError, ModelTier, UsageCallback, system, user
from app.core.llm.structured import structured_complete_tiered
from app.ml.contracts import (
    ClusteringReport,
    CriticFinding,
    CriticReport,
    EvaluationReport,
    ExplainabilityReport,
    FeatureStrategy,
    Leaderboard,
)

logger = logging.getLogger(__name__)

AGENT_NAME = "critic"

# A lead smaller than this many fold standard deviations is not a lead. Set at
# one because that is the point where the winner's and runner-up's fold scores
# visibly overlap -- below it, "the best model" is a statement about which way
# the noise fell, and a leaderboard that does not say so is misleading.
_MEANINGFUL_LEAD_IN_STDS = 1.0

# Above this, the spread across folds is large enough relative to the score that
# the headline number should not be quoted without it.
_UNSTABLE_SPREAD_RATIO = 0.15

# SHAP contributions must reconstruct the model's output. Anything above this is
# no longer a rounding artefact -- see ``ml/explain.py``.
_ADDITIVITY_TOLERANCE = 1e-3

_SYSTEM = (
    "You are a senior data scientist reviewing an automated machine-learning run "
    "before its results are used. You are given a JSON summary of what the "
    "pipeline did, and a list of findings that have already been established by "
    "measurement.\n\n"
    "Your job is to add judgement the measurements cannot supply: whether the "
    "modelling choices suit this dataset, whether the evaluation supports the "
    "claims, what a careful reader should be sceptical of, and what to do next. "
    "The spec asks you specifically to consider simpler models, alternative "
    "strategies, and additional validation.\n\n"
    "Rules:\n"
    "- Never contradict a measured finding or a number in the summary. If you "
    "disagree with an interpretation, say so; do not restate a value differently.\n"
    "- Do not repeat the measured findings back. They are already in the report.\n"
    "- Never invent a number, a column name or a metric that is not in the input.\n"
    "- If the summary says detail was omitted, treat that as a limit on what you "
    "can conclude, and say so rather than reasoning as if you saw everything.\n"
    "- Prefer few specific findings to many generic ones. 'Consider cross-"
    "validation' is worthless here: it was cross-validated, and you can see that.\n"
    "- Every finding needs a concrete recommendation. A finding with no action is "
    "a complaint."
)


def review_run(
    summary: RunSummary,
    *,
    leaderboard: Leaderboard | None = None,
    evaluation: EvaluationReport | None = None,
    clustering: ClusteringReport | None = None,
    explainability: ExplainabilityReport | None = None,
    features: FeatureStrategy | None = None,
    client: LLMClient | None = None,
    on_usage: UsageCallback | None = None,
) -> CriticReport:
    """Review a completed run. Never raises -- a critique must not fail a job.

    The measured findings are computed from the artifacts rather than the summary,
    because the summary is capped and a threshold check should see the real
    numbers even when the model cannot.
    """
    measured = measured_findings(
        leaderboard=leaderboard,
        evaluation=evaluation,
        clustering=clustering,
        explainability=explainability,
        features=features,
    )
    report = CriticReport(
        findings=measured,
        detail_level=summary.detail_level,
        omissions=list(summary.notes),
        source="default",
    )

    if client is None:
        logger.info("Critic: no LLM configured, reporting %d measured finding(s)", len(measured))
        report.verdict = _measured_verdict(measured)
        report.confidence = "low"
        return report

    try:
        result = structured_complete_tiered(
            client,
            [system(_SYSTEM), user(_prompt(summary, measured))],
            _CriticAnswer,
            # The large tier first: this is the reasoning task the spec reserves
            # it for (6.1), and the one place a bigger model earns its latency.
            # Stepping down to the small one when rate limited, because the
            # alternative is a review that is only threshold checks -- and on the
            # free tier the large model's quota is the first to run out.
            tiers=(ModelTier.LARGE, ModelTier.SMALL),
            on_usage=on_usage,
        )
    except LLMError as exc:
        logger.warning("Critic LLM call failed, keeping measured findings only: %s", exc)
        report.verdict = _measured_verdict(measured)
        report.confidence = "low"
        report.omissions.append(f"The written review was not produced: {exc}")
        return report
    except Exception:  # pragma: no cover - defensive; a critique must not fail a job
        logger.exception("Unexpected error while reviewing; keeping measured findings")
        report.verdict = _measured_verdict(measured)
        report.confidence = "low"
        return report

    answer = result.data
    # Measured findings first and unaltered. The model contributes to the review;
    # it does not get to edit the parts that were checked.
    report.findings = [*measured, *(f for f in answer.findings if not f.measured)]
    report.verdict = answer.verdict or _measured_verdict(measured)
    report.confidence = answer.confidence
    report.strengths = answer.strengths
    report.recommended_next_steps = answer.recommended_next_steps
    report.source = "llm"

    logger.info(
        "Critic: %d finding(s) (%d measured), %d blocker(s), confidence %s",
        len(report.findings),
        len(measured),
        len(report.blockers()),
        report.confidence,
    )
    return report


class _CriticAnswer(CriticReport):
    """What the model is asked for -- the report minus the fields code owns.

    Subclassed rather than redefined so the two cannot drift. ``source``,
    ``detail_level`` and ``omissions`` are stamped by this module afterwards: a
    model that set its own ``source`` could record a fallback as a written review.
    """


# ---- What code can establish without asking anyone ---------------------------


def measured_findings(
    *,
    leaderboard: Leaderboard | None = None,
    evaluation: EvaluationReport | None = None,
    clustering: ClusteringReport | None = None,
    explainability: ExplainabilityReport | None = None,
    features: FeatureStrategy | None = None,
) -> list[CriticFinding]:
    """The findings a threshold can decide, computed before the model is asked.

    Each is a claim about numbers that are in the artifacts, so each is checkable
    by whoever reads the report. They are the floor of the critique's quality:
    with no key at all, this is the critique.
    """
    findings: list[CriticFinding] = []

    if leaderboard is not None:
        findings += _leaderboard_findings(leaderboard)
    if evaluation is not None:
        findings += _evaluation_findings(evaluation)
    if clustering is not None:
        findings += _clustering_findings(clustering)
    if explainability is not None:
        findings += _explainability_findings(explainability)
    if features is not None:
        findings += _feature_findings(features)

    return findings


def _leaderboard_findings(board: Leaderboard) -> list[CriticFinding]:
    """Chiefly: is the winner actually winning?"""
    findings: list[CriticFinding] = []
    ranked = [e for e in board.entries if not e.error]

    failed = [e for e in board.entries if e.error]
    if failed:
        findings.append(
            CriticFinding(
                area="modelling",
                severity="note",
                finding=(
                    f"{len(failed)} of {len(board.entries)} candidate models could not be "
                    f"trained: {', '.join(e.model_name for e in failed)}."
                ),
                recommendation=(
                    "Check the recorded error for each. A model that fails on this "
                    "dataset is evidence about the data, not only about the model."
                ),
                measured=True,
            )
        )

    if len(ranked) >= 2:
        winner, runner_up = ranked[0], ranked[1]
        lead = winner.score - runner_up.score
        spread = max(winner.std, runner_up.std)
        if spread > 0 and lead < _MEANINGFUL_LEAD_IN_STDS * spread:
            findings.append(
                CriticFinding(
                    area="modelling",
                    severity="concern",
                    finding=(
                        f"{winner.model_name} leads {runner_up.model_name} by "
                        f"{lead:.4f} on {board.primary_metric}, which is inside the "
                        f"fold-to-fold spread ({spread:.4f}). The ranking between "
                        "them is not established by this evidence."
                    ),
                    recommendation=(
                        f"Treat the top two as tied and prefer {runner_up.model_name} "
                        "if it is simpler, faster or easier to explain. Repeated or "
                        "nested cross-validation would separate them if it matters."
                    ),
                    measured=True,
                )
            )
    return findings


def _evaluation_findings(report: EvaluationReport) -> list[CriticFinding]:
    findings: list[CriticFinding] = []
    summary = report.metrics.get(report.primary_metric)

    if summary is not None and summary.mean:
        ratio = abs(summary.std / summary.mean)
        if ratio > _UNSTABLE_SPREAD_RATIO:
            findings.append(
                CriticFinding(
                    area="evaluation",
                    severity="concern",
                    finding=(
                        f"{report.primary_metric} varies substantially across folds "
                        f"({summary.mean:.4f} ± {summary.std:.4f}). The headline number "
                        "is less stable than one figure suggests."
                    ),
                    recommendation=(
                        "Quote the score with its spread wherever it appears, and treat "
                        "small differences against other models as noise."
                    ),
                    measured=True,
                )
            )

    if report.n_rows and report.n_rows < 200:
        findings.append(
            CriticFinding(
                area="evaluation",
                severity="concern",
                finding=(
                    f"The model was evaluated on {report.n_rows} rows across "
                    f"{report.n_folds} folds, so each held-out fold is small."
                ),
                recommendation=(
                    "Read every metric here as approximate. Repeated k-fold would give "
                    "a steadier estimate at this size."
                ),
                measured=True,
            )
        )

    if report.warnings:
        findings.append(
            CriticFinding(
                area="evaluation",
                severity="note",
                finding=f"The evaluation recorded {len(report.warnings)} caveat(s).",
                recommendation="Read them alongside the scores; they qualify what was measured.",
                measured=True,
            )
        )
    return findings


def _clustering_findings(report: ClusteringReport) -> list[CriticFinding]:
    if report.k and report.is_weak():
        return [
            CriticFinding(
                area="data_quality",
                severity="note",
                finding=(
                    f"The {report.k} groups are weakly separated (silhouette "
                    f"{report.silhouette:.2f}). K-means partitions pure noise at around "
                    "0.36, so this is not clearly structure."
                ),
                recommendation=(
                    "Present the groups as exploratory at most, and do not describe them "
                    "as segments the business could act on."
                ),
                measured=True,
            )
        ]
    return []


def _explainability_findings(report: ExplainabilityReport) -> list[CriticFinding]:
    findings: list[CriticFinding] = []

    if report.additivity_max_error > _ADDITIVITY_TOLERANCE:
        findings.append(
            CriticFinding(
                area="explainability",
                severity="blocker",
                finding=(
                    "The SHAP contributions do not reconstruct the model's output "
                    f"(largest gap {report.additivity_max_error:.4f}). The explanation "
                    "is not a decomposition of this model's predictions."
                ),
                recommendation=(
                    "Do not quote the feature importances until this is resolved. "
                    "Check the explainer's baseline and the model family dispatch."
                ),
                measured=True,
            )
        )

    top = report.global_importance[:1]
    if top and top[0].share > 0.6:
        findings.append(
            CriticFinding(
                area="explainability",
                severity="concern",
                finding=(
                    f"`{top[0].feature}` alone accounts for {top[0].share:.0%} of the "
                    "model's attributed influence."
                ),
                recommendation=(
                    "Check that this column is legitimately available before the "
                    "outcome occurs. A single dominant feature is the usual signature "
                    "of target leakage in the source data."
                ),
                measured=True,
            )
        )
    return findings


def _feature_findings(strategy: FeatureStrategy) -> list[CriticFinding]:
    findings: list[CriticFinding] = []
    if strategy.rejected_columns:
        findings.append(
            CriticFinding(
                area="features",
                severity="note",
                finding=(
                    f"The strategy model referred to {len(strategy.rejected_columns)} "
                    "column(s) that do not exist; code rejected them."
                ),
                recommendation=(
                    "No action needed -- this is the validation working. Worth watching "
                    "if the count grows, as it indicates prompt drift."
                ),
                measured=True,
            )
        )
    if strategy.overrides:
        findings.append(
            CriticFinding(
                area="features",
                severity="note",
                finding=(
                    f"{len(strategy.overrides)} feature decision(s) were overruled by code "
                    "because the data could not support them."
                ),
                recommendation=(
                    "Review the recorded reasons to see whether the prompt should change."
                ),
                measured=True,
            )
        )
    return findings


def _measured_verdict(findings: list[CriticFinding]) -> str:
    """The one-line summary when there is no model to write one.

    Deliberately flat. Without an LLM this report is a list of threshold results,
    and dressing it up as a considered review would misrepresent what produced it.
    """
    if not findings:
        return (
            "No automated check found a problem with this run. This is a checklist "
            "result, not a considered review -- no language model was available."
        )
    blockers = sum(1 for f in findings if f.severity == "blocker")
    concerns = sum(1 for f in findings if f.severity == "concern")
    parts = []
    if blockers:
        parts.append(f"{blockers} blocker(s)")
    if concerns:
        parts.append(f"{concerns} concern(s)")
    parts.append(f"{len(findings)} finding(s) in total")
    return (
        "Automated checks raised " + ", ".join(parts) + ". This is a checklist "
        "result, not a considered review -- no language model was available."
    )


def _prompt(summary: RunSummary, measured: list[CriticFinding]) -> str:
    established = (
        "\n".join(f"- [{f.severity}] {f.area}: {f.finding}" for f in measured)
        or "- (none; no threshold check fired)"
    )
    return (
        "Here is the summary of the run:\n\n"
        f"{summary.as_prompt_block()}\n\n"
        "These findings are already established by measurement and are in the "
        "report already. Do not repeat them; do not contradict them:\n"
        f"{established}\n\n"
        "Review the run and return your findings."
    )
