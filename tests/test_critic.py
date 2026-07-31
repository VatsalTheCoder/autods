"""Tests for the critic (spec 7.11).

A critique is prose, and prose always looks like output. A model asked to review
a machine-learning run will produce four confident paragraphs whether or not it
read anything, so the tests that matter here are not "did it say something" but:

* are the **measured** findings there regardless of what the model did, and are
  they unaltered by it;
* does the report say what the reviewer could not see;
* does a run with no model still get a usable, checkable critique.

The thresholds themselves are tested against numbers chosen to sit either side of
them, because a threshold nobody checks is a magic number.
"""

from __future__ import annotations

import json

from app.agents.critic import measured_findings, review_run
from app.agents.summaries import summarise_run
from app.core.llm.base import ModelTier, RateLimitError, TransientLLMError
from app.core.llm.fake import FakeLLM
from app.ml.contracts import (
    ClusteringReport,
    EvaluationReport,
    ExplainabilityReport,
    FeatureImportance,
    FeatureStrategy,
    Leaderboard,
    LeaderboardEntry,
    MetricSummary,
    StrategyOverride,
)

# One scripted failure per tier. Derived from the enum rather than written as a
# literal so that adding a tier does not quietly turn these degradation tests
# into success tests: with too few failures scripted, FakeLLM falls through to
# its default reply and the agent succeeds, which is what happened when
# ModelTier.FALLBACK was added.
TIERS = len(ModelTier)


def leaderboard_of(*scores: tuple[str, float, float], errors: list[str] | None = None):
    entries = [
        LeaderboardEntry(rank=i + 1, model_name=name, primary_metric="f1_macro", score=s, std=sd)
        for i, (name, s, sd) in enumerate(scores)
    ]
    for name in errors or []:
        entries.append(
            LeaderboardEntry(
                rank=len(entries) + 1,
                model_name=name,
                primary_metric="f1_macro",
                score=float("nan"),
                std=0.0,
                error="could not be trained",
            )
        )
    return Leaderboard(
        task_type="classification",
        target_column="churn",
        primary_metric="f1_macro",
        n_folds=5,
        cv_strategy="StratifiedKFold",
        entries=entries,
    )


def evaluation_of(*, mean=0.81, std=0.01, n_rows=500, warnings=None):
    return EvaluationReport(
        task_type="classification",
        target_column="churn",
        model_name="RandomForest",
        n_folds=5,
        cv_strategy="StratifiedKFold",
        n_rows=n_rows,
        n_features=8,
        primary_metric="f1_macro",
        metrics={"f1_macro": MetricSummary(mean=mean, std=std)},
        warnings=warnings or [],
    )


def summary_of(**kw):
    return summarise_run(budget_tokens=6000, **kw)


def answer_json(**overrides) -> str:
    """A well-formed model reply, so tests vary one thing at a time."""
    payload = {
        "verdict": "A reasonable run with one thing to check.",
        "confidence": "high",
        "strengths": ["Cross-validation was leakage-safe."],
        "findings": [
            {
                "area": "modelling",
                "severity": "note",
                "finding": "A linear baseline would be easier to explain.",
                "recommendation": "Try logistic regression if interpretability matters.",
            }
        ],
        "recommended_next_steps": ["Validate on a later time period."],
    }
    payload.update(overrides)
    return json.dumps(payload)


class TestMeasuredFindings:
    """What code establishes before anyone is asked -- the floor of the critique."""

    def test_a_lead_inside_the_fold_spread_is_flagged(self):
        board = leaderboard_of(("XGBoost", 0.812, 0.04), ("LogisticRegression", 0.805, 0.03))
        findings = measured_findings(leaderboard=board)
        assert any("not established by this evidence" in f.finding for f in findings)

    def test_a_lead_outside_the_fold_spread_is_not_flagged(self):
        """The other side of the threshold: a real lead must not be second-guessed."""
        board = leaderboard_of(("XGBoost", 0.90, 0.01), ("LogisticRegression", 0.70, 0.01))
        findings = measured_findings(leaderboard=board)
        assert not [f for f in findings if "not established" in f.finding]

    def test_a_failed_candidate_is_reported(self):
        board = leaderboard_of(("XGBoost", 0.9, 0.01), errors=["LightGBM"])
        findings = measured_findings(leaderboard=board)
        assert any("could not be trained" in f.finding for f in findings)

    def test_a_wide_spread_relative_to_the_score_is_flagged(self):
        findings = measured_findings(evaluation=evaluation_of(mean=0.8, std=0.2))
        assert any("varies substantially across folds" in f.finding for f in findings)

    def test_a_tight_spread_is_not_flagged(self):
        findings = measured_findings(evaluation=evaluation_of(mean=0.8, std=0.01))
        assert not [f for f in findings if "varies substantially" in f.finding]

    def test_a_small_dataset_is_flagged(self):
        findings = measured_findings(evaluation=evaluation_of(n_rows=120))
        assert any("each held-out fold is small" in f.finding for f in findings)

    def test_weak_clusters_are_flagged_as_not_actionable(self):
        weak = ClusteringReport(method="kmeans", k=3, silhouette=0.3)
        findings = measured_findings(clustering=weak)
        assert any("not clearly structure" in f.finding for f in findings)

    def test_well_separated_clusters_are_not_flagged(self):
        clear = ClusteringReport(method="kmeans", k=3, silhouette=0.8)
        assert not measured_findings(clustering=clear)

    def test_broken_shap_additivity_is_a_blocker(self):
        """The one finding that invalidates another artifact wholesale."""
        report = ExplainabilityReport(
            model_name="XGBoost",
            task_type="classification",
            target_column="churn",
            additivity_max_error=0.05,
        )
        findings = measured_findings(explainability=report)
        assert [f for f in findings if f.severity == "blocker"]

    def test_intact_shap_additivity_raises_nothing(self):
        report = ExplainabilityReport(
            model_name="XGBoost",
            task_type="classification",
            target_column="churn",
            additivity_max_error=1e-14,
        )
        assert not measured_findings(explainability=report)

    def test_one_dominant_feature_is_flagged_as_possible_leakage(self):
        """The commonest way a too-good score happens, and code can spot it."""
        report = ExplainabilityReport(
            model_name="XGBoost",
            task_type="classification",
            target_column="churn",
            global_importance=[FeatureImportance(feature="account_id", importance=1.0, share=0.72)],
        )
        findings = measured_findings(explainability=report)
        assert any("target leakage" in f.recommendation for f in findings)

    def test_overruled_feature_decisions_are_noted(self):
        strategy = FeatureStrategy(
            overrides=[
                StrategyOverride(
                    column="size",
                    field="encode",
                    requested="ordinal",
                    applied="onehot",
                    reason="no ordering supplied",
                )
            ]
        )
        assert any("overruled by code" in f.finding for f in measured_findings(features=strategy))

    def test_every_measured_finding_is_marked_as_measured(self):
        findings = measured_findings(
            leaderboard=leaderboard_of(("A", 0.81, 0.04), ("B", 0.805, 0.03)),
            evaluation=evaluation_of(n_rows=100),
        )
        assert findings
        assert all(f.measured for f in findings)

    def test_every_finding_carries_an_action(self):
        """A finding with no recommendation is a complaint."""
        findings = measured_findings(
            leaderboard=leaderboard_of(("A", 0.81, 0.04), ("B", 0.805, 0.03)),
            evaluation=evaluation_of(n_rows=100, std=0.3),
            clustering=ClusteringReport(method="kmeans", k=3, silhouette=0.2),
        )
        assert all(f.recommendation for f in findings)


class TestWithAModel:
    def test_the_written_review_is_used(self):
        report = review_run(summary_of(), client=FakeLLM([answer_json()]))
        assert report.source == "llm"
        assert report.verdict.startswith("A reasonable run")
        assert report.strengths
        assert report.recommended_next_steps

    def test_measured_findings_survive_alongside_the_written_ones(self):
        """The model contributes to the review; it does not get to edit it."""
        board = leaderboard_of(("A", 0.81, 0.04), ("B", 0.805, 0.03))
        report = review_run(
            summary_of(leaderboard=board), leaderboard=board, client=FakeLLM([answer_json()])
        )
        assert any(f.measured for f in report.findings)
        assert any(not f.measured for f in report.findings)

    def test_a_model_cannot_pass_its_own_findings_off_as_measured(self):
        """Otherwise an opinion acquires the authority of a threshold check."""
        forged = answer_json(
            findings=[
                {
                    "area": "modelling",
                    "severity": "blocker",
                    "finding": "Invented, but claiming to be measured.",
                    "recommendation": "Ignore me.",
                    "measured": True,
                }
            ]
        )
        report = review_run(summary_of(), client=FakeLLM([forged]))
        assert not [f for f in report.findings if "Invented" in f.finding]

    def test_a_model_cannot_declare_its_own_source(self):
        report = review_run(summary_of(), client=FakeLLM([answer_json(source="default")]))
        assert report.source == "llm"

    def test_the_measured_verdict_is_used_when_the_model_omits_one(self):
        board = leaderboard_of(("A", 0.81, 0.04), ("B", 0.805, 0.03))
        report = review_run(
            summary_of(leaderboard=board),
            leaderboard=board,
            client=FakeLLM([answer_json(verdict="")]),
        )
        assert report.verdict


class TestDegradingWithoutAModel:
    def test_no_client_still_produces_a_review(self):
        board = leaderboard_of(("A", 0.81, 0.04), ("B", 0.805, 0.03))
        report = review_run(summary_of(leaderboard=board), leaderboard=board, client=None)

        assert report.source == "default"
        assert report.findings
        assert report.confidence == "low"

    def test_the_fallback_does_not_pass_itself_off_as_a_review(self):
        """Dressing a checklist up as considered judgement would misrepresent it."""
        report = review_run(summary_of(), client=None)
        assert "checklist result" in report.verdict
        assert "no language model was available" in report.verdict

    def test_a_provider_failure_keeps_the_measured_findings(self):
        """A malformed reply or an outage is not something a smaller model fixes."""
        board = leaderboard_of(("A", 0.81, 0.04), ("B", 0.805, 0.03))
        report = review_run(
            summary_of(leaderboard=board),
            leaderboard=board,
            client=FakeLLM([TransientLLMError("503 after retries")] * TIERS),
        )
        assert report.source == "default"
        assert report.findings

    def test_a_rate_limit_steps_down_a_tier_rather_than_giving_up(self):
        """A review that is only threshold checks is the thing being avoided."""
        board = leaderboard_of(("A", 0.81, 0.04), ("B", 0.805, 0.03))
        client = FakeLLM([RateLimitError("429"), answer_json()])
        report = review_run(summary_of(leaderboard=board), leaderboard=board, client=client)
        assert report.source == "llm"

    def test_a_rate_limit_on_every_tier_still_degrades(self):
        board = leaderboard_of(("A", 0.81, 0.04), ("B", 0.805, 0.03))
        client = FakeLLM([RateLimitError("429")] * TIERS)
        report = review_run(summary_of(leaderboard=board), leaderboard=board, client=client)
        assert report.source == "default"
        assert report.findings

    def test_a_provider_failure_says_the_written_review_is_missing(self):
        report = review_run(summary_of(), client=FakeLLM([TransientLLMError("503")] * TIERS))
        assert any("written review was not produced" in note for note in report.omissions)

    def test_persistently_malformed_output_falls_back(self):
        report = review_run(summary_of(), client=FakeLLM(["nonsense"] * 10, default="nonsense"))
        assert report.source == "default"


class TestTheReviewKnowsItsLimits:
    def test_omissions_from_the_summary_reach_the_report(self):
        """A critique of a capped summary must not read as a critique of the run."""
        from tests.test_summaries import wide_run

        summary = summarise_run(budget_tokens=1200, **wide_run())
        report = review_run(summary, client=None)

        assert report.omissions
        assert report.detail_level == summary.detail_level
        # However the omission is worded -- capped or dropped outright -- the true
        # column count has to appear, or the reviewer cannot size what it missed.
        assert any("121" in note for note in report.omissions)

    def test_a_complete_summary_records_no_omissions(self):
        report = review_run(summary_of(evaluation=evaluation_of()), client=None)
        assert report.omissions == []


class TestOrdering:
    def test_findings_can_be_read_worst_first(self):
        report = ExplainabilityReport(
            model_name="X",
            task_type="classification",
            target_column="churn",
            additivity_max_error=0.05,
        )
        review = review_run(
            summary_of(), explainability=report, evaluation=evaluation_of(n_rows=100), client=None
        )
        rank = {"blocker": 0, "concern": 1, "note": 2}
        severities = [f.severity for f in review.by_severity()]
        assert severities == sorted(severities, key=lambda s: rank[s])
        assert review.blockers()
