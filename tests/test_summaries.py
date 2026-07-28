"""Tests for prompt shrinking -- Section 9's actual difficulty (spec 6.3).

The measured problem these exist to solve: on the nine-column example the raw
artifacts come to ~4,300 tokens and fit the 15,000 input-TPM cap comfortably. On
a 121-column dataset the same artifacts come to **~24,000 tokens**, and the
Critic and Report run back to back inside the same minute. A demo dataset proves
nothing here, so the wide case is what most of this file is about.

The property that matters more than the size, though, is honesty. A capped list
that does not announce itself is indistinguishable from a complete one, and a
critic told "here are the feature decisions" will review them as if they were all
of them -- concluding that a 121-column dataset used 20 features and should have
used more. Every cap in this module has to leave a stated omission behind, and
that is what most of these assertions check.
"""

from __future__ import annotations

import json

import pytest

from app.agents.schema_models import ClassBalance
from app.agents.summaries import RunSummary, summarise_run
from app.ml.contracts import (
    CategorySummary,
    CleaningReport,
    ClusteringReport,
    ClusterProfile,
    ColumnStatistics,
    ColumnStrategy,
    CorrelationPair,
    DroppedColumn,
    EdaReport,
    EvaluationReport,
    ExplainabilityReport,
    FeatureImportance,
    FeatureStrategy,
    FinalModelInfo,
    Leaderboard,
    LeaderboardEntry,
    MetricSummary,
    PreprocessingSpec,
    StrategyOverride,
)


def eda_of(n_columns: int) -> EdaReport:
    return EdaReport(
        n_rows=500,
        n_columns=n_columns,
        target_column="outcome",
        columns=[
            ColumnStatistics(
                name=f"column_{i:03d}",
                semantic_type="categorical",
                count=500,
                missing=i % 3,
                missing_rate=(i % 3) / 500,
                categorical=CategorySummary(n_unique=7, top_values={"a": 300, "b": 200}),
            )
            for i in range(n_columns)
        ],
        top_correlations=[
            CorrelationPair(left=f"column_{i:03d}", right="column_000", correlation=0.9 - i / 100)
            for i in range(10)
        ],
        class_balance=ClassBalance(
            counts={"yes": 300, "no": 200}, imbalanced=False, imbalance_ratio=1.5
        ),
    )


def features_of(n_columns: int) -> FeatureStrategy:
    return FeatureStrategy(
        columns=[
            ColumnStrategy(column=f"column_{i:03d}", role="categorical", encode="onehot")
            for i in range(n_columns)
        ],
        rejected_columns=["invented_column"],
        defaulted_columns=["column_001"],
        overrides=[
            StrategyOverride(
                column="column_002",
                field="encode",
                requested="ordinal",
                applied="onehot",
                reason="no ordering was supplied",
            )
        ],
        source="llm",
    )


def evaluation_of() -> EvaluationReport:
    return EvaluationReport(
        task_type="classification",
        target_column="outcome",
        model_name="RandomForest",
        n_folds=5,
        cv_strategy="StratifiedKFold",
        n_rows=500,
        n_features=121,
        metrics={"f1_macro": MetricSummary(mean=0.8123456789, std=0.0141592653)},
        primary_metric="f1_macro",
    )


def explainability_of(n_features: int = 40, n_mapping: int = 300) -> ExplainabilityReport:
    return ExplainabilityReport(
        model_name="RandomForest",
        task_type="classification",
        target_column="outcome",
        explainer="TreeExplainer",
        n_rows_explained=500,
        n_encoded_features=n_mapping,
        global_importance=[
            FeatureImportance(feature=f"column_{i:03d}", importance=1.0 / (i + 1), share=0.1)
            for i in range(n_features)
        ],
        feature_name_mapping={f"encoded_{i}": f"column_{i:03d}" for i in range(n_mapping)},
        additivity_max_error=1.23e-14,
    )


def wide_run(n_columns: int = 121) -> dict:
    """Every artifact a real run produces, at a width that breaks the naive path."""
    return {
        "filename": "wide.csv",
        "cleaning": CleaningReport(
            n_rows_before=520,
            n_rows_after=500,
            n_columns_before=n_columns + 2,
            n_columns_after=n_columns,
            duplicate_rows_removed=20,
            dropped_columns=[DroppedColumn(name="id", reason="identifier")],
            missing_values_left_to_the_pipeline={f"column_{i:03d}": 3 for i in range(30)},
        ),
        "eda": eda_of(n_columns),
        "clustering": ClusteringReport(
            method="kmeans",
            k=4,
            silhouette=0.61,
            profiles=[ClusterProfile(cluster=i, size=125, share=0.25) for i in range(4)],
        ),
        "features": features_of(n_columns),
        "preprocessing": PreprocessingSpec(
            numeric_columns=[f"column_{i:03d}" for i in range(70)],
            categorical_columns=[f"column_{i:03d}" for i in range(70, n_columns)],
            strategy_source="llm",
        ),
        "leaderboard": Leaderboard(
            task_type="classification",
            target_column="outcome",
            primary_metric="f1_macro",
            n_folds=5,
            cv_strategy="StratifiedKFold",
            entries=[
                LeaderboardEntry(
                    rank=i + 1,
                    model_name=f"Model{i}",
                    primary_metric="f1_macro",
                    score=0.8 - i / 100,
                    std=0.01,
                )
                for i in range(4)
            ],
        ),
        "evaluation": evaluation_of(),
        "explainability": explainability_of(),
        "final_model": FinalModelInfo(
            model_name="RandomForest",
            task_type="classification",
            target_column="outcome",
            n_rows=500,
            n_features=n_columns,
            primary_metric="f1_macro",
            cv_score=0.8123456789,
        ),
    }


class TestItFitsTheBudget:
    def test_a_wide_run_fits_where_the_raw_artifacts_would_not(self):
        """The measured failure this module exists for."""
        summary = summarise_run(budget_tokens=6000, **wide_run())
        assert summary.estimated_tokens <= 6000

    def test_a_tighter_budget_gives_up_more_detail(self):
        generous = summarise_run(budget_tokens=6000, **wide_run())
        tight = summarise_run(budget_tokens=1200, **wide_run())

        assert tight.estimated_tokens < generous.estimated_tokens
        assert tight.detail_level != generous.detail_level

    def test_detail_never_increases_as_the_budget_shrinks(self):
        """Monotonicity, asserted over a sweep rather than at three magic budgets.

        Which budget crosses which tier depends on the dataset, so pinning
        specific numbers would make this a test of the fixture. The property that
        must hold for any dataset is that a smaller budget never buys *more*
        detail -- and that all three tiers are reachable.
        """
        order = {"full": 0, "reduced": 1, "minimal": 2}
        levels = [
            order[summarise_run(budget_tokens=budget, **wide_run()).detail_level]
            for budget in range(4000, 100, -100)
        ]
        assert levels == sorted(levels), "detail improved as the budget shrank"
        assert set(levels) == {0, 1, 2}, "not every tier was reachable"

    def test_a_narrow_run_keeps_full_detail(self):
        """Nothing is given up on a dataset that never threatened the cap."""
        summary = summarise_run(budget_tokens=6000, **wide_run(n_columns=6))
        assert summary.detail_level == "full"
        assert not [n for n in summary.notes if "column statistics" in n]


class TestOmissionsAreStated:
    """The property that makes a shrunk prompt safe rather than merely small."""

    def test_a_capped_column_table_says_how_many_it_dropped(self):
        summary = summarise_run(budget_tokens=6000, **wide_run())
        assert any("of 121 column statistics" in note for note in summary.notes)

    def test_a_capped_feature_table_says_how_many_it_dropped(self):
        summary = summarise_run(budget_tokens=6000, **wide_run())
        assert any("of 121 per-column feature decisions" in note for note in summary.notes)

    def test_dropping_a_table_entirely_is_stated_too(self):
        """At minimum detail the per-column tables go; silence would mislead most."""
        summary = summarise_run(budget_tokens=200, **wide_run())
        assert any("omitted" in note for note in summary.notes)

    def test_the_shap_name_mapping_is_declared_rather_than_sent(self):
        """300 rows of encoded-to-source mapping is the widest table in a run."""
        summary = summarise_run(budget_tokens=6000, **wide_run())
        assert any("300 entries" in note and "omitted" in note for note in summary.notes)
        assert "encoded_7" not in json.dumps(summary.payload)

    def test_the_notes_reach_the_prompt_not_just_the_object(self):
        """A note the model never sees protects nobody."""
        summary = summarise_run(budget_tokens=6000, **wide_run())
        block = summary.as_prompt_block()
        assert "known limits of what you can see" in block
        for note in summary.notes:
            assert note in block

    def test_an_unshrinkable_run_admits_it(self):
        """Better an oversized prompt with a warning than a confident lie."""
        summary = summarise_run(budget_tokens=1, **wide_run())
        assert any("larger than the prompt budget" in note for note in summary.notes)

    def test_the_true_column_count_survives_every_tier(self):
        """A capped table is safe only if the real total is still visible."""
        for budget in (6000, 1200, 200):
            summary = summarise_run(budget_tokens=budget, **wide_run())
            assert summary.payload["data"]["columns"] == 121
            assert summary.payload["feature_decisions"]["columns_decided"] == 121


class TestWhatSurvivesShrinking:
    """Detail is given up before substance -- these are the load-bearing facts."""

    @pytest.mark.parametrize("budget", [6000, 1200, 200])
    def test_the_scores_are_never_dropped(self, budget):
        summary = summarise_run(budget_tokens=budget, **wide_run())
        assert summary.payload["evaluation"]["metrics"]["f1_macro"]["mean"] == pytest.approx(0.8123)

    @pytest.mark.parametrize("budget", [6000, 1200, 200])
    def test_the_leakage_guardrail_is_never_dropped(self, budget):
        """A critic must never be able to conclude cluster labels were features."""
        summary = summarise_run(budget_tokens=budget, **wide_run())
        assert summary.payload["clusters"]["used_as_features"] is False

    @pytest.mark.parametrize("budget", [6000, 1200, 200])
    def test_the_served_models_caveat_is_never_dropped(self, budget):
        """The thing a reviewer is likeliest to get wrong about this project."""
        summary = summarise_run(budget_tokens=budget, **wide_run())
        assert "cross-validated estimate" in summary.payload["served_model"]["note"]

    @pytest.mark.parametrize("budget", [6000, 1200, 200])
    def test_the_llm_was_checked_claim_is_never_dropped(self, budget):
        summary = summarise_run(budget_tokens=budget, **wide_run())
        decisions = summary.payload["feature_decisions"]
        assert decisions["invented_columns_rejected"] == 1
        assert decisions["decisions_overruled_by_code"]

    @pytest.mark.parametrize("budget", [6000, 1200, 200])
    def test_the_additivity_check_is_never_dropped(self, budget):
        """It is the number saying whether the other SHAP numbers mean anything."""
        summary = summarise_run(budget_tokens=budget, **wide_run())
        assert "additivity_error" in summary.payload["explanation"]


class TestShrinkingPrimitives:
    def test_floats_are_rounded_rather_than_sent_at_full_precision(self):
        summary = summarise_run(budget_tokens=6000, **wide_run())
        assert summary.payload["served_model"]["cv_score"] == 0.8123
        assert "0.8123456789" not in json.dumps(summary.payload)

    def test_a_meaningful_zero_is_kept(self):
        """ "No duplicates were removed" is a finding; an absent key is a guess."""
        run = wide_run()
        run["cleaning"].duplicate_rows_removed = 0
        summary = summarise_run(budget_tokens=6000, **run)
        assert summary.payload["cleaning"]["duplicate_rows_removed"] == 0

    def test_a_meaningful_false_is_kept(self):
        summary = summarise_run(budget_tokens=6000, **wide_run())
        assert summary.payload["data"]["class_balance"]["imbalanced"] is False

    def test_empty_values_are_not_paid_for(self):
        run = wide_run()
        run["evaluation"].warnings = []
        summary = summarise_run(budget_tokens=6000, **run)
        assert "warnings" not in summary.payload["evaluation"]

    def test_a_run_missing_artifacts_still_summarises(self):
        """A job whose EDA failed still deserves a critic."""
        summary = summarise_run(budget_tokens=6000, evaluation=evaluation_of())
        assert "evaluation" in summary.payload
        assert "data" not in summary.payload

    def test_the_payload_is_json_serialisable(self):
        summary = summarise_run(budget_tokens=6000, **wide_run())
        assert json.loads(json.dumps(summary.payload, default=str))

    def test_the_estimate_matches_what_is_actually_sent(self):
        """An estimate over the payload alone would miss the notes it carries."""
        summary: RunSummary = summarise_run(budget_tokens=6000, **wide_run())
        block = summary.as_prompt_block()
        assert summary.estimated_tokens == pytest.approx(len(block) // 4, rel=0.01)
