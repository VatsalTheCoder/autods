"""Tests for the Markdown report.

The report is what a human actually reads, so these check the things a reader
depends on: that the headline number is there, that the methodology is stated in
plain English, that the per-fold row counts appear (they are the evidence the
split was real), and that the limitations section stops a weak first slice from
reading as a finished product.
"""

from __future__ import annotations

import pytest

from app.agents.schema_models import ClassBalance
from app.ml.contracts import (
    CleaningReport,
    ClusteringReport,
    ClusterProfile,
    ColumnStatistics,
    CorrelationPair,
    DroppedColumn,
    DtypeCorrection,
    EdaReport,
    EvaluationReport,
    ExplainabilityReport,
    FeatureImportance,
    FinalModelInfo,
    FoldScore,
    Leaderboard,
    LeaderboardEntry,
    LocalContribution,
    LocalExplanation,
    MetricSummary,
    NumericSummary,
    PlannerPlan,
    PreprocessingSpec,
)
from app.ml.report import build_markdown_report


@pytest.fixture
def cleaning() -> CleaningReport:
    return CleaningReport(
        n_rows_before=120,
        n_rows_after=100,
        n_columns_before=8,
        n_columns_after=5,
        duplicate_rows_removed=15,
        missing_target_rows_removed=5,
        dropped_columns=[DroppedColumn(name="email", reason="excluded at the schema checkpoint")],
        dtype_corrections=[DtypeCorrection(name="amount", from_dtype="object", to_dtype="float64")],
        missing_values_left_to_the_pipeline={"age": 12},
    )


@pytest.fixture
def preprocessing() -> PreprocessingSpec:
    return PreprocessingSpec(
        numeric_columns=["age", "income"],
        categorical_columns=["city"],
        unhandled_columns=[DroppedColumn(name="signed_up", reason="datetime -- Section 7")],
        numeric_strategy="median imputation, then standard scaling",
        categorical_strategy="most-frequent imputation, then one-hot encoding",
    )


@pytest.fixture
def evaluation() -> EvaluationReport:
    return EvaluationReport(
        task_type="classification",
        target_column="churn",
        model_name="RandomForestClassifier",
        n_folds=5,
        cv_strategy="StratifiedKFold",
        n_rows=100,
        n_features=3,
        folds=[
            FoldScore(fold=i, n_train=80, n_test=20, metrics={"f1_macro": 0.70 + i / 100})
            for i in range(1, 6)
        ],
        metrics={
            "f1_macro": MetricSummary(mean=0.73, std=0.014),
            "accuracy": MetricSummary(mean=0.81, std=0.02),
            "roc_auc": MetricSummary(mean=0.88, std=0.01),
        },
        primary_metric="f1_macro",
    )


@pytest.fixture
def markdown(cleaning, preprocessing, evaluation) -> str:
    return build_markdown_report(
        filename="customers.csv",
        plan=PlannerPlan(source="llm", rationale="Duplicates look accidental."),
        cleaning=cleaning,
        preprocessing=preprocessing,
        evaluation=evaluation,
    )


class TestHeadline:
    def test_the_dataset_and_target_are_named(self, markdown):
        assert "customers.csv" in markdown
        assert "churn" in markdown

    def test_the_primary_score_is_shown_with_its_spread(self, markdown):
        assert "0.7300" in markdown
        assert "0.0140" in markdown

    def test_the_model_is_named(self, markdown):
        assert "RandomForestClassifier" in markdown


class TestMethodologyIsStated:
    """The claim an examiner probes should be in the document, not just the code."""

    def test_the_fold_count_and_strategy_are_stated(self, markdown):
        assert "5 folds" in markdown
        assert "StratifiedKFold" in markdown

    def test_it_says_fitting_happened_on_training_rows_only(self, markdown):
        assert "training rows only" in markdown

    def test_it_says_nothing_was_prepared_up_front(self, markdown):
        assert "whole dataset beforehand" in markdown

    def test_it_explains_why_that_matters(self, markdown):
        assert "inflates" in markdown


class TestTables:
    def test_every_metric_appears(self, markdown):
        assert "Accuracy" in markdown
        assert "ROC-AUC" in markdown
        assert "F1 (macro)" in markdown

    def test_each_fold_shows_the_rows_it_was_fitted_and_scored_on(self, markdown):
        """The numbers that let a reader verify the split was genuine."""
        assert "Rows fitted on" in markdown
        assert "Rows scored on" in markdown
        assert markdown.count("| 80 | 20 |") == 5


class TestDataQuality:
    def test_row_and_column_changes_are_reported(self, markdown):
        assert "120" in markdown and "100" in markdown

    def test_removals_are_itemised(self, markdown):
        assert "15" in markdown  # duplicates
        assert "email" in markdown

    def test_dtype_corrections_are_listed(self, markdown):
        assert "amount" in markdown
        assert "float64" in markdown

    def test_remaining_gaps_are_explained_as_deliberate(self, markdown):
        """The report must say the gaps were left on purpose, not overlooked."""
        assert "left in place" in markdown
        assert "inside each cross-validation fold" in markdown

    def test_no_gaps_says_so_plainly(self, cleaning, preprocessing, evaluation):
        cleaning.missing_values_left_to_the_pipeline = {}
        report = build_markdown_report(
            filename="x.csv",
            plan=PlannerPlan(),
            cleaning=cleaning,
            preprocessing=preprocessing,
            evaluation=evaluation,
        )
        assert "No missing values remained" in report


class TestPreparation:
    def test_strategies_are_described(self, markdown):
        assert "median imputation" in markdown
        assert "one-hot encoding" in markdown

    def test_unhandled_columns_are_disclosed(self, markdown):
        assert "signed_up" in markdown

    def test_an_llm_plan_is_attributed_to_the_model(self, markdown):
        assert "chosen by the planning model" in markdown

    def test_a_default_plan_is_not_attributed_to_the_model(
        self, cleaning, preprocessing, evaluation
    ):
        """An artifact must never imply a model made a decision defaults made."""
        report = build_markdown_report(
            filename="x.csv",
            plan=PlannerPlan(source="default"),
            cleaning=cleaning,
            preprocessing=preprocessing,
            evaluation=evaluation,
        )
        assert "built-in defaults" in report
        assert "chosen by the planning model" not in report


class TestHonesty:
    def test_limitations_are_stated(self, markdown):
        """A bare run: no charts, one model, no explanation -- say all three."""
        assert "does not tell you" in markdown
        assert "no exploratory charts" in markdown.lower()
        assert "only one model was trained" in markdown.lower()
        assert "library defaults" in markdown

    def test_limitations_do_not_contradict_the_report(self, cleaning, preprocessing, evaluation):
        """The bug this replaced: a report with a leaderboard and six charts still
        closed by calling model comparison and plots "still to come"."""
        board = Leaderboard(
            task_type="classification",
            target_column="churn",
            primary_metric="f1_macro",
            n_folds=5,
            cv_strategy="StratifiedKFold",
            entries=[
                LeaderboardEntry(
                    rank=1,
                    model_name="LightGBM",
                    primary_metric="f1_macro",
                    score=0.812,
                    std=0.021,
                    fit_seconds=1.4,
                ),
                LeaderboardEntry(
                    rank=2,
                    model_name="RandomForest",
                    primary_metric="f1_macro",
                    score=0.799,
                    std=0.030,
                    fit_seconds=8.2,
                ),
            ],
        )
        eda = EdaReport(
            n_rows=100,
            n_columns=4,
            target_column="churn",
            plots=["target_counts.png", "missing_values.png"],
        )
        report = build_markdown_report(
            filename="customers.csv",
            plan=PlannerPlan(),
            cleaning=cleaning,
            preprocessing=preprocessing,
            evaluation=evaluation,
            leaderboard=board,
            eda=eda,
        )
        assert "only one model was trained" not in report.lower()
        assert "no exploratory charts" not in report.lower()
        # The two that are true of every run stay put.
        assert "library defaults" in report
        assert "single pass of cross-validation" in report

    def test_caveats_appear_when_there_are_any(self, cleaning, preprocessing, evaluation):
        evaluation.warnings = ["Reduced to 3 folds: the rarest class has only 3 rows."]
        report = build_markdown_report(
            filename="x.csv",
            plan=PlannerPlan(),
            cleaning=cleaning,
            preprocessing=preprocessing,
            evaluation=evaluation,
        )
        assert "Caveats" in report
        assert "rarest class" in report

    def test_no_caveats_section_when_there_is_nothing_to_caveat(self, markdown):
        assert "## Caveats" not in markdown

    def test_a_wide_spread_is_flagged_as_less_trustworthy(
        self, cleaning, preprocessing, evaluation
    ):
        evaluation.metrics["f1_macro"] = MetricSummary(mean=0.73, std=0.2)
        report = build_markdown_report(
            filename="x.csv",
            plan=PlannerPlan(),
            cleaning=cleaning,
            preprocessing=preprocessing,
            evaluation=evaluation,
        )
        assert "rough estimate" in report

    def test_a_tight_spread_is_described_as_stable(self, markdown):
        assert "barely moved" in markdown


class TestRegressionReport:
    def test_regression_metrics_are_labelled_and_formatted(self, cleaning, preprocessing):
        evaluation = EvaluationReport(
            task_type="regression",
            target_column="price",
            model_name="RandomForestRegressor",
            n_folds=5,
            cv_strategy="KFold",
            n_rows=100,
            n_features=3,
            folds=[FoldScore(fold=1, n_train=80, n_test=20, metrics={"r2": 0.62})],
            metrics={
                "r2": MetricSummary(mean=0.62, std=0.03),
                "rmse": MetricSummary(mean=15234.5, std=900.25),
            },
            primary_metric="r2",
        )
        report = build_markdown_report(
            filename="houses.csv",
            plan=PlannerPlan(),
            cleaning=cleaning,
            preprocessing=preprocessing,
            evaluation=evaluation,
        )
        assert "R²" in report
        assert "RMSE" in report
        # Large error values get thousands separators, not a long decimal tail.
        assert "15,234.50" in report


class TestEdaAndClusteringSections:
    """Section 6's findings, and the guardrail restated where a reader sees it."""

    def build(self, cleaning, preprocessing, evaluation, *, eda=None, clustering=None):
        return build_markdown_report(
            filename="x.csv",
            plan=PlannerPlan(),
            cleaning=cleaning,
            preprocessing=preprocessing,
            evaluation=evaluation,
            eda=eda,
            clustering=clustering,
        )

    def test_the_report_still_builds_without_eda(self, cleaning, preprocessing, evaluation):
        """A failed descriptive stage must not cost the reader the model results."""
        report = self.build(cleaning, preprocessing, evaluation)
        assert "## Result" in report
        assert "## Natural groups" not in report

    def test_imbalance_is_explained_not_just_stated(self, cleaning, preprocessing, evaluation):
        eda = EdaReport(
            n_rows=100,
            n_columns=4,
            target_column="churn",
            class_balance=ClassBalance(
                counts={"no": 90, "yes": 10}, imbalance_ratio=9.0, imbalanced=True
            ),
        )
        report = self.build(cleaning, preprocessing, evaluation, eda=eda)
        assert "imbalanced" in report
        assert "macro F1" in report

    def test_correlations_are_listed(self, cleaning, preprocessing, evaluation):
        eda = EdaReport(
            n_rows=100,
            n_columns=4,
            target_column="churn",
            top_correlations=[CorrelationPair(left="age", right="income", correlation=0.82)],
        )
        report = self.build(cleaning, preprocessing, evaluation, eda=eda)
        assert "age" in report and "+0.82" in report

    def test_outliers_are_reported_as_kept(self, cleaning, preprocessing, evaluation):
        eda = EdaReport(
            n_rows=100,
            n_columns=1,
            target_column="churn",
            columns=[
                ColumnStatistics(
                    name="income",
                    semantic_type="numeric",
                    count=100,
                    missing=0,
                    missing_rate=0.0,
                    numeric=NumericSummary(
                        mean=1,
                        std=1,
                        minimum=0,
                        q1=0,
                        median=1,
                        q3=2,
                        maximum=9,
                        outlier_count=7,
                    ),
                )
            ],
        )
        report = self.build(cleaning, preprocessing, evaluation, eda=eda)
        assert "left in the data" in report

    def test_groups_are_described_with_their_sizes(self, cleaning, preprocessing, evaluation):
        clustering = ClusteringReport(
            method="kmeans",
            k=2,
            silhouette=0.62,
            profiles=[
                ClusterProfile(cluster=0, size=60, share=0.6, description="Younger customers."),
                ClusterProfile(cluster=1, size=40, share=0.4),
            ],
        )
        report = self.build(cleaning, preprocessing, evaluation, clustering=clustering)
        assert "Group 0" in report
        assert "Younger customers." in report
        assert "60" in report

    def test_the_guardrail_is_stated_in_the_report(self, cleaning, preprocessing, evaluation):
        """A reader should learn the labels were not fed to the model."""
        clustering = ClusteringReport(
            method="kmeans",
            k=2,
            silhouette=0.62,
            profiles=[ClusterProfile(cluster=0, size=100, share=1.0)],
        )
        report = self.build(cleaning, preprocessing, evaluation, clustering=clustering)
        assert "not** given to the model" in report
        assert "inflate" in report

    def test_a_skipped_clustering_run_adds_no_section(self, cleaning, preprocessing, evaluation):
        clustering = ClusteringReport(method="kmeans", k=0, silhouette=0.0)
        report = self.build(cleaning, preprocessing, evaluation, clustering=clustering)
        assert "## Natural groups" not in report


def test_the_report_is_valid_markdown_structure(markdown):
    """One H1, several H2s, and no stray empty sections."""
    assert markdown.startswith("# ")
    assert markdown.count("\n# ") == 0  # exactly one top-level heading
    assert markdown.count("## ") >= 5
    assert "\n\n\n" not in markdown


class TestTheLeaderboard:
    """Section 7: the report should show what was compared, not just what won."""

    @pytest.fixture
    def leaderboard(self) -> Leaderboard:
        return Leaderboard(
            task_type="classification",
            target_column="churn",
            primary_metric="f1_macro",
            n_folds=5,
            cv_strategy="StratifiedKFold",
            entries=[
                LeaderboardEntry(
                    rank=1,
                    model_name="LightGBM",
                    primary_metric="f1_macro",
                    score=0.812,
                    std=0.021,
                    fit_seconds=1.4,
                ),
                LeaderboardEntry(
                    rank=2,
                    model_name="RandomForest",
                    primary_metric="f1_macro",
                    score=0.790,
                    std=0.090,
                    fit_seconds=2.2,
                ),
                LeaderboardEntry(
                    rank=3,
                    model_name="XGBoost",
                    primary_metric="f1_macro",
                    score=float("nan"),
                    std=0.0,
                    error="it exploded",
                ),
            ],
            resampling="SMOTE, applied inside each training fold only",
        )

    @pytest.fixture
    def with_board(self, cleaning, preprocessing, evaluation, leaderboard) -> str:
        return build_markdown_report(
            filename="customers.csv",
            plan=PlannerPlan(source="llm", use_smote=True),
            cleaning=cleaning,
            preprocessing=preprocessing,
            evaluation=evaluation,
            leaderboard=leaderboard,
        )

    def test_every_candidate_is_listed(self, with_board):
        for name in ("LightGBM", "RandomForest", "XGBoost"):
            assert name in with_board

    def test_the_spread_is_beside_the_score_not_in_a_footnote(self, with_board):
        """0.79 ± 0.09 losing to 0.81 ± 0.02 is a judgement the reader must make."""
        assert "± 0.021" in with_board
        assert "± 0.090" in with_board

    def test_a_failed_candidate_says_so_rather_than_showing_a_score(self, with_board):
        assert "could not be trained" in with_board

    def test_the_fair_comparison_is_stated(self, with_board):
        assert "same 5 folds" in with_board

    def test_resampling_is_recorded_next_to_the_scores(self, with_board):
        assert "SMOTE" in with_board

    def test_a_run_without_a_leaderboard_still_builds(self, markdown):
        """Older runs, and any run whose modelling produced only one result."""
        assert "## Models compared" not in markdown


class TestSkippedStepsAreReported:
    """A run that did less should say what it chose not to do (spec 11)."""

    def test_the_steps_that_were_turned_off_are_named(self, cleaning, preprocessing, evaluation):
        report = build_markdown_report(
            filename="customers.csv",
            plan=PlannerPlan(source="llm", use_smote=False, run_feature_selection=False),
            cleaning=cleaning,
            preprocessing=preprocessing,
            evaluation=evaluation,
        )
        assert "Steps not used on this dataset" in report
        assert "oversampling the rare outcome" in report

    def test_a_step_that_ran_is_not_listed_as_unused(self, cleaning, preprocessing, evaluation):
        report = build_markdown_report(
            filename="customers.csv",
            plan=PlannerPlan(source="llm", use_smote=True),
            cleaning=cleaning,
            preprocessing=preprocessing,
            evaluation=evaluation,
        )
        assert "oversampling the rare outcome" not in report

    def test_sampling_is_explained_where_it_happened(self, cleaning, preprocessing, evaluation):
        report = build_markdown_report(
            filename="customers.csv",
            plan=PlannerPlan(source="llm", run_sampling=True),
            cleaning=cleaning,
            preprocessing=preprocessing,
            evaluation=evaluation,
            sampling_note="Trained on a random sample of 20,000 rows drawn from 500,000.",
        )
        assert "random sample of 20,000 rows" in report

    def test_the_new_column_kinds_are_described(self, cleaning, evaluation):
        spec = PreprocessingSpec(
            numeric_columns=["age"],
            ordinal_columns=["size"],
            datetime_columns=["signed_up"],
            text_columns=["user_ref"],
            numeric_strategy="median imputation, then standard scaling",
            strategy_source="llm",
        )
        report = build_markdown_report(
            filename="customers.csv",
            plan=PlannerPlan(source="llm"),
            cleaning=cleaning,
            preprocessing=spec,
            evaluation=evaluation,
        )
        assert "rank order" in report
        assert "calendar features" in report
        assert "how often each value occurs" in report

    def test_an_llm_chosen_strategy_says_it_was_checked(self, cleaning, evaluation):
        spec = PreprocessingSpec(numeric_columns=["age"], strategy_source="llm")
        report = build_markdown_report(
            filename="customers.csv",
            plan=PlannerPlan(source="llm"),
            cleaning=cleaning,
            preprocessing=spec,
            evaluation=evaluation,
        )
        assert "checked against the real columns" in report


class TestBothPlanSourcesReadAsSentences:
    """The LLM branch of this line was unreachable until a live model was used."""

    def test_the_default_branch_reads_correctly(self, cleaning, preprocessing, evaluation):
        report = build_markdown_report(
            filename="c.csv",
            plan=PlannerPlan(source="default"),
            cleaning=cleaning,
            preprocessing=preprocessing,
            evaluation=evaluation,
        )
        assert "Preparation steps came from the built-in defaults." in report

    def test_the_llm_branch_reads_correctly(self, cleaning, preprocessing, evaluation):
        report = build_markdown_report(
            filename="c.csv",
            plan=PlannerPlan(source="llm"),
            cleaning=cleaning,
            preprocessing=preprocessing,
            evaluation=evaluation,
        )
        assert "Preparation steps were chosen by the planning model." in report
        assert "came from chosen by" not in report


class TestExplainabilityAndTheServedModel:
    """Section 8's two sections -- the explanation, and what is actually saved.

    The second one exists to carry a single sentence: the served model's score
    was not measured, it was inherited from cross-validation. A report that
    showed the same number without that sentence would be implying the model had
    been scored on data it had never seen, which is the opposite of true.
    """

    @pytest.fixture
    def explainability(self) -> ExplainabilityReport:
        return ExplainabilityReport(
            model_name="RandomForest",
            task_type="classification",
            target_column="churn",
            explainer="TreeExplainer",
            n_rows_explained=500,
            n_encoded_features=19,
            classes=["no", "yes"],
            aggregation="Mean absolute SHAP value per feature.",
            global_importance=[
                FeatureImportance(
                    feature="support_calls",
                    importance=0.31,
                    share=0.5,
                    encoded_features=["support_calls"],
                    direction="higher values push the prediction up",
                ),
                FeatureImportance(
                    feature="city",
                    importance=0.09,
                    share=0.15,
                    encoded_features=["city_London", "city_Leeds"],
                ),
            ],
            examples=[
                LocalExplanation(
                    row_label="42",
                    predicted="yes",
                    explained_class="yes",
                    probability=0.87,
                    base_value=0.5,
                    contributions=[
                        LocalContribution(feature="support_calls", value="7", contribution=0.3),
                        LocalContribution(feature="city", value="London", contribution=-0.05),
                    ],
                    other_contribution=0.02,
                    output_value=0.77,
                )
            ],
            feature_name_mapping={"city_London": "city"},
        )

    @pytest.fixture
    def final_model(self) -> FinalModelInfo:
        return FinalModelInfo(
            model_name="RandomForest",
            task_type="classification",
            target_column="churn",
            n_rows=100,
            n_features=5,
            primary_metric="f1_macro",
            cv_score=0.8123,
            artifact="final_model.pkl",
        )

    @pytest.fixture
    def report(self, cleaning, preprocessing, evaluation, explainability, final_model) -> str:
        return build_markdown_report(
            filename="customers.csv",
            plan=PlannerPlan(source="llm"),
            cleaning=cleaning,
            preprocessing=preprocessing,
            evaluation=evaluation,
            explainability=explainability,
            final_model=final_model,
        )

    def test_the_influences_are_listed_in_the_users_column_names(self, report):
        assert "## Why the model predicts what it does" in report
        assert "`support_calls`" in report
        assert "`city`" in report
        # The encoded names are the model's business, not the reader's.
        assert "city_London" not in report

    def test_a_column_with_no_defined_direction_is_left_blank_not_guessed(self, report):
        """A one-hot column has no "higher value" to have a direction about."""
        city_row = next(line for line in report.splitlines() if line.startswith("| `city`"))
        assert city_row.rstrip().endswith("| — |")

    def test_one_prediction_is_shown_adding_up(self, report):
        assert "### One prediction, in full" in report
        assert "Baseline: +0.5000" in report
        assert "Model output: +0.7700" in report

    def test_the_served_model_is_named_and_its_score_disclaimed(self, report):
        assert "## The model that gets served" in report
        assert "`final_model.pkl`" in report
        assert "no unseen data left to score itself against" in report

    def test_an_unexplainable_model_says_so_rather_than_leaving_a_gap(
        self, cleaning, preprocessing, evaluation
    ):
        report = build_markdown_report(
            filename="c.csv",
            plan=PlannerPlan(source="default"),
            cleaning=cleaning,
            preprocessing=preprocessing,
            evaluation=evaluation,
            explainability=ExplainabilityReport(
                model_name="GaussianNB",
                task_type="classification",
                target_column="churn",
                warnings=["No SHAP explainer is defined for GaussianNB."],
            ),
        )
        assert "No SHAP explainer is defined for GaussianNB." in report

    def test_the_sections_are_absent_when_the_run_never_got_there(self, markdown):
        assert "## Why the model predicts what it does" not in markdown
        assert "## The model that gets served" not in markdown


class TestRegressionErrorsAreReadNotJustListed:
    """The table has always held the numbers; nothing made the comparison.

    A run on listing prices reported RMSE 202 beside MAE 81 and R² 0.07, and a
    reader had to know that RMSE squares its errors to see that a minority of
    rows owned the score. That inference belongs in the report.
    """

    @staticmethod
    def _report(mae: float, rmse: float, median_ae: float | None = None, **kwargs) -> str:
        metrics = {
            "mae": MetricSummary(mean=mae, std=1.0),
            "rmse": MetricSummary(mean=rmse, std=1.0),
            "r2": MetricSummary(mean=0.07, std=0.21),
        }
        if median_ae is not None:
            metrics["median_ae"] = MetricSummary(mean=median_ae, std=1.0)
        return build_markdown_report(
            filename="listings.csv",
            plan=PlannerPlan(),
            cleaning=CleaningReport(
                n_rows_before=100, n_rows_after=100, n_columns_before=4, n_columns_after=4
            ),
            preprocessing=PreprocessingSpec(numeric_columns=["rooms"]),
            evaluation=EvaluationReport(
                task_type="regression",
                target_column="price",
                model_name="LGBMRegressor",
                n_folds=5,
                cv_strategy="KFold",
                n_rows=100,
                n_features=3,
                folds=[FoldScore(fold=1, n_train=80, n_test=20, metrics={"r2": 0.07})],
                metrics=metrics,
                primary_metric="r2",
            ),
            **kwargs,
        )

    def test_a_squared_error_dominated_run_is_called_out(self):
        report = self._report(mae=80.84, rmse=201.76)
        assert "RMSE is 2.5x MAE" in report
        assert "minority of rows" in report
        assert "R² inherits the same bias" in report

    def test_evenly_sized_errors_are_called_out_too(self):
        report = self._report(mae=80.0, rmse=88.0)
        assert "close to MAE" in report
        assert "no small group of rows" in report

    def test_the_median_error_is_contrasted_with_the_mean(self):
        report = self._report(mae=80.84, rmse=201.76, median_ae=30.0)
        assert "median error is" in report
        assert "half of all predictions" in report

    def test_a_classification_report_says_nothing_about_it(self, markdown):
        assert "RMSE" not in markdown

    def test_mape_is_shown_as_a_percentage(self):
        report = self._report(mae=80.84, rmse=201.76)
        assert "MAPE" not in report  # absent from the metrics dict above
        with_mape = build_markdown_report(
            filename="listings.csv",
            plan=PlannerPlan(),
            cleaning=CleaningReport(
                n_rows_before=100, n_rows_after=100, n_columns_before=4, n_columns_after=4
            ),
            preprocessing=PreprocessingSpec(numeric_columns=["rooms"]),
            evaluation=EvaluationReport(
                task_type="regression",
                target_column="price",
                model_name="LGBMRegressor",
                n_folds=5,
                cv_strategy="KFold",
                n_rows=100,
                n_features=3,
                metrics={"mape": MetricSummary(mean=0.4312, std=0.02)},
                primary_metric="mape",
            ),
        )
        assert "43.1%" in with_mape


class TestTheTargetsTailIsReported:
    @staticmethod
    def _with_outliers(**kwargs) -> str:
        from app.ml.contracts import TargetOutliers

        return build_markdown_report(
            filename="listings.csv",
            plan=PlannerPlan(),
            cleaning=CleaningReport(
                n_rows_before=100,
                n_rows_after=100,
                n_columns_before=4,
                n_columns_after=4,
                target_outliers=TargetOutliers(column="price", n_detected=250, note="TAIL NOTE"),
            ),
            preprocessing=PreprocessingSpec(numeric_columns=["rooms"]),
            evaluation=EvaluationReport(
                task_type="regression",
                target_column="price",
                model_name="LGBMRegressor",
                n_folds=5,
                cv_strategy="KFold",
                n_rows=100,
                n_features=3,
                metrics={"r2": MetricSummary(mean=0.07, std=0.21)},
                primary_metric="r2",
            ),
            **kwargs,
        )

    def test_the_tail_note_reaches_the_data_quality_section(self):
        assert "TAIL NOTE" in self._with_outliers()

    def test_an_infinite_target_row_is_accounted_for(self):
        report = build_markdown_report(
            filename="listings.csv",
            plan=PlannerPlan(),
            cleaning=CleaningReport(
                n_rows_before=102,
                n_rows_after=100,
                n_columns_before=4,
                n_columns_after=4,
                non_finite_target_rows_removed=2,
            ),
            preprocessing=PreprocessingSpec(numeric_columns=["rooms"]),
            evaluation=EvaluationReport(
                task_type="regression",
                target_column="price",
                model_name="LGBMRegressor",
                n_folds=5,
                cv_strategy="KFold",
                n_rows=100,
                n_features=3,
                metrics={"r2": MetricSummary(mean=0.07, std=0.21)},
                primary_metric="r2",
            ),
        )
        assert "2 rows whose target was infinite" in report


class TestTheBaselineRowIsMarked:
    def test_the_baseline_is_labelled_and_explained(self, cleaning, preprocessing, evaluation):
        report = build_markdown_report(
            filename="customers.csv",
            plan=PlannerPlan(),
            cleaning=cleaning,
            preprocessing=preprocessing,
            evaluation=evaluation,
            leaderboard=Leaderboard(
                task_type="classification",
                target_column="churn",
                primary_metric="f1_macro",
                n_folds=5,
                cv_strategy="StratifiedKFold",
                entries=[
                    LeaderboardEntry(
                        rank=1,
                        model_name="LightGBM",
                        primary_metric="f1_macro",
                        score=0.73,
                        std=0.01,
                    ),
                    LeaderboardEntry(
                        rank=2,
                        model_name="Baseline (most frequent class)",
                        primary_metric="f1_macro",
                        score=0.34,
                        std=0.00,
                        is_baseline=True,
                    ),
                ],
            ),
        )
        assert "Baseline (most frequent class) *(baseline)*" in report
        assert "never the model that gets served" in report
