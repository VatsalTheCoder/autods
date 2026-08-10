"""Tests for cross-validation mechanics.

The leakage guarantees live in ``test_leakage.py``. This file covers the rest of
what the CV loop has to get right: the splitter it chooses, how it degrades on
data too small or too skewed for five stratified folds, reproducibility, and that
imblearn's pipeline is the one being used so Section 7's SMOTE lands inside the
fold rather than outside it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from imblearn.pipeline import Pipeline as ImbPipeline

from app.ml.modeling import (
    ModelingError,
    build_pipeline,
    cross_validate_model,
    preprocessing_of,
    run_leaderboard,
)
from app.ml.preprocessing import build_preprocessor


def frame_of(n: int, *, classes: list[str] | None = None) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    labels = classes or (["yes", "no"] * ((n + 1) // 2))[:n]
    return pd.DataFrame(
        {
            "x1": rng.normal(0, 1, n),
            "city": rng.choice(["London", "Leeds"], n),
            "churn": labels,
        }
    )


def run(frame: pd.DataFrame, *, target="churn", task_type="classification", **kw):
    preprocessor = build_preprocessor(frame, target=target).transformer
    return cross_validate_model(
        frame, target=target, task_type=task_type, preprocessor=preprocessor, **kw
    )


class TestSplitterChoice:
    def test_classification_is_stratified(self):
        assert run(frame_of(100)).cv_strategy == "StratifiedKFold"

    def test_regression_uses_plain_kfold(self):
        rng = np.random.default_rng(0)
        frame = pd.DataFrame({"x1": rng.normal(0, 1, 60), "price": rng.normal(100, 10, 60)})
        assert run(frame, target="price", task_type="regression").cv_strategy == "KFold"

    def test_five_folds_by_default(self):
        result = run(frame_of(100))
        assert result.n_folds == 5
        assert len(result.folds) == 5


class TestSmallAndSkewedData:
    """Reduce the folds rather than abandon the run -- but say so in the report."""

    def test_folds_are_reduced_when_the_rarest_class_is_too_small(self):
        # 3 members of the rare class cannot support 5 stratified folds.
        labels = ["rare"] * 3 + ["common"] * 47
        result = run(frame_of(50, classes=labels))
        assert result.n_folds == 3
        assert any("rarest class" in w for w in result.warnings)

    def test_a_single_member_class_falls_back_to_unstratified_folds(self):
        labels = ["rare"] + ["common"] * 49
        result = run(frame_of(50, classes=labels))
        assert result.cv_strategy == "KFold"
        assert any("could not be stratified" in w for w in result.warnings)

    def test_a_reduction_is_always_warned_about(self):
        """A report must never imply five folds it did not run."""
        labels = ["rare"] * 2 + ["common"] * 48
        result = run(frame_of(50, classes=labels))
        assert result.n_folds < 5
        assert result.warnings

    def test_too_few_rows_is_a_clear_error(self):
        with pytest.raises(ModelingError, match="at least 4"):
            run(frame_of(3))


class TestReproducibility:
    def test_the_same_seed_gives_the_same_scores(self):
        frame = frame_of(100)
        first = run(frame, random_seed=7)
        second = run(frame, random_seed=7)
        assert [f.metrics for f in first.folds] == [f.metrics for f in second.folds]

    def test_a_different_seed_changes_the_folds(self):
        """Proof the seed is actually wired to the shuffling, not just stored."""
        frame = frame_of(100)
        assert [f.metrics for f in run(frame, random_seed=1).folds] != [
            f.metrics for f in run(frame, random_seed=2).folds
        ]


class TestPipelineConstruction:
    def test_the_pipeline_is_imblearns_so_smote_can_go_inside_the_fold(self):
        """Spec 8: sklearn's Pipeline would put a future resampler outside the fold."""
        frame = frame_of(20)
        preprocessor = build_preprocessor(frame, target="churn").transformer
        pipeline = build_pipeline(preprocessor, "classification", random_seed=0)
        assert isinstance(pipeline, ImbPipeline)

    def test_the_pipeline_holds_preprocessing_and_model_as_one_object(self):
        """One fit call per fold, on one object -- they cannot be fitted separately."""
        frame = frame_of(20)
        preprocessor = build_preprocessor(frame, target="churn").transformer
        pipeline = build_pipeline(preprocessor, "classification", random_seed=0)
        assert [name for name, _ in pipeline.steps] == ["preprocess", "model"]

    def test_the_estimator_matches_the_task(self):
        frame = frame_of(20)
        preprocessor = build_preprocessor(frame, target="churn").transformer
        classifier = build_pipeline(preprocessor, "classification", random_seed=0)
        regressor = build_pipeline(preprocessor, "regression", random_seed=0)
        assert type(classifier.named_steps["model"]).__name__ == "RandomForestClassifier"
        assert type(regressor.named_steps["model"]).__name__ == "RandomForestRegressor"


class TestFeatureSelectionFits:
    """The selection recipe has to survive being put in an imblearn pipeline.

    imblearn refuses a nested ``Pipeline`` as an intermediate step, and
    ``build_preprocessor`` returns one whenever the planner asks for feature
    selection. Nesting it produced an object that could not be fitted at all --
    and because ``run_leaderboard`` catches each candidate's failure separately,
    the visible symptom was every model in the roster failing at once with a
    message about pipelines. These tests fit the thing rather than inspecting it,
    which is the only shape in which that bug exists.
    """

    def test_a_selection_recipe_is_spliced_in_rather_than_nested(self):
        preprocessor = build_preprocessor(frame_of(20), target="churn", select_k=2).transformer
        pipeline = build_pipeline(preprocessor, "classification", random_seed=0)
        assert [name for name, _ in pipeline.steps] == ["columns", "select", "model"]

    def test_cross_validation_runs_with_feature_selection_on(self):
        frame = frame_of(40)
        preprocessor = build_preprocessor(frame, target="churn", select_k=2).transformer
        result = cross_validate_model(
            frame, target="churn", task_type="classification", preprocessor=preprocessor
        )
        assert len(result.folds) == result.n_folds

    def test_the_whole_roster_trains_with_feature_selection_on(self):
        frame = frame_of(40)
        preprocessor = build_preprocessor(frame, target="churn", select_k=2).transformer
        leaderboard, _ = run_leaderboard(
            frame,
            target="churn",
            task_type="classification",
            preprocessor=preprocessor,
            cv_folds=3,
        )
        assert not [entry.model_name for entry in leaderboard.entries if entry.error]

    def test_the_preprocessing_half_is_recoverable_from_a_fitted_pipeline(self):
        """Explainability needs the transform half, and it is not one named step."""
        frame = frame_of(40)
        preprocessor = build_preprocessor(frame, target="churn", select_k=2).transformer
        pipeline = build_pipeline(preprocessor, "classification", random_seed=0)
        pipeline.fit(frame.drop(columns=["churn"]), frame["churn"])

        preprocess = preprocessing_of(pipeline)
        assert preprocess.transform(frame.drop(columns=["churn"])).shape[1] == 2
        assert len(preprocess.get_feature_names_out()) == 2


class TestResultMetadata:
    def test_the_result_describes_the_run(self):
        result = run(frame_of(100))
        assert result.model_name == "RandomForestClassifier"
        assert result.n_rows == 100
        assert result.n_features == 2  # x1 and city; churn is the target

    def test_every_fold_produced_classification_metrics(self):
        result = run(frame_of(100))
        for fold in result.folds:
            assert {"accuracy", "f1_macro", "precision_macro", "recall_macro"} <= set(fold.metrics)

    def test_regression_folds_produce_regression_metrics(self):
        rng = np.random.default_rng(0)
        frame = pd.DataFrame({"x1": rng.normal(0, 1, 60), "price": rng.normal(100, 10, 60)})
        result = run(frame, target="price", task_type="regression")
        for fold in result.folds:
            assert {"mae", "mse", "rmse", "r2"} <= set(fold.metrics)

    def test_folds_are_numbered_from_one(self):
        result = run(frame_of(100))
        assert [f.fold for f in result.folds] == [1, 2, 3, 4, 5]


class TestConcentratedFoldError:
    """Why one fold scored far below the others, when a few rows explain it.

    Built from the Ames run: fold 3 scored R² 0.69 against a median of 0.888
    because two of its 292 rows -- huge houses sold incomplete, at a third of
    their built value -- produced 64% of its squared error. The report used to
    answer every such spread with "more data would tighten it", which was the
    opposite of the truth. Nothing about those two records improves with volume.
    """

    def _frame_with_one_liar(self, n: int = 200) -> pd.DataFrame:
        """A tight linear target, then one row priced as if the trend did not exist.

        The shape matters. The honest rows fit closely, so the fold's error is
        almost entirely the one bad record -- which is what makes a *few rows*
        the explanation rather than the dataset being hard. Its feature value is
        deliberately unremarkable and its price sits inside the target's ordinary
        range, so a quantile fence on the target could never find it.
        """
        rng = np.random.default_rng(7)
        x = rng.normal(0, 1, n)
        frame = pd.DataFrame({"x1": x, "price": 100 + 50 * x + rng.normal(0, 0.5, n)})
        frame.loc[0, ["x1", "price"]] = [0.0, 320.0]
        return frame

    def test_it_names_the_fold_and_the_rows_responsible(self):
        result = run(self._frame_with_one_liar(), target="price", task_type="regression")
        found = result.concentrated_fold_error
        assert found is not None
        assert found.n_dominant_rows <= 3
        assert found.dominant_error_share >= 0.5
        assert found.score < found.median_score

    def test_it_says_more_data_would_not_help(self):
        result = run(self._frame_with_one_liar(), target="price", task_type="regression")
        assert "more rows would not close it" in result.concentrated_fold_error.note

    def test_it_reports_what_the_fold_scores_without_them(self):
        result = run(self._frame_with_one_liar(), target="price", task_type="regression")
        found = result.concentrated_fold_error
        assert found.score_without_dominant > found.score

    def test_it_stays_quiet_when_the_folds_agree(self):
        """A diagnosis that fires on every run teaches the reader to skip it."""
        rng = np.random.default_rng(0)
        n = 200
        x = rng.normal(0, 1, n)
        frame = pd.DataFrame({"x1": x, "price": 100 + 20 * x + rng.normal(0, 1, n)})
        result = run(frame, target="price", task_type="regression")
        assert result.concentrated_fold_error is None

    def test_it_stays_quiet_when_the_error_is_spread_evenly(self):
        """A fold that is simply harder is not this finding."""
        rng = np.random.default_rng(3)
        n = 200
        frame = pd.DataFrame({"x1": rng.normal(0, 1, n), "price": rng.normal(100, 30, n)})
        result = run(frame, target="price", task_type="regression")
        found = result.concentrated_fold_error
        # Pure noise: no small set of rows owns the error, whatever the spread.
        assert found is None or found.n_dominant_rows > 3

    def test_classification_runs_produce_no_diagnosis(self):
        """Squared error of a label is not a thing, so this is regression-only."""
        assert run(frame_of(100)).concentrated_fold_error is None


class TestHowTheFoldsWereDrawn:
    """The report states the split; these pin down what it is told."""

    def test_a_shuffled_split_records_its_seed(self):
        result = run(frame_of(100), random_seed=42)
        assert result.shuffled is True
        assert result.random_seed == 42

    def test_a_time_ordered_split_is_not_shuffled(self):
        frame = frame_of(100)
        frame["signed_up"] = pd.date_range("2024-01-01", periods=100, freq="D")
        result = run(frame, time_column="signed_up")
        assert result.shuffled is False
