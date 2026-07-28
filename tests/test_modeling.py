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
