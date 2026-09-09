"""Tests for the full-dataset refit (spec 7.9).

Everything else in ``app/ml`` is tested for keeping a fitted thing away from the
rows it will be scored on. This module is the deliberate exception, so its tests
are about keeping the exception *contained*:

* it fits on every row, and the artifact says so in a number a reader can check;
* the unfitted recipe it was handed is still unfitted afterwards, so the evidence
  the preprocessing node registered is not quietly invalidated;
* no score is computed here. The one in the artifact is the cross-validated
  figure, carried over -- a training-set score on a model fitted to those exact
  rows would be the most flattering and least meaningful number in the project.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted

from app.ml.contracts import ColumnStrategy, FeatureStrategy
from app.ml.final_training import FinalTrainingError, train_final_model
from app.ml.preprocessing import build_preprocessor

N_ROWS = 80


def frame_of(n: int = N_ROWS) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "age": rng.integers(18, 80, n),
            "city": rng.choice(["London", "Leeds"], n),
            "churn": ["yes", "no"] * (n // 2),
        }
    )


def train(frame: pd.DataFrame, *, model_name: str = "RandomForest", **kw):
    recipe = build_preprocessor(frame, target="churn").transformer
    return recipe, train_final_model(
        frame,
        target="churn",
        task_type="classification",
        preprocessor=recipe,
        model_name=model_name,
        **kw,
    )


class TestFittingOnEverything:
    def test_the_model_is_fitted(self):
        _, final = train(frame_of())
        check_is_fitted(final.pipeline.named_steps["model"])

    def test_the_artifact_records_the_full_row_count(self):
        """The check a reader can actually perform: does it match the dataset?"""
        frame = frame_of()
        _, final = train(frame)
        assert final.info.n_rows == len(frame)
        assert final.info.n_features == len(frame.columns) - 1

    def test_the_recipe_it_was_handed_is_still_unfitted(self):
        """The preprocessing node's artifact must not be invalidated behind its back.

        ``train_final_model`` clones before fitting for exactly this reason: the
        object on the pipeline state is the one whose unfittedness the test suite
        asserts elsewhere, and a refit that mutated it would make that claim
        depend on test ordering.
        """
        recipe, _ = train(frame_of())
        with pytest.raises(NotFittedError):
            check_is_fitted(recipe)

    def test_it_refuses_a_frame_with_no_features(self):
        """Refused here rather than failing later inside the fit, with a reason.

        The recipe is built from a frame that does have features, because a
        target-only frame cannot produce one at all -- the point is that this
        function checks its own inputs rather than relying on what it is given.
        """
        recipe = build_preprocessor(frame_of(), target="churn").transformer
        with pytest.raises(FinalTrainingError, match="No feature columns"):
            train_final_model(
                pd.DataFrame({"churn": ["yes", "no"] * 10}),
                target="churn",
                task_type="classification",
                preprocessor=recipe,
                model_name="RandomForest",
            )


class TestScoresAreNotRecomputed:
    def test_the_cross_validated_score_is_carried_over_verbatim(self):
        _, final = train(frame_of(), primary_metric="f1_macro", cv_score=0.8123)
        assert final.info.primary_metric == "f1_macro"
        assert final.info.cv_score == 0.8123

    def test_no_score_is_invented_when_none_was_given(self):
        """Absent, not zero. A zero would read as a measured result."""
        _, final = train(frame_of())
        assert final.info.cv_score is None


class TestTheWinnerIsRebuiltByName:
    @pytest.mark.parametrize(
        "model_name", ["RandomForest", "LogisticRegression", "XGBoost", "LightGBM"]
    )
    def test_each_roster_member_can_be_the_winner(self, model_name):
        _, final = train(frame_of(), model_name=model_name)
        assert final.info.model_name == model_name

    def test_an_unknown_winner_falls_back_and_says_so(self):
        """Should be impossible -- the names come from the same roster.

        Recorded rather than silent because the failure mode is serving a
        different model from the one the report names, which nothing downstream
        could detect.
        """
        _, final = train(frame_of(), model_name="Perceptron")
        assert final.info.model_name == "RandomForest"
        assert any("Perceptron" in warning for warning in final.info.warnings)

    def test_the_wrapped_classifier_reports_the_users_own_labels(self):
        """XGBoost trains on 0/1 internally; the artifact must not say so."""
        _, final = train(frame_of(), model_name="XGBoost")
        assert final.info.classes == ["no", "yes"]


class TestWhatThePredictionFormNeeds:
    def test_every_input_column_is_described(self):
        frame = frame_of()
        _, final = train(frame)
        assert [c.name for c in final.info.feature_columns] == ["age", "city"]
        assert all(column.dtype for column in final.info.feature_columns)

    def test_the_example_value_comes_from_the_data(self):
        """Pre-filling a form with an invented value teaches the wrong units."""
        frame = frame_of()
        _, final = train(frame)
        city = next(c for c in final.info.feature_columns if c.name == "city")
        assert city.example in set(frame["city"])

    def test_a_dropped_column_is_listed_but_marked_unused(self):
        """It stays in the model's input frame; it should not stay in the form.

        The recipe was fitted against a frame containing it, so omitting it
        entirely would break the transformer -- but a user asked for a customer
        ID the model discards would reasonably conclude it mattered.
        """
        frame = frame_of()
        frame["customer_id"] = [f"CUST-{i}" for i in range(len(frame))]
        strategy = FeatureStrategy(
            columns=[
                ColumnStrategy(column="customer_id", role="drop", rationale="an identifier"),
                ColumnStrategy(column="age", role="numeric", impute="median"),
                ColumnStrategy(column="city", role="categorical", encode="onehot"),
            ],
            source="llm",
        )
        recipe = build_preprocessor(frame, target="churn", strategy=strategy).transformer
        final = train_final_model(
            frame,
            target="churn",
            task_type="classification",
            preprocessor=recipe,
            model_name="RandomForest",
            strategy=strategy,
        )

        by_name = {c.name: c for c in final.info.feature_columns}
        assert by_name["customer_id"].used is False
        assert by_name["age"].used is True

    def test_columns_are_used_by_default_when_no_strategy_decided(self):
        _, final = train(frame_of())
        assert all(column.used for column in final.info.feature_columns)

    def test_a_column_that_is_entirely_missing_gets_no_example(self):
        frame = frame_of()
        frame["notes"] = None
        _, final = train(frame)
        notes = next(c for c in final.info.feature_columns if c.name == "notes")
        assert notes.example == ""


class TestResampling:
    def test_smote_is_applied_when_the_ranked_configuration_used_it(self):
        """The served model has to be the configuration that was ranked."""
        frame = pd.DataFrame(
            {
                "age": np.arange(60),
                "churn": ["no"] * 45 + ["yes"] * 15,
            }
        )
        recipe = build_preprocessor(frame, target="churn").transformer
        final = train_final_model(
            frame,
            target="churn",
            task_type="classification",
            preprocessor=recipe,
            model_name="RandomForest",
            use_smote=True,
        )
        assert "resample" in final.pipeline.named_steps
        assert "SMOTE" in final.info.resampling

    def test_a_refused_resampler_is_reported_not_silently_dropped(self):
        frame = pd.DataFrame({"age": np.arange(40), "churn": ["no"] * 38 + ["yes"] * 2})
        recipe = build_preprocessor(frame, target="churn").transformer
        final = train_final_model(
            frame,
            target="churn",
            task_type="classification",
            preprocessor=recipe,
            model_name="RandomForest",
            use_smote=True,
        )
        assert final.info.resampling == "none"
        assert final.info.warnings


class TestTheServedModelIsTheModelThatWasRanked:
    """The winner's *configuration*, not just its name.

    Cross-validation may fit a skewed regression target through ``log1p`` by
    wrapping the estimator in a ``TransformedTargetRegressor``. Final training
    rebuilt the winner from the roster by name and did not wrap it, so the model
    written to storage was a bare estimator carrying the score of a different
    one. On a target where the transform matters this is not a rounding
    difference: the ranked configuration scored r2 1.0 across every fold on
    ``y = expm1(2x)`` while the served one managed 0.479 against the data it had
    just been fitted on.

    ``final_training`` already takes this care with SMOTE, for a reason its own
    comment states -- the served model should be the configuration that was
    ranked, not a different one that happens to share its name. This asserts the
    same of the target transform.
    """

    @staticmethod
    def _exponential_frame() -> pd.DataFrame:
        x = np.linspace(0, 5, 300)
        return pd.DataFrame({"x": x, "y": np.expm1(2 * x)})

    def test_the_transform_cross_validation_used_survives_into_the_final_model(self):
        from sklearn.compose import TransformedTargetRegressor

        from app.ml.modeling import cross_validate_model

        frame = self._exponential_frame()
        recipe = build_preprocessor(frame, target="y", task_type="regression")
        cv = cross_validate_model(
            frame,
            target="y",
            task_type="regression",
            preprocessor=recipe.transformer,
            random_seed=0,
        )
        assert cv.target_transform is not None, "fixture no longer triggers a transform"

        final = train_final_model(
            frame,
            target="y",
            task_type="regression",
            preprocessor=recipe.transformer,
            model_name=cv.model_name,
            target_transform=cv.target_transform,
            random_seed=0,
        )
        served = final.pipeline.named_steps["model"]
        ranked = cv.pipeline_template.named_steps["model"]
        assert isinstance(served, TransformedTargetRegressor), (
            "the served model dropped the transform cross-validation measured"
        )
        assert type(served) is type(ranked)

    def test_the_served_model_fits_the_data_it_was_trained_on(self):
        """The symptom a reader would actually notice, stated as a floor.

        A model cannot normally fit its own training data worse than it scored
        on rows it never saw. When it does, the two are not the same model.
        """
        from sklearn.linear_model import LinearRegression
        from sklearn.metrics import r2_score

        from app.ml.modeling import Candidate, cross_validate_model

        # A linear model on purpose. A tree fits this curve well with or without
        # the transform, so it cannot tell the two configurations apart -- the
        # first draft of this test used the roster winner and passed even with
        # the transform dropped, which is worse than having no test at all.
        frame = self._exponential_frame()
        recipe = build_preprocessor(frame, target="y", task_type="regression")
        cv = cross_validate_model(
            frame,
            target="y",
            task_type="regression",
            preprocessor=recipe.transformer,
            random_seed=0,
            candidate=Candidate(name="LinearRegression", estimator=LinearRegression()),
        )
        assert cv.target_transform is not None
        final = train_final_model(
            frame,
            target="y",
            task_type="regression",
            preprocessor=recipe.transformer,
            model_name="LinearRegression",
            target_transform=cv.target_transform,
            random_seed=0,
        )
        fitted_r2 = r2_score(frame["y"], final.pipeline.predict(frame[["x"]]))
        assert fitted_r2 > 0.99, f"served model scores {fitted_r2:.3f} on its own training data"

    def test_no_transform_leaves_the_estimator_bare(self):
        """The common path must not grow a wrapper it does not need."""
        from sklearn.compose import TransformedTargetRegressor

        frame = pd.DataFrame({"x": np.linspace(0, 1, 60), "y": np.linspace(0, 1, 60)})
        recipe = build_preprocessor(frame, target="y", task_type="regression")
        final = train_final_model(
            frame,
            target="y",
            task_type="regression",
            preprocessor=recipe.transformer,
            model_name="Ridge",
            target_transform=None,
            random_seed=0,
        )
        assert not isinstance(final.pipeline.named_steps["model"], TransformedTargetRegressor)
