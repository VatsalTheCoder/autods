"""Modelling a skewed regression target on a log scale.

The dataset behind this is NYC Airbnb listings: ``price`` has skew 19.1, a median
of $106 and a maximum of $10,000. Squared-error training on that spends its
effort on the tail, and the run reported R² 0.07.

Two properties matter here and the rest is edge cases. First, the transform is
**applied inside the fold and undone before scoring**, so no metric is ever
computed in log units -- ``TestScoresStayInTheUsersUnits``. Second, it fires only
when it is actually warranted, because a transform nobody needs is a paragraph of
explanation the report has to carry for nothing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.compose import TransformedTargetRegressor

from app.ml.modeling import build_pipeline, cross_validate_model, unwrap_estimator
from app.ml.preprocessing import build_preprocessor
from app.ml.target import MIN_SKEW_FOR_LOG, choose_target_transform

N_ROWS = 300


def _skewed(n: int = N_ROWS, seed: int = 4) -> pd.Series:
    """A right-skewed, strictly positive target shaped like listing prices."""
    rng = np.random.default_rng(seed)
    return pd.Series(np.round(np.exp(rng.normal(4.6, 0.9, n)))).astype(float)


@pytest.fixture
def priced_frame() -> pd.DataFrame:
    """Features that genuinely predict a heavy-tailed price."""
    rng = np.random.default_rng(4)
    n = N_ROWS
    rooms = rng.integers(1, 5, n)
    central = rng.normal(0, 1, n)
    noise = rng.normal(0, 0.3, n)
    price = np.round(np.exp(3.4 + 0.7 * rooms + 0.45 * central + noise))
    return pd.DataFrame({"rooms": rooms.astype(float), "central": central, "price": price})


class TestWhenItFires:
    def test_a_heavily_skewed_positive_target_is_logged(self):
        transform = choose_target_transform(_skewed(), task_type="regression")
        assert transform is not None
        assert transform.name == "log1p"
        assert abs(transform.skew_after) < abs(transform.skew_before)

    def test_a_symmetric_target_is_left_alone(self):
        rng = np.random.default_rng(1)
        even = pd.Series(rng.normal(500, 80, N_ROWS))
        assert choose_target_transform(even, task_type="regression") is None

    def test_a_classification_target_is_never_transformed(self):
        labels = pd.Series(["yes", "no"] * (N_ROWS // 2))
        assert choose_target_transform(labels, task_type="classification") is None

    def test_a_target_with_negative_values_is_left_alone(self):
        """log1p is undefined below -1, and shifting invents an origin."""
        profit = _skewed() - 400.0
        assert profit.min() < 0
        assert choose_target_transform(profit, task_type="regression") is None

    def test_a_zero_is_fine_because_log1p_starts_at_zero(self):
        with_zeros = pd.concat([_skewed(), pd.Series([0.0] * 5)], ignore_index=True)
        assert choose_target_transform(with_zeros, task_type="regression") is not None

    def test_skew_the_log_cannot_fix_is_not_transformed(self):
        """A transform that leaves the target just as skewed is obfuscation."""
        # Two far-apart spikes: skewed, but not by a multiplicative mechanism.
        bimodal = pd.Series([1.0] * 280 + [1e6] * 20)
        assert abs(bimodal.skew()) > MIN_SKEW_FOR_LOG
        assert choose_target_transform(bimodal, task_type="regression") is None


class TestScoresStayInTheUsersUnits:
    """The property that makes the whole thing safe to report.

    A log target with the inverse missing would produce a magnificent R² about a
    question nobody asked.
    """

    def test_predictions_come_back_on_the_targets_own_scale(self, priced_frame):
        transform = choose_target_transform(priced_frame["price"], task_type="regression")
        assert transform is not None
        pipeline = build_pipeline(
            build_preprocessor(priced_frame, target="price", task_type="regression").transformer,
            "regression",
            random_seed=42,
            target_transform=transform,
        )
        X = priced_frame.drop(columns=["price"])

        predictions = pipeline.fit(X, priced_frame["price"]).predict(X)

        # Every prediction clears the *largest* value the log scale can produce,
        # which is what rules out the failure this test exists for -- an inverse
        # that never ran and left the caller holding log dollars.
        assert predictions.min() > np.log1p(priced_frame["price"]).max(), predictions[:5]
        assert np.median(predictions) == pytest.approx(priced_frame["price"].median(), rel=0.35)

    def test_the_reported_metrics_are_in_those_units(self, priced_frame):
        result = cross_validate_model(
            priced_frame,
            target="price",
            task_type="regression",
            preprocessor=build_preprocessor(
                priced_frame, target="price", task_type="regression"
            ).transformer,
            cv_folds=4,
        )

        assert result.target_transform is not None
        # MAE in log units would be a fraction; in dollars it is tens.
        assert all(fold.metrics["mae"] > 1.0 for fold in result.folds)

    def test_the_run_says_the_target_was_transformed(self, priced_frame):
        result = cross_validate_model(
            priced_frame,
            target="price",
            task_type="regression",
            preprocessor=build_preprocessor(
                priced_frame, target="price", task_type="regression"
            ).transformer,
            cv_folds=4,
        )
        assert any("log1p(target)" in w for w in result.warnings), result.warnings
        assert any("original units" in w for w in result.warnings), result.warnings


class TestItStaysInsideTheFold:
    def test_cross_validation_leaves_the_template_unfitted(self, priced_frame):
        """The wrapper must not become the one component fitted outside a fold."""
        from sklearn.exceptions import NotFittedError
        from sklearn.utils.validation import check_is_fitted

        result = cross_validate_model(
            priced_frame,
            target="price",
            task_type="regression",
            preprocessor=build_preprocessor(
                priced_frame, target="price", task_type="regression"
            ).transformer,
            cv_folds=4,
        )

        model = result.pipeline_template.named_steps["model"]
        assert isinstance(model, TransformedTargetRegressor)
        with pytest.raises(NotFittedError):
            check_is_fitted(model)


class TestUnwrapping:
    def test_the_real_estimator_is_reachable_through_the_wrapper(self, priced_frame):
        """SHAP and the model-name label both need what is underneath."""
        from sklearn.ensemble import RandomForestRegressor

        transform = choose_target_transform(priced_frame["price"], task_type="regression")
        wrapped = transform.wrap(RandomForestRegressor(3, random_state=0))

        assert isinstance(unwrap_estimator(wrapped), RandomForestRegressor)

        X = priced_frame.drop(columns=["price"])
        wrapped.fit(X, priced_frame["price"])
        assert isinstance(unwrap_estimator(wrapped), RandomForestRegressor)

    def test_a_plain_estimator_passes_straight_through(self):
        from sklearn.linear_model import LinearRegression

        model = LinearRegression()
        assert unwrap_estimator(model) is model
