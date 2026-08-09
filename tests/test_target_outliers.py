"""The extreme tail of a numeric target: always measured, removed only on request.

Two separate promises, and the split between them is the point of the feature.

*Measured always*, because a reader looking at R² 0.07 deserves to be told that
250 rows carry most of the squared error rather than left to work it out.

*Removed only when the plan says so*, because deleting real records changes what
is being predicted. On an insurance-loss or peak-demand target the largest values
are the entire reason for the model, and a default that quietly dropped them
would produce a better-looking number about a question nobody asked.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.agents.schema_models import SchemaReport
from app.ml.cleaning import clean_frame
from app.ml.contracts import PlannerPlan

N_ROWS = 1_000


@pytest.fixture
def priced() -> pd.DataFrame:
    """A tame target with a handful of absurd values bolted onto the top."""
    rng = np.random.default_rng(2)
    price = np.round(rng.normal(100, 15, N_ROWS)).astype(float)
    # Four in a thousand: inside the half-percent the rule can reach, so the
    # tests below can assert that *all* of them go rather than most of them.
    price[:4] = [10_000, 12_000, 9_500, 13_000]
    return pd.DataFrame({"rooms": rng.normal(3, 1, N_ROWS), "price": price})


def _clean(frame: pd.DataFrame, *, trim: bool = False, task_type: str = "regression"):
    return clean_frame(
        frame,
        SchemaReport(n_rows=len(frame), n_columns=frame.shape[1], columns=[]),
        target="price",
        task_type=task_type,
        # Off so that every row count in this module is attributable to the tail
        # rule; duplicate removal has its own tests.
        plan=PlannerPlan(trim_target_outliers=trim, drop_duplicate_rows=False),
    )


class TestItIsAlwaysMeasured:
    def test_the_tail_is_reported_even_when_nothing_is_removed(self, priced):
        outliers = _clean(priced).report.target_outliers

        assert outliers is not None
        assert outliers.column == "price"
        assert outliers.n_detected > 0
        assert outliers.n_removed == 0
        assert outliers.maximum == 13_000

    def test_the_note_says_the_rows_were_kept(self, priced):
        note = _clean(priced).report.target_outliers.note
        assert "kept" in note
        assert "dominated" in note

    def test_measuring_does_not_remove_anything(self, priced):
        assert _clean(priced).frame.shape[0] == N_ROWS

    def test_a_classification_target_has_no_tail_to_measure(self):
        frame = pd.DataFrame({"x": range(200), "price": ["yes", "no"] * 100})
        assert _clean(frame, task_type="classification").report.target_outliers is None

    def test_an_evenly_spread_target_reports_its_tail_too(self):
        """Even a well-behaved target has an outermost half-percent.

        Reporting it is not a complaint -- it is the number that lets a reader
        see the tail is *not* the problem, which is worth as much as the
        opposite.
        """
        rng = np.random.default_rng(6)
        frame = pd.DataFrame({"x": rng.normal(0, 1, N_ROWS), "price": rng.normal(50, 5, N_ROWS)})
        outliers = _clean(frame).report.target_outliers
        assert outliers is not None
        assert outliers.n_detected <= N_ROWS * 0.01 + 1


class TestItIsRemovedOnlyOnRequest:
    def test_the_plan_switches_the_removal_on(self, priced):
        result = _clean(priced, trim=True)

        assert result.report.target_outliers.n_removed > 0
        assert result.frame.shape[0] < N_ROWS
        assert result.frame["price"].max() < 1_000

    def test_the_rows_removed_are_the_rows_reported(self, priced):
        result = _clean(priced, trim=True)
        outliers = result.report.target_outliers

        assert outliers.n_removed == outliers.n_detected
        assert result.frame.shape[0] == N_ROWS - outliers.n_removed

    def test_the_note_says_scores_describe_the_remainder(self, priced):
        note = _clean(priced, trim=True).report.target_outliers.note
        assert "were removed" in note
        assert "not the extremes" in note

    def test_nothing_is_removed_from_a_classification_target(self):
        frame = pd.DataFrame({"x": range(200), "price": ["yes", "no"] * 100})
        result = _clean(frame, trim=True, task_type="classification")
        assert result.frame.shape[0] == 200


class TestTheBoundIsBoundedByConstruction:
    def test_at_most_one_percent_of_rows_can_ever_leave(self):
        """The guard against a plan that would delete a genuine population.

        A lognormal target has no clean break between "typical" and "extreme",
        so an IQR fence would take several percent of it. A quantile rule cannot,
        whatever the distribution does.
        """
        rng = np.random.default_rng(3)
        heavy = pd.DataFrame(
            {
                "x": rng.normal(0, 1, N_ROWS),
                "price": np.round(np.exp(rng.normal(4.6, 1.2, N_ROWS))),
            }
        )

        result = _clean(heavy, trim=True)

        removed = N_ROWS - result.frame.shape[0]
        assert 0 < removed <= N_ROWS * 0.01 + 1, removed

    def test_a_dataset_too_small_for_a_tail_is_left_alone(self):
        """Below 200 rows the outermost half-percent is less than one row."""
        rng = np.random.default_rng(8)
        small = pd.DataFrame({"x": rng.normal(0, 1, 50), "price": rng.normal(100, 20, 50)})

        result = _clean(small, trim=True)

        assert result.report.target_outliers is None
        assert result.frame.shape[0] == 50

    def test_a_constant_target_is_not_reported_as_all_tail(self):
        # Caught by ``_guard_usable`` for a different reason, but the outlier
        # measurement must not be what raises -- its quantiles coincide.
        from app.ml.cleaning import CleaningError

        flat = pd.DataFrame({"x": range(300), "price": [7.0] * 300})
        with pytest.raises(CleaningError, match="only one distinct value"):
            _clean(flat, trim=True)


class TestNonFiniteTargets:
    """Not negotiable, unlike the tail: one infinity makes every metric useless."""

    def test_an_infinite_target_row_is_always_removed(self):
        rng = np.random.default_rng(9)
        frame = pd.DataFrame({"x": rng.normal(0, 1, 300), "price": rng.normal(100, 10, 300)})
        frame.loc[frame.index[:2], "price"] = [np.inf, -np.inf]

        result = _clean(frame)  # trim off -- this happens regardless

        assert result.report.non_finite_target_rows_removed == 2
        assert np.isfinite(result.frame["price"]).all()

    def test_a_clean_target_reports_none_removed(self, priced):
        assert _clean(priced).report.non_finite_target_rows_removed == 0
