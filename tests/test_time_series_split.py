"""Time-ordered cross-validation.

The rest of the leakage work in this repo defends one kind of leak: preprocessing
fitted across a split. This defends a different one, and neither implies the
other. A pipeline can be immaculate about fold hygiene and still report an
inflated score, because random folds on time-ordered data let a model learn from
next week to predict last week.

The project's own critic raised it unprompted on a house-prices run, and the
PaySim fraud dataset is the case that made it concrete: ``step`` is an hour index,
so every score from a random split is optimistic by an unknown amount.

The load-bearing assertion here is ``test_no_training_row_comes_after_its_test_rows``.
Everything else is about behaviour around the edges; that one is the guarantee.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.ml.modeling import ModelingError, cross_validate_model, order_by_time
from app.ml.preprocessing import build_preprocessor

N_ROWS = 60


@pytest.fixture
def timed_frame() -> pd.DataFrame:
    """Rows deliberately *shuffled*, so sorting has to actually happen."""
    rng = np.random.default_rng(7)
    frame = pd.DataFrame(
        {
            "day": pd.date_range("2026-01-01", periods=N_ROWS, freq="D"),
            "x": rng.normal(0, 1, N_ROWS),
            "churn": ["yes", "no"] * (N_ROWS // 2),
        }
    )
    return frame.sample(frac=1.0, random_state=3).reset_index(drop=True)


def _run(frame: pd.DataFrame, **kwargs):
    return cross_validate_model(
        frame,
        target="churn",
        task_type="classification",
        preprocessor=build_preprocessor(frame, target="churn").transformer,
        cv_folds=4,
        **kwargs,
    )


def _undate(frame: pd.DataFrame, n: int) -> pd.DataFrame:
    """Make ``n`` of the frame's dates unreadable, as a CSV would deliver them.

    As strings, because a ``datetime64`` column cannot hold the bad values that
    make this case exist in the first place.
    """
    frame = frame.copy()
    frame["day"] = frame["day"].dt.strftime("%Y-%m-%d")
    frame.loc[frame.index[:n], "day"] = "not a date"
    return frame


class TestOrdering:
    def test_rows_come_back_oldest_first(self, timed_frame):
        ordered, applied = order_by_time(timed_frame, "day", [])
        assert applied
        assert ordered["day"].is_monotonic_increasing

    def test_the_sort_key_is_not_left_behind(self, timed_frame):
        """A scratch column reaching the recipe would become a feature."""
        ordered, _ = order_by_time(timed_frame, "day", [])
        assert list(ordered.columns) == list(timed_frame.columns)

    def test_no_rows_are_gained_or_lost(self, timed_frame):
        ordered, _ = order_by_time(timed_frame, "day", [])
        assert len(ordered) == len(timed_frame)
        assert sorted(ordered["x"].tolist()) == sorted(timed_frame["x"].tolist())

    def test_a_column_with_no_dates_at_all_is_refused(self, timed_frame):
        frame = timed_frame.assign(day="not a date")
        with pytest.raises(ModelingError, match="no readable dates"):
            order_by_time(frame, "day", [])

    def test_a_missing_column_is_refused(self, timed_frame):
        with pytest.raises(ModelingError, match="not in the dataset"):
            order_by_time(timed_frame, "nope", [])


class TestUndatedRowsAreSetAsideNotSortedToOneEnd:
    """The defect this class exists for.

    Undated rows used to sort last, which handed the final fold a validation set
    made entirely of them -- a different population from anything trained on. NYC
    Airbnb is the case that showed it: a fifth of the listings have no
    ``last_review`` and the last fold scored negative R² against a wall of them.
    """

    def test_they_are_left_out_of_the_ordering_and_counted(self, timed_frame):
        warnings: list[str] = []

        ordered, applied = order_by_time(_undate(timed_frame, 3), "day", warnings)

        assert applied
        assert len(ordered) == N_ROWS - 3
        assert pd.to_datetime(ordered["day"], errors="coerce").notna().all()
        assert any("3 rows" in w and "left out" in w for w in warnings), warnings

    def test_no_undated_row_survives_at_either_end(self, timed_frame):
        """Sorting them first would be the same bug pointed the other way."""
        ordered, _ = order_by_time(_undate(timed_frame, 3), "day", [])
        parsed = pd.to_datetime(ordered["day"], errors="coerce")
        assert parsed.is_monotonic_increasing
        assert not parsed.head(3).isna().any()

    def test_too_many_undated_rows_abandons_time_ordering_entirely(self, timed_frame):
        """Past the threshold the dated remainder is no longer the dataset."""
        warnings: list[str] = []
        frame = _undate(timed_frame, 20)  # a third of 60, well past 10%

        ordered, applied = order_by_time(frame, "day", warnings)

        assert not applied
        assert len(ordered) == N_ROWS, "the caller's rows must all still be there"
        assert any("not ordered by time" in w and "33%" in w for w in warnings), warnings

    def test_the_run_falls_back_to_random_folds_and_says_so(self, timed_frame):
        result = _run(_undate(timed_frame, 20), time_column="day")

        assert result.cv_strategy == "StratifiedKFold"
        assert result.n_rows == N_ROWS
        assert any("not ordered by time" in w for w in result.warnings), result.warnings

    def test_a_few_undated_rows_still_get_time_ordered_folds(self, timed_frame):
        result = _run(_undate(timed_frame, 3), time_column="day")

        assert result.cv_strategy == "TimeSeriesSplit"
        assert result.n_rows == N_ROWS - 3


class TestTheGuarantee:
    def test_no_training_row_comes_after_its_test_rows(self, timed_frame):
        """The whole point: a fold never trains on rows later than it scores.

        Asserted on the dates themselves rather than on the splitter's type,
        because the class name is not the guarantee -- ``TimeSeriesSplit`` over an
        unsorted frame satisfies the name and violates the property.
        """
        from sklearn.model_selection import TimeSeriesSplit

        ordered, _ = order_by_time(timed_frame, "day", [])
        days = ordered["day"].to_numpy()

        for train_idx, test_idx in TimeSeriesSplit(n_splits=4).split(ordered):
            assert days[train_idx].max() <= days[test_idx].min(), (
                "a training row is dated later than a row it is scored against"
            )

    def test_the_strategy_is_recorded_as_time_ordered(self, timed_frame):
        result = _run(timed_frame, time_column="day")
        assert result.cv_strategy == "TimeSeriesSplit"

    def test_without_a_time_column_nothing_changes(self, timed_frame):
        """The default has to stay exactly what every earlier run did."""
        result = _run(timed_frame)
        assert result.cv_strategy == "StratifiedKFold"

    def test_the_caller_s_frame_is_not_reordered(self, timed_frame):
        """Sorting is internal; a caller's dataframe is not theirs to rearrange."""
        before = timed_frame["x"].tolist()
        _run(timed_frame, time_column="day")
        assert timed_frame["x"].tolist() == before


class TestImbalanceIsWarnedAboutNotHidden:
    def test_a_single_class_fold_is_reported(self):
        """Time-ordered folds cannot be stratified, and on a rare event an early
        fold can hold no positives. That is a real property of validating
        chronologically, so it is warned about rather than silently avoided."""
        rng = np.random.default_rng(11)
        n = 60
        # Every positive case sits at the end, so the early folds have none.
        churn = ["no"] * (n - 6) + ["yes"] * 6
        frame = pd.DataFrame(
            {
                "day": pd.date_range("2026-01-01", periods=n, freq="D"),
                "x": rng.normal(0, 1, n),
                "churn": churn,
            }
        )

        result = _run(frame, time_column="day")

        assert result.cv_strategy == "TimeSeriesSplit"
        assert any("single class" in w for w in result.warnings), result.warnings

    def test_a_balanced_run_says_nothing_about_it(self, timed_frame):
        result = _run(timed_frame, time_column="day")
        assert not any("single class" in w for w in result.warnings)


class TestATimeColumnThatCountsRatherThanDates:
    """PaySim numbers its hours in ``step`` and has no date anywhere.

    That dataset is what prompted this feature, so refusing counters would have
    meant shipping time-ordered validation that could not be applied to the case
    it was built for. Ordering needs values that compare, not values that parse.
    """

    @pytest.fixture
    def counted_frame(self) -> pd.DataFrame:
        rng = np.random.default_rng(5)
        frame = pd.DataFrame(
            {
                "step": np.arange(N_ROWS),
                "x": rng.normal(0, 1, N_ROWS),
                "churn": ["yes", "no"] * (N_ROWS // 2),
            }
        )
        return frame.sample(frac=1.0, random_state=9).reset_index(drop=True)

    def test_rows_come_back_in_counter_order(self, counted_frame):
        ordered, _ = order_by_time(counted_frame, "step", [])
        assert ordered["step"].is_monotonic_increasing

    def test_a_counter_is_not_read_as_nanoseconds_since_the_epoch(self, counted_frame):
        """Parsing 0..59 as dates would 'work' and mean nothing."""
        ordered, _ = order_by_time(counted_frame, "step", [])
        assert ordered["step"].tolist() == list(range(N_ROWS))

    def test_the_run_uses_time_ordered_folds(self, counted_frame):
        result = cross_validate_model(
            counted_frame,
            target="churn",
            task_type="classification",
            preprocessor=build_preprocessor(counted_frame, target="churn").transformer,
            cv_folds=4,
            time_column="step",
        )
        assert result.cv_strategy == "TimeSeriesSplit"
