"""Tests for the cleaning stage.

Pure functions over DataFrames -- no database, no storage, no worker -- which is
the payoff of keeping the thinking out of the graph nodes.

The most important test here is ``TestMissingValuesAreLeftAlone``. Cleaning
looks like the natural place to fill in blanks, and filling them here would leak
test-fold information into training through the median (see ``ml/cleaning.py``).
So the correct behaviour is the counter-intuitive one: count the gaps and leave
them, which is exactly what that class pins down.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.ml.cleaning import CleaningError, clean_frame
from app.ml.contracts import PlannerPlan
from app.services.profiling import profile_dataset


def clean(frame: pd.DataFrame, *, target: str = "churn", task_type: str = "classification", **kw):
    """Clean a frame, profiling it first the way the runner does."""
    return clean_frame(frame, profile_dataset(frame), target=target, task_type=task_type, **kw)


@pytest.fixture
def frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "email": ["a@b.com", "c@d.com", "e@f.com", "g@h.com", "i@j.com", "k@l.com"],
            "age": [34, 28, 45, 51, 39, 22],
            "city": ["London", "Leeds", "Bristol", "London", "Leeds", "Bristol"],
            "churn": ["yes", "no", "no", "yes", "no", "yes"],
        }
    )


class TestExcludedColumns:
    def test_excluded_columns_are_dropped(self, frame):
        result = clean(frame, excluded=["email"])
        assert "email" not in result.frame.columns
        assert any(d.name == "email" for d in result.report.dropped_columns)

    def test_the_target_is_never_dropped_even_if_excluded(self, frame):
        """A confused request must not remove the thing being predicted."""
        result = clean(frame, excluded=["churn"])
        assert "churn" in result.frame.columns

    def test_a_missing_target_is_a_clear_error(self, frame):
        with pytest.raises(CleaningError, match="not in the dataset"):
            clean(frame, target="nope")


class TestRows:
    def test_duplicate_rows_are_removed(self, frame):
        doubled = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
        result = clean(doubled)
        assert result.report.duplicate_rows_removed == 1
        assert result.report.n_rows_after == len(frame)

    def test_duplicates_are_kept_when_the_plan_says_so(self, frame):
        doubled = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
        result = clean(doubled, plan=PlannerPlan(drop_duplicate_rows=False))
        assert result.report.duplicate_rows_removed == 0
        assert result.report.n_rows_after == len(doubled)

    def test_duplicates_are_judged_after_excluded_columns_go(self, frame):
        """Two rows differing only in a dropped ID column are duplicates.

        The ordering matters: detect duplicates first and a unique email would
        hide an otherwise identical record.
        """
        near = frame.copy()
        near.loc[len(near)] = ["unique@x.com", 34, "London", "yes"]
        result = clean(near, excluded=["email"])
        assert result.report.duplicate_rows_removed == 1

    def test_rows_with_no_target_are_removed(self, frame):
        frame.loc[2, "churn"] = None
        result = clean(frame)
        assert result.report.missing_target_rows_removed == 1
        assert result.frame["churn"].notna().all()


class TestColumns:
    def test_constant_columns_are_dropped(self, frame):
        frame["country"] = "UK"
        result = clean(frame)
        assert "country" not in result.frame.columns
        assert any("constant" in d.reason for d in result.report.dropped_columns)

    def test_mostly_empty_columns_are_dropped(self, frame):
        # Two distinct non-null values, so this is dropped for being 67% empty
        # rather than for being constant -- the rule under test here.
        frame["notes"] = [None, None, None, None, "x", "y"]
        result = clean(frame)
        assert "notes" not in result.frame.columns
        dropped = result.report.dropped_columns
        assert any(d.name == "notes" and "missing" in d.reason for d in dropped)

    def test_mostly_empty_columns_are_kept_when_the_plan_says_so(self, frame):
        frame["notes"] = [None, None, None, None, "x", "y"]
        result = clean(frame, plan=PlannerPlan(drop_high_null_columns=False))
        assert "notes" in result.frame.columns

    def test_a_sparse_column_under_the_threshold_survives(self, frame):
        frame["score"] = [1.0, 2.0, None, 4.0, 5.0, 6.0]  # 17% missing
        result = clean(frame, max_null_rate=0.6)
        assert "score" in result.frame.columns


class TestDtypeCorrections:
    def test_high_cardinality_numeric_strings_become_numeric(self):
        """A column of many distinct numeric strings is a number, not a category."""
        frame = pd.DataFrame(
            {
                "amount": [str(v) for v in np.arange(1000, 1040)],
                "churn": ["yes", "no"] * 20,
            }
        )
        result = clean(frame)
        assert pd.api.types.is_numeric_dtype(result.frame["amount"])
        assert any(c.name == "amount" for c in result.report.dtype_corrections)

    def test_low_cardinality_numeric_strings_stay_categorical(self, frame):
        """A small set of numeric codes is a category; one-hot handles it better."""
        frame["band"] = ["1", "2", "3", "1", "2", "3"]
        result = clean(frame)
        assert not pd.api.types.is_numeric_dtype(result.frame["band"])

    def test_a_mostly_unparseable_column_is_left_alone(self):
        """Coercion must not silently blank a column that is really text."""
        frame = pd.DataFrame(
            {
                "note": [f"comment {i}" for i in range(40)],
                "churn": ["yes", "no"] * 20,
            }
        )
        result = clean(frame)
        assert not result.report.dtype_corrections
        assert result.frame["note"].notna().all()

    def test_a_regression_target_stored_as_text_is_coerced(self):
        frame = pd.DataFrame(
            {
                "age": list(range(20, 60)),
                "price": [str(float(v)) for v in np.arange(100, 140)],
            }
        )
        result = clean(frame, target="price", task_type="regression")
        assert pd.api.types.is_numeric_dtype(result.frame["price"])

    def test_a_classification_target_is_not_coerced(self):
        """The user said these are labels, so they stay labels."""
        frame = pd.DataFrame(
            {
                "age": list(range(20, 60)),
                "band": [str(v % 3) for v in range(40)],
            }
        )
        result = clean(frame, target="band", task_type="classification")
        assert not pd.api.types.is_numeric_dtype(result.frame["band"])


class TestMissingValuesAreLeftAlone:
    """Cleaning must not impute. This is a leakage guard, not a style preference."""

    def test_gaps_survive_cleaning(self, frame):
        frame["age"] = [34, None, 45, 51, None, 22]
        result = clean(frame)
        assert result.frame["age"].isna().sum() == 2

    def test_gaps_are_reported_for_the_pipeline_to_handle(self, frame):
        frame["age"] = [34, None, 45, 51, None, 22]
        result = clean(frame)
        assert result.report.missing_values_left_to_the_pipeline == {"age": 2}

    def test_no_gaps_reports_nothing(self, frame):
        result = clean(frame)
        assert result.report.missing_values_left_to_the_pipeline == {}


class TestUnusableDatasets:
    """Fail early with a reason a user can act on, not deep inside scikit-learn."""

    def test_no_features_left_is_an_error(self):
        frame = pd.DataFrame({"only": ["a", "b", "c", "d"], "churn": ["y", "n", "y", "n"]})
        with pytest.raises(CleaningError, match="No usable feature columns"):
            clean(frame, excluded=["only"])

    def test_a_single_class_target_is_an_error(self):
        frame = pd.DataFrame({"age": [1, 2, 3, 4], "churn": ["yes"] * 4})
        with pytest.raises(CleaningError, match="only one distinct value"):
            clean(frame)

    def test_too_few_rows_is_an_error(self):
        frame = pd.DataFrame({"age": [1], "churn": ["yes"]})
        with pytest.raises(CleaningError):
            clean(frame)


class TestReportArithmetic:
    def test_the_report_counts_match_the_frame(self, frame):
        frame["country"] = "UK"
        result = clean(frame, excluded=["email"])
        assert result.report.n_rows_after == result.frame.shape[0]
        assert result.report.n_columns_after == result.frame.shape[1]
        assert result.report.n_rows_before == frame.shape[0]
        assert result.report.n_columns_before == frame.shape[1]

    def test_the_input_frame_is_not_mutated(self, frame):
        before = frame.copy()
        clean(frame, excluded=["email"])
        pd.testing.assert_frame_equal(frame, before)
