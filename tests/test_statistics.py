"""Tests for descriptive statistics.

Pure arithmetic over DataFrames, so these assert on exact numbers rather than
"it produced something". The two behaviours worth pinning hardest are that
outliers are *counted and not removed* -- deleting them is a modelling decision
the user should make knowingly -- and that class balance is recomputed on the
cleaned frame rather than reused from upload, since cleaning can change it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.ml.statistics import compute_statistics, heatmap_columns, top_correlations


@pytest.fixture
def frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "age": [20.0, 30.0, 40.0, 50.0, 60.0, None],
            "score": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "city": ["London", "Leeds", "London", "Leeds", "London", "Bristol"],
            "churn": ["yes", "no", "no", "no", "no", "no"],
        }
    )


def stats_for(frame: pd.DataFrame, name: str, **kwargs):
    report = compute_statistics(
        frame,
        target=kwargs.get("target", "churn"),
        task_type=kwargs.get("task_type", "classification"),
    )
    return next(c for c in report.columns if c.name == name)


class TestNumericColumns:
    def test_the_five_number_summary_is_correct(self, frame):
        age = stats_for(frame, "age")
        assert age.numeric.minimum == 20.0
        assert age.numeric.median == 40.0
        assert age.numeric.maximum == 60.0
        assert age.numeric.mean == pytest.approx(40.0)

    def test_missing_values_are_counted_and_excluded_from_the_stats(self, frame):
        age = stats_for(frame, "age")
        assert age.missing == 1
        assert age.count == 5
        assert age.missing_rate == pytest.approx(1 / 6)
        # The mean is over the five present values, not five-sixths of six.
        assert age.numeric.mean == pytest.approx(40.0)

    def test_a_column_with_no_spread_reports_no_outliers(self):
        frame = pd.DataFrame({"flat": [5.0] * 10, "churn": ["yes", "no"] * 5})
        assert stats_for(frame, "flat").numeric.outlier_count == 0


class TestOutliersAreCountedNotRemoved:
    """Counting is the whole contract -- see the module docstring."""

    def test_an_extreme_value_is_counted(self):
        frame = pd.DataFrame(
            {
                "amount": list(np.linspace(1.0, 20.0, 20)) + [10_000.0],
                "churn": ["yes", "no"] * 10 + ["yes"],
            }
        )
        assert stats_for(frame, "amount").numeric.outlier_count == 1

    def test_a_column_with_no_interquartile_spread_reports_none(self):
        """Tukey's fences are degenerate when the middle half is a single value.

        Every non-repeated value would be flagged, which says more about the
        method than the data -- so the guard reports none rather than noise.
        """
        frame = pd.DataFrame(
            {"amount": [10.0] * 20 + [10_000.0], "churn": ["yes", "no"] * 10 + ["yes"]}
        )
        assert stats_for(frame, "amount").numeric.outlier_count == 0

    def test_the_outlier_still_influences_the_reported_maximum(self):
        """Proof it was left in the data rather than quietly dropped."""
        frame = pd.DataFrame(
            {
                "amount": list(np.linspace(1.0, 20.0, 20)) + [10_000.0],
                "churn": ["yes", "no"] * 10 + ["yes"],
            }
        )
        assert stats_for(frame, "amount").numeric.maximum == 10_000.0

    def test_a_clean_column_has_none(self):
        frame = pd.DataFrame({"amount": np.linspace(1, 20, 20), "churn": ["yes", "no"] * 10})
        assert stats_for(frame, "amount").numeric.outlier_count == 0


class TestCategoricalColumns:
    def test_cardinality_and_top_values(self, frame):
        city = stats_for(frame, "city")
        assert city.categorical.n_unique == 3
        assert city.categorical.top_values["London"] == 3

    def test_booleans_are_summarised_as_categories(self):
        """A mean of 0.62 over True/False is harder to read than two counts."""
        frame = pd.DataFrame({"flag": [True, False, True, True], "churn": ["y", "n", "y", "n"]})
        flag = stats_for(frame, "flag")
        assert flag.semantic_type == "boolean"
        assert flag.categorical is not None
        assert flag.numeric is None


class TestCorrelations:
    def test_a_perfect_relationship_is_found(self):
        frame = pd.DataFrame({"a": [1.0, 2, 3, 4, 5], "b": [2.0, 4, 6, 8, 10]})
        pairs = top_correlations(frame)
        assert pairs[0].correlation == pytest.approx(1.0)

    def test_a_negative_relationship_keeps_its_sign(self):
        frame = pd.DataFrame({"a": [1.0, 2, 3, 4, 5], "b": [10.0, 8, 6, 4, 2]})
        assert top_correlations(frame)[0].correlation == pytest.approx(-1.0)

    def test_each_pair_appears_once(self):
        frame = pd.DataFrame({"a": [1.0, 2, 3, 4], "b": [2.0, 4, 6, 8], "c": [1.0, 3, 2, 4]})
        pairs = [frozenset((p.left, p.right)) for p in top_correlations(frame)]
        assert len(pairs) == len(set(pairs))

    def test_no_column_correlates_with_itself(self):
        frame = pd.DataFrame({"a": [1.0, 2, 3, 4], "b": [2.0, 4, 6, 8]})
        assert all(p.left != p.right for p in top_correlations(frame))

    def test_pairs_are_ranked_by_strength_not_sign(self):
        frame = pd.DataFrame(
            {
                "a": [1.0, 2, 3, 4, 5],
                "b": [5.0, 4, 3, 2, 1],  # r = -1.0
                "c": [1.0, 2, 3, 4, 6],  # r ≈ +0.99 with a
            }
        )
        assert abs(top_correlations(frame)[0].correlation) == pytest.approx(1.0)

    def test_a_single_numeric_column_yields_nothing(self):
        assert top_correlations(pd.DataFrame({"a": [1.0, 2, 3]})) == []

    def test_categorical_columns_are_not_correlated(self, frame):
        names = {p.left for p in top_correlations(frame)} | {
            p.right for p in top_correlations(frame)
        }
        assert "city" not in names


class TestClassBalance:
    def test_balance_is_computed_for_classification(self, frame):
        report = compute_statistics(frame, target="churn", task_type="classification")
        assert report.class_balance.counts == {"no": 5, "yes": 1}
        assert report.class_balance.imbalanced is True

    def test_no_balance_for_regression(self, frame):
        report = compute_statistics(frame, target="score", task_type="regression")
        assert report.class_balance is None

    def test_a_balanced_target_is_not_flagged(self):
        frame = pd.DataFrame({"x": range(10), "churn": ["yes", "no"] * 5})
        report = compute_statistics(frame, target="churn", task_type="classification")
        assert report.class_balance.imbalance_ratio == pytest.approx(1.0)
        assert report.class_balance.imbalanced is False


class TestHeatmapColumns:
    def test_all_numeric_columns_are_kept_when_few(self, frame):
        assert set(heatmap_columns(frame)) == {"age", "score"}

    def test_the_list_is_capped_for_wide_datasets(self):
        """A grid of eighty columns is an unreadable smear."""
        rng = np.random.default_rng(0)
        wide = pd.DataFrame({f"c{i}": rng.normal(0, i + 1, 30) for i in range(60)})
        assert len(heatmap_columns(wide)) <= 25

    def test_the_most_variable_columns_survive_the_cap(self):
        """A near-constant column's correlations are dominated by noise."""
        rng = np.random.default_rng(0)
        wide = pd.DataFrame({f"c{i}": rng.normal(0, i + 1, 40) for i in range(40)})
        kept = heatmap_columns(wide)
        assert "c39" in kept  # widest spread
        assert "c0" not in kept  # narrowest


def test_the_report_describes_every_column(frame):
    report = compute_statistics(frame, target="churn", task_type="classification")
    assert {c.name for c in report.columns} == set(frame.columns)
    assert report.n_rows == 6
    assert report.n_columns == 4
