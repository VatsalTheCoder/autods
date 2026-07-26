"""Tests for chart rendering.

A chart's *appearance* is not automatically testable and was checked by eye. What
is testable, and what these cover, is the contract around it: that real PNG bytes
come back, that charts which do not apply are omitted rather than emitted blank,
that a broken column costs one chart and not the run, and that figures are closed
so a long-lived worker does not leak memory across jobs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from matplotlib import pyplot as plt

from app.ml.plots import render_charts, render_cluster_scatter

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


@pytest.fixture
def frame() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 120
    return pd.DataFrame(
        {
            "age": rng.normal(40, 9, n),
            "income": rng.normal(50_000, 8_000, n),
            "city": rng.choice(["London", "Leeds", "Bristol"], n),
            "churn": rng.choice(["yes", "no"], n),
        }
    )


def names_of(charts) -> set[str]:
    return {c.name for c in charts}


class TestTheChartSet:
    def test_the_expected_charts_are_produced(self, frame):
        charts = render_charts(frame, target="churn", task_type="classification")
        assert names_of(charts) >= {
            "target_distribution.png",
            "numeric_distributions.png",
            "numeric_boxplots.png",
            "correlation_heatmap.png",
        }

    def test_every_chart_is_a_real_png(self, frame):
        for chart in render_charts(frame, target="churn", task_type="classification"):
            assert chart.png.startswith(PNG_MAGIC), chart.name
            assert len(chart.png) > 1000

    def test_every_chart_has_a_title(self, frame):
        for chart in render_charts(frame, target="churn", task_type="classification"):
            assert chart.title

    def test_names_are_unique(self, frame):
        charts = render_charts(frame, target="churn", task_type="classification")
        assert len(names_of(charts)) == len(charts)


class TestChartsThatDoNotApplyAreOmitted:
    """A blank chart is worse than no chart -- it looks like a failure."""

    def test_no_missingness_chart_when_nothing_is_missing(self, frame):
        charts = render_charts(frame, target="churn", task_type="classification")
        assert "missing_values.png" not in names_of(charts)

    def test_a_missingness_chart_appears_when_there_are_gaps(self, frame):
        frame.loc[:20, "income"] = np.nan
        charts = render_charts(frame, target="churn", task_type="classification")
        assert "missing_values.png" in names_of(charts)

    def test_no_heatmap_with_a_single_numeric_column(self):
        frame = pd.DataFrame({"age": [1.0, 2, 3, 4], "churn": ["y", "n", "y", "n"]})
        charts = render_charts(frame, target="churn", task_type="classification")
        assert "correlation_heatmap.png" not in names_of(charts)

    def test_no_numeric_charts_when_there_are_no_numeric_features(self):
        frame = pd.DataFrame({"city": ["London", "Leeds"] * 10, "churn": ["yes", "no"] * 10})
        charts = render_charts(frame, target="churn", task_type="classification")
        assert "numeric_distributions.png" not in names_of(charts)


class TestRegressionTargets:
    def test_a_continuous_target_is_drawn_as_a_histogram(self):
        rng = np.random.default_rng(0)
        frame = pd.DataFrame({"x": rng.normal(0, 1, 80), "price": rng.normal(100, 20, 80)})
        charts = render_charts(frame, target="price", task_type="regression")
        assert "target_distribution.png" in names_of(charts)


class TestRobustness:
    def test_one_broken_chart_does_not_lose_the_others(self, frame, monkeypatch):
        """EDA is descriptive: a renderer failing must not cost the whole set."""
        import app.ml.plots as plots

        def boom(*_args, **_kwargs):
            raise ValueError("chart exploded")

        monkeypatch.setattr(plots, "_correlation_heatmap", boom)
        charts = render_charts(frame, target="churn", task_type="classification")

        assert "correlation_heatmap.png" not in names_of(charts)
        assert "target_distribution.png" in names_of(charts)

    def test_figures_are_closed(self, frame):
        """A long-lived worker renders for job after job; leaks accumulate."""
        plt.close("all")
        render_charts(frame, target="churn", task_type="classification")
        assert plt.get_fignums() == []

    def test_a_missing_target_column_is_survivable(self, frame):
        charts = render_charts(frame, target="not_a_column", task_type="classification")
        assert "target_distribution.png" not in names_of(charts)


class TestClusterScatter:
    def test_a_scatter_is_drawn(self):
        rng = np.random.default_rng(0)
        coords = np.vstack([rng.normal(c, 0.5, (40, 2)) for c in [(0, 0), (5, 5)]])
        labels = np.repeat([0, 1], 40)
        chart = render_cluster_scatter(coords, labels)
        assert chart.png.startswith(PNG_MAGIC)
        assert chart.name == "cluster_scatter.png"

    def test_more_groups_than_hues_still_renders(self):
        """Past four clusters, identity moves to hue x shape -- it must not crash."""
        rng = np.random.default_rng(0)
        centres = [(0, 0), (5, 0), (10, 0), (0, 5), (5, 5), (10, 5), (0, 10)]
        coords = np.vstack([rng.normal(c, 0.4, (20, 2)) for c in centres])
        labels = np.repeat(np.arange(len(centres)), 20)
        chart = render_cluster_scatter(coords, labels)
        assert chart.png.startswith(PNG_MAGIC)

    def test_empty_input_produces_nothing(self):
        assert render_cluster_scatter(np.empty((0, 2)), np.array([])) is None
