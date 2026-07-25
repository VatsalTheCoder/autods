"""Tests for clustering.

The guardrail -- cluster labels never becoming model features -- lives in
``test_leakage.py`` alongside the other leakage proofs, because that is what it
is. This file covers the rest: that the method is chosen from the data rather
than taken on trust, that k is genuinely searched, that groups planted in
synthetic data are actually recovered, and that a dataset clustering cannot
handle costs the run its scatter plot and nothing more.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.ml.clustering import choose_method, profile_clusters, run_clustering
from app.ml.contracts import PlannerPlan


def blobs(rng, n: int, age: float, income: float, city: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "age": rng.normal(age, 3, n),
            "income": rng.normal(income, 4000, n),
            "city": [city] * n,
            "churn": rng.choice(["yes", "no"], n),
        }
    )


@pytest.fixture
def three_groups() -> pd.DataFrame:
    """Three clearly separated groups, so recovery is unambiguous."""
    rng = np.random.default_rng(5)
    return pd.concat(
        [
            blobs(rng, 100, 25, 30_000, "London"),
            blobs(rng, 100, 45, 80_000, "Leeds"),
            blobs(rng, 100, 65, 50_000, "Bristol"),
        ],
        ignore_index=True,
    )


class TestMethodSelection:
    """Spec 9: the planner proposes, the data decides."""

    def test_categorical_data_forces_kprototypes(self, three_groups):
        method, reason = choose_method(three_groups, "kmeans")
        assert method == "kprototypes"
        assert "fabricated" in reason

    def test_all_numeric_data_forces_kmeans(self):
        frame = pd.DataFrame({"a": [1.0, 2, 3], "b": [4.0, 5, 6]})
        method, reason = choose_method(frame, "kprototypes")
        assert method == "kmeans"
        assert reason

    def test_a_correct_choice_is_left_alone(self, three_groups):
        method, reason = choose_method(three_groups, "kprototypes")
        assert method == "kprototypes"
        assert reason is None

    def test_an_override_is_recorded_in_the_report(self, three_groups):
        """An artifact must never imply a choice that was not obeyed."""
        result = run_clustering(
            three_groups, target="churn", plan=PlannerPlan(clustering_method="kmeans")
        )
        assert result.report.method == "kprototypes"
        assert result.report.method_override_reason


class TestFindingGroups:
    def test_planted_groups_are_recovered(self, three_groups):
        result = run_clustering(three_groups, target="churn")
        assert result.report.k == 3

    def test_the_groups_are_well_separated(self, three_groups):
        result = run_clustering(three_groups, target="churn")
        assert result.report.silhouette > 0.5
        assert not result.report.is_weak()

    def test_k_is_searched_not_assumed(self, three_groups):
        """Several k values are tried and the best kept -- visible in the report."""
        result = run_clustering(three_groups, target="churn")
        assert len(result.report.silhouette_by_k) > 1
        best = max(result.report.silhouette_by_k, key=lambda k: result.report.silhouette_by_k[k])
        assert result.report.k == best

    def test_every_row_lands_in_a_group(self, three_groups):
        result = run_clustering(three_groups, target="churn")
        assert len(result.labels) == len(three_groups)

    def test_a_scatter_is_produced(self, three_groups):
        result = run_clustering(three_groups, target="churn")
        assert result.scatter is not None
        assert result.scatter.png.startswith(b"\x89PNG")

    def test_the_target_is_not_clustered_on(self, three_groups):
        """Grouping rows by the answer would make the groups meaningless."""
        result = run_clustering(three_groups, target="churn")
        named = {name for p in result.report.profiles for name in p.distinguishing_features}
        assert "churn" not in named

    def test_pure_noise_is_flagged_rather_than_presented_as_a_finding(self):
        """The failure mode that matters: clustering always returns groups.

        K-means partitions a single Gaussian blob into respectable-looking
        "clusters" scoring around 0.36 -- so a threshold that only caught scores
        below 0.25 would let the tool report invented segments in random data as
        discoveries. Both the flag and the explanation are asserted, because a
        silent flag helps nobody reading the report.
        """
        rng = np.random.default_rng(0)
        noise = pd.DataFrame(
            {
                "a": rng.normal(0, 1, 200),
                "b": rng.normal(0, 1, 200),
                "churn": rng.choice(["yes", "no"], 200),
            }
        )
        result = run_clustering(noise, target="churn")

        assert result.report.is_weak()
        assert any("data that has none" in w for w in result.report.warnings)

    def test_genuinely_separated_groups_are_not_flagged_as_weak(self, three_groups):
        """The threshold must not cry wolf on real structure."""
        assert not run_clustering(three_groups, target="churn").report.is_weak()


class TestProfiles:
    def test_every_group_is_profiled(self, three_groups):
        result = run_clustering(three_groups, target="churn")
        assert len(result.report.profiles) == result.report.k

    def test_shares_sum_to_one(self, three_groups):
        result = run_clustering(three_groups, target="churn")
        assert sum(p.share for p in result.report.profiles) == pytest.approx(1.0)

    def test_distinguishing_features_describe_real_differences(self):
        """A group with high income must be described as having high income."""
        rng = np.random.default_rng(1)
        frame = pd.concat(
            [blobs(rng, 80, 30, 20_000, "A"), blobs(rng, 80, 30, 90_000, "B")],
            ignore_index=True,
        )
        labels = np.array([0] * 80 + [1] * 80)
        profiles = profile_clusters(
            frame.drop(columns=["churn"]), labels, numeric=["age", "income"], categorical=["city"]
        )
        rich = next(p for p in profiles if p.cluster == 1)
        assert "higher than average" in rich.distinguishing_features["income"]

    def test_an_unremarkable_group_lists_nothing(self):
        """No invented distinctions when a group is simply average."""
        frame = pd.DataFrame({"a": [1.0] * 20})
        profiles = profile_clusters(frame, np.zeros(20, dtype=int), numeric=["a"], categorical=[])
        assert profiles[0].distinguishing_features == {}


class TestDegradingGracefully:
    """Clustering is descriptive -- it must never cost a run its model."""

    def test_the_plan_can_turn_it_off(self, three_groups):
        result = run_clustering(
            three_groups, target="churn", plan=PlannerPlan(run_clustering=False)
        )
        assert result.labels is None
        assert result.skipped_reason

    def test_too_few_rows_skips_rather_than_raises(self):
        frame = pd.DataFrame({"a": [1.0, 2.0, 3.0], "churn": ["y", "n", "y"]})
        result = run_clustering(frame, target="churn")
        assert result.labels is None
        assert "too few" in result.skipped_reason.lower()

    def test_a_frame_with_only_a_target_skips(self):
        frame = pd.DataFrame({"churn": ["yes", "no"] * 20})
        result = run_clustering(frame, target="churn")
        assert result.labels is None
        assert result.skipped_reason

    def test_a_skipped_run_still_returns_a_readable_report(self, three_groups):
        result = run_clustering(
            three_groups, target="churn", plan=PlannerPlan(run_clustering=False)
        )
        assert result.report.k == 0
        assert result.report.warnings


class TestReproducibility:
    def test_the_same_data_gives_the_same_groups(self, three_groups):
        first = run_clustering(three_groups, target="churn")
        second = run_clustering(three_groups, target="churn")
        assert first.report.k == second.report.k
        assert np.array_equal(first.labels, second.labels)
