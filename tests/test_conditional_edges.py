"""Tests for the conditional edges -- the "dynamic orchestration" claim (spec 11).

The build plan's Done-when for this section is that *two datasets with different
shapes visibly take different routes through the graph*. That word "visibly" is
what these tests are about. A pipeline that quietly does less work on a small
dataset is just an optimisation; a pipeline that records which steps it decided
not to run is making a claim a reader can check.

So there are two things to prove, and they are different:

1. **The route changes.** A plan with a flag on enters the node; a plan with it
   off does not, and the node's work never happens.
2. **The skip is legible.** The node keeps its row in the roadmap, is marked
   SKIPPED rather than left PENDING, and says why. Without that, "we did not
   need feature selection" and "we have not reached feature selection" are the
   same picture.

Run against the compiled graph with the node bodies stubbed, so this is a test
of the *wiring* rather than of what the nodes do.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import app.worker.graph as graph
from app.ml.contracts import PlannerPlan
from app.ml.sampling import sample_frame
from app.worker.state import PIPELINE_NODES


@pytest.fixture
def visited(monkeypatch) -> list[str]:
    """Replace every node with a recorder, so a run yields the route it took."""
    seen: list[str] = []

    def recorder(name):
        def node(state):
            seen.append(name)
            return {}

        return node

    monkeypatch.setattr(graph, "NODE_FUNCTIONS", {name: recorder(name) for name in PIPELINE_NODES})
    monkeypatch.setattr(graph, "start_node", lambda *a, **k: None)
    monkeypatch.setattr(graph, "finish_node", lambda *a, **k: None)
    return seen


@pytest.fixture
def skipped(monkeypatch) -> list[tuple[str, str]]:
    """Capture what the router marked SKIPPED, and the reason it gave."""
    marks: list[tuple[str, str]] = []
    monkeypatch.setattr(graph, "skip_node", lambda _job, name, reason: marks.append((name, reason)))
    return marks


def _run(plan: PlannerPlan) -> None:
    graph.build_pipeline_graph().invoke({"job_id": 1, "plan": plan, "completed": [], "notes": {}})


class TestTheRouteChanges:
    def test_a_plan_with_nothing_optional_skips_both_steps(self, visited, skipped):
        _run(PlannerPlan())

        assert "sampling" not in visited
        assert "feature_selection" not in visited
        assert {name for name, _ in skipped} == {"sampling", "feature_selection"}

    def test_a_plan_asking_for_sampling_enters_that_node(self, visited, skipped):
        _run(PlannerPlan(run_sampling=True))

        assert "sampling" in visited
        assert [name for name, _ in skipped] == ["feature_selection"]

    def test_a_plan_asking_for_selection_enters_that_node(self, visited, skipped):
        _run(PlannerPlan(run_feature_selection=True))

        assert "feature_selection" in visited
        assert [name for name, _ in skipped] == ["sampling"]

    def test_a_plan_asking_for_both_runs_the_whole_graph(self, visited, skipped):
        _run(PlannerPlan(run_sampling=True, run_feature_selection=True))

        assert visited == PIPELINE_NODES
        assert skipped == []

    def test_two_different_plans_take_different_routes(self, visited, skipped):
        """The build plan's Done-when, stated as one assertion.

        The same graph, two plans, two different sequences of nodes -- which is
        what "the pipeline adapts to the dataset" has to mean if it means
        anything checkable.
        """
        _run(PlannerPlan())
        lean = list(visited)

        visited.clear()
        _run(PlannerPlan(run_sampling=True, run_feature_selection=True))
        full = list(visited)

        assert lean != full
        assert len(lean) == len(PIPELINE_NODES) - 2
        assert full == PIPELINE_NODES

    def test_the_required_steps_run_whatever_the_plan_says(self, visited, skipped):
        """Only genuinely optional steps are optional -- cleaning is not."""
        _run(PlannerPlan())

        for name in ("planner", "cleaning", "preprocessing", "modeling", "evaluation", "report"):
            assert name in visited


class TestTheSkipIsLegible:
    def test_a_skipped_node_is_given_a_reason(self, visited, skipped):
        _run(PlannerPlan())
        assert all(reason for _, reason in skipped)

    def test_the_reason_names_the_planner_as_the_decider(self, visited, skipped):
        _run(PlannerPlan())
        assert all("planner" in reason.lower() for _, reason in skipped)

    def test_a_node_that_ran_is_not_also_marked_skipped(self, visited, skipped):
        _run(PlannerPlan(run_sampling=True))
        assert "sampling" not in [name for name, _ in skipped]

    def test_the_roadmap_still_lists_every_node(self, visited, skipped):
        """A skipped node keeps its row; it is not removed from the pipeline."""
        _run(PlannerPlan())
        assert set(graph.NODE_FUNCTIONS) == set(PIPELINE_NODES)


class TestGraphShape:
    def test_only_genuinely_optional_steps_are_optional(self):
        assert set(graph.OPTIONAL_NODES) == {"sampling", "feature_selection"}

    def test_every_optional_node_has_a_skip_reason(self):
        assert set(graph.OPTIONAL_NODES) <= set(graph._SKIP_REASONS)

    def test_adjacent_optional_nodes_are_refused_rather_than_mis_wired(self, monkeypatch):
        """The router can only skip one node at a time, so say so loudly.

        Two adjacent optional nodes would leave the second unreachable when the
        first was skipped -- a silent hole in the graph rather than an error.
        """
        monkeypatch.setattr(
            graph,
            "OPTIONAL_NODES",
            {"sampling": lambda p: True, "feature_strategy": lambda p: True},
        )
        with pytest.raises(RuntimeError, match="adjacent"):
            graph.build_pipeline_graph()


class TestSampling:
    """The one reduction that is safe outside a fold -- because it learns nothing."""

    @pytest.fixture
    def big(self) -> pd.DataFrame:
        rng = np.random.default_rng(0)
        return pd.DataFrame(
            {
                "x1": rng.normal(0, 1, 1000),
                # 95:5, so a uniform sample could plausibly lose the rare class.
                "churn": (["no"] * 950) + (["yes"] * 50),
            }
        )

    def test_a_small_frame_is_returned_untouched(self, big):
        assert sample_frame(big, target="churn", task_type="classification", limit=5000) is big

    def test_it_cuts_the_frame_down_to_the_limit(self, big):
        sampled = sample_frame(big, target="churn", task_type="classification", limit=200)
        assert 200 <= len(sampled) <= 210

    def test_the_rare_class_survives(self, big):
        """A uniform sample of a 95:5 dataset can contain no minority rows at all."""
        sampled = sample_frame(big, target="churn", task_type="classification", limit=100)
        assert set(sampled["churn"]) == {"no", "yes"}

    def test_the_class_proportions_are_kept(self, big):
        sampled = sample_frame(big, target="churn", task_type="classification", limit=200)
        share = (sampled["churn"] == "yes").mean()
        assert share == pytest.approx(0.05, abs=0.02)

    def test_a_regression_target_is_sampled_uniformly(self):
        frame = pd.DataFrame({"x1": np.arange(1000.0), "price": np.arange(1000.0)})
        sampled = sample_frame(frame, target="price", task_type="regression", limit=100)
        assert len(sampled) == 100

    def test_it_is_reproducible(self, big):
        first = sample_frame(big, target="churn", task_type="classification", limit=200)
        second = sample_frame(big, target="churn", task_type="classification", limit=200)
        pd.testing.assert_frame_equal(first, second)

    def test_it_takes_rows_away_and_nothing_else(self, big):
        """No column is added, renamed or altered -- this is not a transformation."""
        sampled = sample_frame(big, target="churn", task_type="classification", limit=200)
        assert list(sampled.columns) == list(big.columns)
        pd.testing.assert_frame_equal(sampled, big.loc[sampled.index])
