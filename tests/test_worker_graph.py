"""Tests for the pipeline graph structure.

Pure -- builds and inspects the compiled graph without invoking it, so no
database or worker is needed. Invocation (which writes status and artifacts) is
covered by the integration tests in test_pipeline.py.
"""

from __future__ import annotations

from app.worker.graph import NODE_FUNCTIONS, build_pipeline_graph
from app.worker.state import PIPELINE_NODES


def test_graph_contains_every_pipeline_node():
    graph = build_pipeline_graph().get_graph()
    node_names = set(graph.nodes.keys())
    for name in PIPELINE_NODES:
        assert name in node_names


def test_pipeline_is_the_expected_sequence():
    # Section 5's vertical slice, with Section 6's descriptive stage inserted
    # after cleaning -- so the charts describe the data the model actually saw --
    # and Section 7's strategy step before preprocessing, because choosing how
    # each column should be prepared is a separate act from building the recipe.
    # Section 8 adds the refit and its explanation between evaluation and the
    # report, so the report is written with both of them available to describe.
    # Section 9 puts the critic immediately before the report, so the report can
    # reflect the review rather than append it. Section 10 adds indexing at the
    # very end, over the report the previous node has just written.
    assert PIPELINE_NODES == [
        "planner",
        "cleaning",
        "eda",
        "sampling",
        "feature_strategy",
        "feature_selection",
        "preprocessing",
        "modeling",
        "evaluation",
        "final_training",
        "explainability",
        "critic",
        "report",
        "chat_index",
    ]


def test_the_model_is_refitted_after_it_has_been_scored():
    """Order matters methodologically, not just mechanically (spec 7.9).

    Fitting on the full dataset is only defensible once cross-validation has
    produced the honest number. If final training ran first, the temptation --
    and the opportunity -- to report a score from it would exist.
    """
    assert PIPELINE_NODES.index("evaluation") < PIPELINE_NODES.index("final_training")
    assert PIPELINE_NODES.index("final_training") < PIPELINE_NODES.index("explainability")


def test_eda_runs_after_cleaning_and_before_modelling():
    """Its inputs come from cleaning; nothing downstream depends on its output."""
    assert PIPELINE_NODES.index("cleaning") < PIPELINE_NODES.index("eda")
    assert PIPELINE_NODES.index("eda") < PIPELINE_NODES.index("modeling")


def test_every_node_has_a_real_implementation():
    """Section 4's placeholders are gone; each node maps to actual work."""
    assert set(NODE_FUNCTIONS) == set(PIPELINE_NODES)
    assert all(callable(fn) for fn in NODE_FUNCTIONS.values())


def test_graph_compiles_to_something_invocable():
    compiled = build_pipeline_graph()
    assert hasattr(compiled, "invoke")


def test_the_report_is_written_after_the_review_of_it():
    """Order is the design, not an accident (spec 7.11, 7.12).

    A report written before its own critique could only bolt the findings on at
    the end. The executive summary is exactly where a blocker needs to appear,
    and that is only possible if the critic has already run.
    """
    assert PIPELINE_NODES.index("critic") < PIPELINE_NODES.index("report")
    assert PIPELINE_NODES.index("explainability") < PIPELINE_NODES.index("critic")


def test_indexing_happens_after_there_is_something_to_index():
    """Section 10's node reads the run's written output, so it goes last."""
    assert PIPELINE_NODES[-1] == "chat_index"
    assert PIPELINE_NODES.index("report") < PIPELINE_NODES.index("chat_index")
