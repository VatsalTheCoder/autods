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
    # The vertical-slice order, implemented for real in Section 5.
    assert PIPELINE_NODES == [
        "planner",
        "cleaning",
        "preprocessing",
        "modeling",
        "evaluation",
        "report",
    ]


def test_every_node_has_a_real_implementation():
    """Section 4's placeholders are gone; each node maps to actual work."""
    assert set(NODE_FUNCTIONS) == set(PIPELINE_NODES)
    assert all(callable(fn) for fn in NODE_FUNCTIONS.values())


def test_graph_compiles_to_something_invocable():
    compiled = build_pipeline_graph()
    assert hasattr(compiled, "invoke")
