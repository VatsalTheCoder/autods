"""The LangGraph pipeline graph -- placeholder nodes for now.

This is the Section 4 "prove the wiring before filling it" step. Each node just
marks itself running, sleeps, and marks itself done; when the Progress bar ticks
cleanly through all of them, the distributed execution is correct and Section 5
can swap the sleeps for real cleaning, training and reporting without touching
the graph structure.

Every node reports its own status (start → finish) to the database. Keeping that
responsibility in the node -- rather than inferring it from outside -- is the
pattern the real agents will follow too: a node owns its own progress.
"""

from __future__ import annotations

import logging
import time

from langgraph.graph import END, START, StateGraph

from app.core.config import get_settings
from app.worker.progress import fail_node, finish_node, start_node
from app.worker.state import PIPELINE_NODES, PipelineState

logger = logging.getLogger(__name__)


def _make_placeholder(name: str):
    """Build a node that sleeps and reports progress, standing in for real work."""

    def node(state: PipelineState) -> dict:
        job_id = state["job_id"]
        start_node(job_id, name)
        logger.info("[job %s] node %s running", job_id, name)
        try:
            time.sleep(get_settings().pipeline_placeholder_sleep_seconds)
        except Exception as exc:
            # Mark this exact node FAILED, then let it propagate so the runner
            # fails the whole job. This is the pattern real nodes will follow.
            fail_node(job_id, name, str(exc))
            raise
        finish_node(job_id, name)
        return {"completed": [name]}

    node.__name__ = name
    return node


def build_pipeline_graph():
    """Compile the linear placeholder pipeline: START → ... nodes ... → END."""
    graph = StateGraph(PipelineState)

    for name in PIPELINE_NODES:
        graph.add_node(name, _make_placeholder(name))

    graph.add_edge(START, PIPELINE_NODES[0])
    for earlier, later in zip(PIPELINE_NODES, PIPELINE_NODES[1:], strict=False):
        graph.add_edge(earlier, later)
    graph.add_edge(PIPELINE_NODES[-1], END)

    return graph.compile()
