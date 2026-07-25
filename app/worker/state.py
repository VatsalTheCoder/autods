"""The shared state passed between pipeline nodes, and the node roster.

``PipelineState`` is LangGraph's "shared clipboard" (build-plan Section 4): the
one object handed from each node to the next. In Section 4 it carries almost
nothing -- the nodes are placeholders -- but the shape is established now so that
Section 5 onward only has to add fields (the cleaned frame's key, the
leaderboard, ...) rather than rethink how state flows.

A TypedDict, not a Pydantic model, because that is LangGraph's native state type
and it keeps the reducer annotations (``completed`` accumulates across nodes)
straightforward.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict


class PipelineState(TypedDict, total=False):
    """Everything passed along the graph.

    ``completed`` uses an ``operator.add`` reducer so each node appends its own
    name and the list grows as the run progresses -- the canonical LangGraph
    pattern for accumulating across nodes.
    """

    job_id: int
    # Rows in the dataset, read from S3 by the runner -- proof the worker pulls
    # the file back from object storage rather than a local disk it cannot see.
    n_rows: int
    completed: Annotated[list[str], operator.add]
    # Free-form per-node scratch space; real artifacts land in S3, not here.
    notes: dict[str, Any]


# The pipeline, in order. These names are the Section 5 vertical slice's nodes,
# chosen now so that section swaps sleeping placeholders for real work without
# touching the graph wiring, the agent_runs rows, or the Progress page.
PIPELINE_NODES: list[str] = [
    "planner",
    "cleaning",
    "preprocessing",
    "modeling",
    "evaluation",
    "report",
]
