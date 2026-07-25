"""The shared state passed between pipeline nodes, and the node roster.

``PipelineState`` is LangGraph's "shared clipboard" (build-plan Section 4): the
one object handed from each node to the next. Section 4 established the shape
with almost nothing in it; Section 5 fills it in, which is exactly the change
that section was ordered to make easy.

A TypedDict, not a Pydantic model, because that is LangGraph's native state type
and it keeps the reducer annotations (``completed`` accumulates across nodes)
straightforward. It also means the state can hold live objects -- a DataFrame, an
unfitted ``ColumnTransformer`` -- which a Pydantic model would want to validate
and copy.

**The frames live in memory, not in S3 between nodes.** The graph runs inside a
single worker process, so passing the DataFrame down the clipboard is both
correct and far cheaper than a serialise/upload/download round trip per node. The
*outputs* still go to object storage as artifacts (spec 13) -- that is what makes
them durable and inspectable -- but they are not the transport between steps.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict

import pandas as pd
from sklearn.compose import ColumnTransformer

from app.agents.schema_models import SchemaReport
from app.ml.contracts import (
    CleaningReport,
    ClusteringReport,
    EdaReport,
    EvaluationReport,
    PlannerPlan,
    PreprocessingSpec,
)
from app.ml.modeling import CrossValidationResult


class PipelineState(TypedDict, total=False):
    """Everything passed along the graph.

    ``completed`` uses an ``operator.add`` reducer so each node appends its own
    name and the list grows as the run progresses -- the canonical LangGraph
    pattern for accumulating across nodes. Every other field is written by at
    most one node, so plain overwrite semantics are what is wanted.
    """

    job_id: int
    completed: Annotated[list[str], operator.add]
    # Free-form per-node scratch space; real artifacts land in S3, not here.
    notes: dict[str, Any]

    # ---- Set up by the runner before the graph starts -----------------------
    # The original filename, for the report's title.
    filename: str
    # Read back from S3 by the runner -- proof the worker pulls the file from
    # object storage rather than a local disk it cannot see.
    frame: pd.DataFrame
    n_rows: int
    # The schema report, and the parts of it the user confirmed at the checkpoint.
    schema: SchemaReport
    target: str
    task_type: str
    excluded: list[str]

    # ---- Produced by the nodes ---------------------------------------------
    plan: PlannerPlan
    cleaned: pd.DataFrame
    cleaning_report: CleaningReport

    # Section 6. Descriptive only -- note there is no field for cluster *labels*.
    # They exist inside the EDA node and are never put on the clipboard, because
    # a label on the shared state is one autocomplete away from becoming a model
    # feature, which would leak (spec 9). The reports carry the findings.
    eda_report: EdaReport
    clustering_report: ClusteringReport
    # The unfitted recipe. It travels as an object precisely so that the next
    # node can clone it per fold rather than reach for something already fitted.
    preprocessor: ColumnTransformer
    preprocessing_spec: PreprocessingSpec
    cv_result: CrossValidationResult
    evaluation: EvaluationReport
    report_markdown: str


# The pipeline, in order. Section 4 chose these names for the vertical slice so
# that Section 5 could swap sleeping placeholders for real work without touching
# the graph wiring, the agent_runs rows, or the Progress page -- which is what
# happened.
PIPELINE_NODES: list[str] = [
    "planner",
    "cleaning",
    # Section 6. Sits after cleaning so the charts describe the data the model
    # actually saw, and before preprocessing because it is purely descriptive --
    # nothing downstream depends on its output, which is what lets it fail
    # without taking the model with it.
    "eda",
    "preprocessing",
    "modeling",
    "evaluation",
    "report",
]
