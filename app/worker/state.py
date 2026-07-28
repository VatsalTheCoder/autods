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
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.compose import ColumnTransformer

from app.agents.schema_models import SchemaReport
from app.ml.contracts import (
    CleaningReport,
    ClusteringReport,
    CriticReport,
    EdaReport,
    EvaluationReport,
    ExplainabilityReport,
    FeatureStrategy,
    FinalModelInfo,
    Leaderboard,
    NarrativeReport,
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
    # Section 7. The per-column decisions, chosen before anything is built --
    # ``preprocessing`` is what turns them into a transformer.
    feature_strategy: FeatureStrategy
    # How many features to keep, or None for all of them. Set by the planner's
    # conditional branch, read by the recipe builder, which puts the selector
    # *inside* the pipeline so it is fitted per fold.
    select_k: int | None
    # What the sampling step did, for the report. Empty when it did not run.
    sampling_note: str

    # The unfitted recipe. It travels as an object precisely so that the next
    # node can clone it per fold rather than reach for something already fitted.
    preprocessor: ColumnTransformer
    preprocessing_spec: PreprocessingSpec
    cv_result: CrossValidationResult
    # Section 7. The whole roster, ranked; ``cv_result`` is the winner's folds.
    leaderboard: Leaderboard
    evaluation: EvaluationReport

    # Section 8. The winner refitted on every row -- the fitted object, because
    # the explainability node has to explain the model that will actually be
    # served rather than an equivalent one it refitted for itself.
    final_model: ImbPipeline
    final_model_info: FinalModelInfo
    explainability: ExplainabilityReport

    # Section 9. The review, and the report's prose. Kept apart from
    # ``report_markdown`` because they are the model's contribution and that is
    # the half a reader may want to weigh differently from the tables.
    critic: CriticReport
    narrative: NarrativeReport
    # The shrunk view both Section 9 agents read, built once by the critic node
    # and passed on -- summarising twice would double the token spend on the
    # limit that binds hardest, and risk two different views of one run.
    run_summary: object

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
    # Section 7, optional (see graph.OPTIONAL_NODES). After EDA so the charts
    # always describe every uploaded row, and only the modelling sees a subset.
    "sampling",
    # Section 7. Separate from ``preprocessing`` on purpose: this node decides,
    # that one builds. Two nodes means the Progress page shows the LLM's choice
    # landing as its own step, and means a strategy artifact exists to read even
    # if building the recipe from it later fails.
    "feature_strategy",
    # Section 7, optional. Decides *how many* features to keep; the selector
    # itself is a step in the recipe below, so it is fitted per fold.
    "feature_selection",
    "preprocessing",
    "modeling",
    "evaluation",
    # Section 8. Refitting the winner on every row, then explaining it. Two nodes
    # for the same reason the feature strategy is separate from preprocessing:
    # they can fail independently and for different reasons. A model that trains
    # but cannot be explained should still be saved and served, and the Progress
    # page should say which of the two happened.
    "final_training",
    "explainability",
    # Section 9. The critic reads everything above it, so it goes last but one;
    # the report then has the review available to reflect rather than to append.
    # A report written before its own critique would be the wrong way round --
    # the executive summary is exactly where a concern needs to appear.
    "critic",
    "report",
]
