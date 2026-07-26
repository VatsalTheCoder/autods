"""The LangGraph pipeline -- real nodes (Section 5, spec M1).

Section 4 proved this wiring with nodes that only slept. Section 5 replaced the
sleeps with the actual work and changed nothing about the structure: the same six
nodes in the same order, each still reporting its own status to the database so
the Progress page needs no knowledge of what any node does.

Each node here is a thin adapter, and deliberately so. The thinking lives in
``app/ml`` and ``app/agents`` as pure functions over DataFrames -- which is what
lets the whole pipeline be tested without a graph, a worker, or a broker. A node's
only jobs are to pull its inputs off the shared state, call one of those
functions, persist whatever artifacts it produced, and put its outputs back.

Artifacts are committed per node rather than once at the end. A job that fails at
modelling should still leave its cleaning report behind for someone to look at --
which is the whole reason the run is inspectable after a failure instead of just
being marked red.
"""

from __future__ import annotations

import io
import logging
from collections.abc import Callable

import joblib
from langgraph.graph import END, START, StateGraph

from app.agents.cluster_profiles import AGENT_NAME as CLUSTER_PROFILE_AGENT
from app.agents.cluster_profiles import describe_clusters
from app.agents.planner import AGENT_NAME as PLANNER_AGENT
from app.agents.planner import make_plan
from app.core.db import SessionLocal
from app.core.llm.factory import get_optional_llm
from app.core.llm.usage import make_usage_recorder
from app.ml.cleaning import clean_frame
from app.ml.clustering import run_clustering
from app.ml.evaluation import build_evaluation_report
from app.ml.modeling import cross_validate_model
from app.ml.plots import render_charts
from app.ml.preprocessing import build_preprocessor
from app.ml.report import build_markdown_report
from app.ml.statistics import compute_statistics
from app.models.artifact import ArtifactKind
from app.services.artifacts import (
    CLEANED_DATASET_ARTIFACT,
    CLEANING_ARTIFACT,
    CLUSTERING_ARTIFACT,
    EDA_ARTIFACT,
    EVALUATION_ARTIFACT,
    PLANNER_ARTIFACT,
    PREPROCESSING_ARTIFACT,
    PREPROCESSOR_ARTIFACT,
    REPORT_ARTIFACT,
    register_bytes_artifact,
    register_json_artifact,
)
from app.worker.progress import fail_node, finish_node, start_node
from app.worker.state import PIPELINE_NODES, PipelineState

logger = logging.getLogger(__name__)


# ---- The nodes --------------------------------------------------------------


def planner_node(state: PipelineState) -> dict:
    """Ask the LLM for a small plan the later nodes obey (spec 7.3)."""
    job_id = state["job_id"]
    with SessionLocal() as db:
        # The usage recorder and the artifact share one transaction, so a job's
        # recorded token spend can never outlive the plan it paid for.
        plan = make_plan(
            state["schema"],
            client=get_optional_llm(),
            on_usage=make_usage_recorder(db, job_id, PLANNER_AGENT),
        )
        register_json_artifact(db, job_id, PLANNER_ARTIFACT, plan)
        db.commit()
    return {"plan": plan}


def cleaning_node(state: PipelineState) -> dict:
    """Structural cleaning. Note what it does *not* do -- see ``ml/cleaning.py``."""
    job_id = state["job_id"]
    result = clean_frame(
        state["frame"],
        state["schema"],
        target=state["target"],
        task_type=state["task_type"],
        excluded=state.get("excluded", []),
        plan=state["plan"],
    )

    with SessionLocal() as db:
        register_json_artifact(db, job_id, CLEANING_ARTIFACT, result.report)
        # The cleaned dataset is stored so a reader can check the report's claims
        # against the actual data, and so later sections have a defined starting
        # point that is not "re-run cleaning and hope it matches".
        register_bytes_artifact(
            db,
            job_id,
            CLEANED_DATASET_ARTIFACT,
            result.frame.to_csv(index=False).encode("utf-8"),
            content_type="text/csv",
            kind=ArtifactKind.CLEANED_DATASET,
        )
        db.commit()

    return {"cleaned": result.frame, "cleaning_report": result.report}


def eda_node(state: PipelineState) -> dict:
    """Describe the data: statistics, charts, and natural groupings (spec 7.5, 9).

    Purely descriptive. Nothing downstream reads its output, which is deliberate
    -- it means a dataset that defeats the charts or the clustering still gets
    modelled, and it is why this node can be this forgiving.

    **The guardrail is enforced here, not just documented.** Cluster labels are
    computed over every row including the ones that later land in a test fold, so
    using them as a feature would leak (spec 9). The labels never go on the shared
    state, and this node asserts that the frame it returns is column-for-column
    the one it received -- a check that costs nothing and would catch the mistake
    the moment somebody made it.
    """
    job_id = state["job_id"]
    frame = state["cleaned"]
    columns_before = list(frame.columns)

    eda = compute_statistics(frame, target=state["target"], task_type=state["task_type"])
    charts = render_charts(frame, target=state["target"], task_type=state["task_type"])

    clustering = run_clustering(frame, target=state["target"], plan=state["plan"])
    if clustering.scatter is not None:
        charts = [*charts, clustering.scatter]

    with SessionLocal() as db:
        clustering.report.profiles = describe_clusters(
            clustering.report.profiles,
            client=get_optional_llm(),
            on_usage=make_usage_recorder(db, job_id, CLUSTER_PROFILE_AGENT),
        )
        for chart in charts:
            register_bytes_artifact(
                db,
                job_id,
                chart.name,
                chart.png,
                content_type="image/png",
                kind=ArtifactKind.PLOT,
            )
        eda.plots = [chart.name for chart in charts]
        register_json_artifact(db, job_id, EDA_ARTIFACT, eda)
        register_json_artifact(db, job_id, CLUSTERING_ARTIFACT, clustering.report)
        db.commit()

    if list(frame.columns) != columns_before:
        raise RuntimeError(
            "EDA modified the dataset's columns. Cluster labels and other "
            "descriptive output must never become model features (spec 9)."
        )

    return {"eda_report": eda, "clustering_report": clustering.report}


def preprocessing_node(state: PipelineState) -> dict:
    """Build the **unfitted** recipe and store it without ever fitting it (spec 7.6).

    The pickle written here is the evidence for the project's central claim: load
    it back and it is still unfitted, because nothing between this node and the
    cross-validation loop fits it. The test suite asserts exactly that against
    this object.
    """
    job_id = state["job_id"]
    result = build_preprocessor(state["cleaned"], target=state["target"])

    buffer = io.BytesIO()
    joblib.dump(result.transformer, buffer)

    with SessionLocal() as db:
        register_json_artifact(db, job_id, PREPROCESSING_ARTIFACT, result.spec)
        register_bytes_artifact(
            db,
            job_id,
            PREPROCESSOR_ARTIFACT,
            buffer.getvalue(),
            content_type="application/octet-stream",
            # Not a trained model -- it is the recipe a model will be trained
            # behind. MODEL is the closest kind, and keeping the pickle in the
            # registry is what makes the unfitted claim externally checkable.
            kind=ArtifactKind.MODEL,
        )
        db.commit()

    return {"preprocessor": result.transformer, "preprocessing_spec": result.spec}


def modeling_node(state: PipelineState) -> dict:
    """Cross-validate one model, fitting the recipe only inside each fold (spec 7.7).

    Writes no artifact of its own: in Section 5 there is a single model, so the
    scores are the evaluation report's story to tell. Section 7 adds
    ``model_candidates.json`` and a leaderboard here, when there is a comparison
    to record.
    """
    return {
        "cv_result": cross_validate_model(
            state["cleaned"],
            target=state["target"],
            task_type=state["task_type"],
            preprocessor=state["preprocessor"],
        )
    }


def evaluation_node(state: PipelineState) -> dict:
    """Aggregate the folds into ``evaluation_report.json`` (spec 7.8)."""
    job_id = state["job_id"]
    cv = state["cv_result"]
    report = build_evaluation_report(
        cv.folds,
        task_type=state["task_type"],
        target_column=state["target"],
        model_name=cv.model_name,
        n_folds=cv.n_folds,
        cv_strategy=cv.cv_strategy,
        n_rows=cv.n_rows,
        n_features=cv.n_features,
        warnings=cv.warnings,
    )
    with SessionLocal() as db:
        register_json_artifact(db, job_id, EVALUATION_ARTIFACT, report)
        db.commit()
    return {"evaluation": report}


def report_node(state: PipelineState) -> dict:
    """Write the Markdown report -- the thing a human actually reads."""
    job_id = state["job_id"]
    markdown = build_markdown_report(
        filename=state.get("filename", "dataset.csv"),
        plan=state["plan"],
        cleaning=state["cleaning_report"],
        preprocessing=state["preprocessing_spec"],
        evaluation=state["evaluation"],
        # Optional: a run whose EDA stage failed still gets a report about the
        # model, which is the part a reader cannot do without.
        eda=state.get("eda_report"),
        clustering=state.get("clustering_report"),
    )
    with SessionLocal() as db:
        register_bytes_artifact(
            db,
            job_id,
            REPORT_ARTIFACT,
            markdown.encode("utf-8"),
            content_type="text/markdown",
            kind=ArtifactKind.REPORT,
        )
        db.commit()
    return {"report_markdown": markdown}


NODE_FUNCTIONS: dict[str, Callable[[PipelineState], dict]] = {
    "planner": planner_node,
    "cleaning": cleaning_node,
    "eda": eda_node,
    "preprocessing": preprocessing_node,
    "modeling": modeling_node,
    "evaluation": evaluation_node,
    "report": report_node,
}


# ---- Wiring -----------------------------------------------------------------


def _tracked(name: str, work: Callable[[PipelineState], dict]):
    """Wrap a node so it reports its own start, finish and failure.

    Keeping this responsibility in the node -- rather than inferring it from
    outside -- means a node that fails marks *itself* failed before the exception
    propagates, so the UI can point at the exact step rather than only telling
    the user that something, somewhere, went wrong (spec 10).
    """

    def node(state: PipelineState) -> dict:
        job_id = state["job_id"]
        start_node(job_id, name)
        logger.info("[job %s] node %s running", job_id, name)
        try:
            update = work(state)
        except Exception as exc:
            fail_node(job_id, name, str(exc))
            raise
        finish_node(job_id, name)
        return {**update, "completed": [name]}

    node.__name__ = name
    return node


def build_pipeline_graph():
    """Compile the linear pipeline: START → planner → ... → report → END.

    Still linear in Section 5. The spec's conditional edges (7.3, 11) arrive with
    Section 7, when the planner has optional steps worth branching around.
    """
    graph = StateGraph(PipelineState)

    for name in PIPELINE_NODES:
        graph.add_node(name, _tracked(name, NODE_FUNCTIONS[name]))

    graph.add_edge(START, PIPELINE_NODES[0])
    for earlier, later in zip(PIPELINE_NODES, PIPELINE_NODES[1:], strict=False):
        graph.add_edge(earlier, later)
    graph.add_edge(PIPELINE_NODES[-1], END)

    return graph.compile()
