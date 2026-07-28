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
from app.agents.critic import AGENT_NAME as CRITIC_AGENT
from app.agents.critic import review_run
from app.agents.feature_strategy import AGENT_NAME as FEATURE_STRATEGY_AGENT
from app.agents.feature_strategy import make_strategy
from app.agents.planner import AGENT_NAME as PLANNER_AGENT
from app.agents.planner import make_plan
from app.agents.report_writer import AGENT_NAME as REPORT_WRITER_AGENT
from app.agents.report_writer import write_narrative
from app.agents.summaries import summarise_run
from app.core.config import get_settings
from app.core.db import SessionLocal
from app.core.llm.factory import get_optional_llm
from app.core.llm.usage import make_usage_recorder
from app.ml.chunking import build_chunks
from app.ml.cleaning import clean_frame
from app.ml.clustering import run_clustering
from app.ml.contracts import ExplainabilityReport, PlannerPlan
from app.ml.evaluation import build_evaluation_report
from app.ml.explain import ExplainabilityError, ExplainabilityResult, explain_model
from app.ml.final_training import train_final_model
from app.ml.modeling import run_leaderboard
from app.ml.pdf import PdfError, render_pdf
from app.ml.plots import render_charts
from app.ml.preprocessing import build_preprocessor
from app.ml.report import build_markdown_report
from app.ml.sampling import sample_frame
from app.ml.statistics import compute_statistics
from app.models.artifact import ArtifactKind
from app.services.artifacts import (
    CLEANED_DATASET_ARTIFACT,
    CLEANING_ARTIFACT,
    CLUSTERING_ARTIFACT,
    CRITIC_ARTIFACT,
    EDA_ARTIFACT,
    EVALUATION_ARTIFACT,
    EXPLAINABILITY_ARTIFACT,
    FEATURE_ARTIFACT,
    FINAL_MODEL_ARTIFACT,
    FINAL_MODEL_INFO_ARTIFACT,
    LEADERBOARD_ARTIFACT,
    NARRATIVE_ARTIFACT,
    PLANNER_ARTIFACT,
    PREPROCESSING_ARTIFACT,
    PREPROCESSOR_ARTIFACT,
    REPORT_ARTIFACT,
    REPORT_PDF_ARTIFACT,
    register_bytes_artifact,
    register_json_artifact,
)
from app.services.retrieval import index_run
from app.worker.progress import fail_node, finish_node, skip_node, start_node
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


def sampling_node(state: PipelineState) -> dict:
    """Train on a random subset, because the dataset is big enough to warrant it.

    Conditional -- the planner decides whether this runs at all, and on most
    datasets it does not. Placed after EDA on purpose: the charts and statistics
    describe every row that was uploaded, and only the modelling is done on a
    sample, so nothing in the descriptive half is quietly computed from a subset.

    Sampling is one of the few things here that can safely happen outside a fold.
    It removes rows without learning anything from them -- no statistic crosses
    from the discarded rows into the kept ones -- so the fold discipline is
    untouched. Stratified for classification so the sample keeps the target's
    class proportions; a random subset of a 99:1 dataset can easily contain no
    minority rows at all.
    """
    frame = state["cleaned"]
    limit = get_settings().max_modelling_rows
    if len(frame) <= limit:
        return {
            "sampling_note": (
                f"Sampling was planned but the dataset has {len(frame):,} rows, "
                f"under the {limit:,}-row threshold, so every row was used."
            )
        }

    sampled = sample_frame(frame, target=state["target"], task_type=state["task_type"], limit=limit)
    note = (
        f"Trained on a random sample of {len(sampled):,} rows drawn from "
        f"{len(frame):,}, to keep training time reasonable. Charts and statistics "
        "above describe the full dataset."
    )
    logger.info("[job %s] sampled %d rows from %d", state["job_id"], len(sampled), len(frame))
    return {"cleaned": sampled, "sampling_note": note}


def feature_selection_node(state: PipelineState) -> dict:
    """Decide how many features to keep -- the recipe does the keeping (spec 7.6).

    Conditional, and deliberately thin: all it does is put a number on the shared
    state. The selector itself becomes a *step inside the unfitted pipeline*
    (``preprocessing.py``), which is what makes it fitted per fold like every
    other fitted thing. Ranking features against the target over the whole
    dataset would use the held-out rows' targets to decide what the model is
    allowed to see -- a leak that looks like good feature engineering.

    So this node chooses a policy and preprocessing enacts it. The folds may
    legitimately end up selecting different columns from one another, which is
    what an honest per-fold selection looks like.
    """
    n_features = max(1, len([c for c in state["cleaned"].columns if c != state["target"]]))
    select_k = min(get_settings().feature_selection_k, n_features)
    logger.info("[job %s] feature selection will keep the top %d", state["job_id"], select_k)
    return {"select_k": select_k}


def feature_strategy_node(state: PipelineState) -> dict:
    """Ask the LLM how each column should be prepared (spec 7.6).

    Decision only -- nothing is built here and no data is touched. Splitting the
    choosing from the building is the point: this node's output is a JSON table
    that can be read, checked and disagreed with before any pipeline exists.
    """
    job_id = state["job_id"]
    with SessionLocal() as db:
        strategy = make_strategy(
            state["cleaned"],
            target=state["target"],
            report=state.get("schema"),
            client=get_optional_llm(),
            on_usage=make_usage_recorder(db, job_id, FEATURE_STRATEGY_AGENT),
        )
        register_json_artifact(db, job_id, FEATURE_ARTIFACT, strategy)
        db.commit()
    return {"feature_strategy": strategy}


def preprocessing_node(state: PipelineState) -> dict:
    """Build the **unfitted** recipe and store it without ever fitting it (spec 7.6).

    The pickle written here is the evidence for the project's central claim: load
    it back and it is still unfitted, because nothing between this node and the
    cross-validation loop fits it. The test suite asserts exactly that against
    this object.
    """
    job_id = state["job_id"]
    result = build_preprocessor(
        state["cleaned"],
        target=state["target"],
        strategy=state.get("feature_strategy"),
        # Feature selection is a *step in the recipe*, not a pass over the data,
        # so it is fitted per fold with everything else. The planner decides
        # whether it happens at all; see ``select_k_for``.
        select_k=state.get("select_k"),
        task_type=state["task_type"],
    )

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
    """Cross-validate the roster and rank it, fitting only inside folds (spec 7.7).

    Section 5 trained one model and left the scores to the evaluation report.
    There are four now, so there is a comparison worth recording: the leaderboard
    is written here, and the winner is passed on for the evaluation report to
    describe in full.
    """
    job_id = state["job_id"]
    plan = state["plan"]
    leaderboard, winner = run_leaderboard(
        state["cleaned"],
        target=state["target"],
        task_type=state["task_type"],
        preprocessor=state["preprocessor"],
        use_smote=bool(plan and plan.use_smote),
    )
    with SessionLocal() as db:
        register_json_artifact(db, job_id, LEADERBOARD_ARTIFACT, leaderboard)
        db.commit()
    return {"cv_result": winner, "leaderboard": leaderboard}


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


def final_training_node(state: PipelineState) -> dict:
    """Refit the winner on every row and save it for serving (spec 7.9).

    The one node in the pipeline that fits on the full dataset -- correct here
    and nowhere else, for the reason set out at length in ``ml/final_training.py``.
    The score is not recomputed: what goes in the artifact is the cross-validated
    figure, carried over and named so it cannot be mistaken for a measurement of
    this model.
    """
    job_id = state["job_id"]
    plan = state.get("plan")
    leaderboard = state.get("leaderboard")
    winner = leaderboard.winner() if leaderboard is not None else None

    final = train_final_model(
        state["cleaned"],
        target=state["target"],
        task_type=state["task_type"],
        preprocessor=state["preprocessor"],
        model_name=state["cv_result"].model_name,
        use_smote=bool(plan and plan.use_smote),
        strategy=state.get("feature_strategy"),
        primary_metric=leaderboard.primary_metric if leaderboard is not None else "",
        cv_score=winner.score if winner is not None else None,
    )
    final.info.artifact = FINAL_MODEL_ARTIFACT

    buffer = io.BytesIO()
    joblib.dump(final.pipeline, buffer)

    with SessionLocal() as db:
        register_bytes_artifact(
            db,
            job_id,
            FINAL_MODEL_ARTIFACT,
            buffer.getvalue(),
            content_type="application/octet-stream",
            kind=ArtifactKind.MODEL,
        )
        register_json_artifact(db, job_id, FINAL_MODEL_INFO_ARTIFACT, final.info)
        db.commit()

    return {"final_model": final.pipeline, "final_model_info": final.info}


def explainability_node(state: PipelineState) -> dict:
    """SHAP over the saved model, in the user's column names (spec 7.10).

    Explains ``final_model`` rather than refitting anything, so the account is of
    the object ``POST /jobs/{id}/predict`` will actually load.

    A model family with no explainer is recorded, not raised. The pipeline's
    headline claim is explainability, which is precisely why an unexplainable
    model must produce a report that *says so* -- the alternative is a failed job
    that throws away a perfectly good trained model and its evaluation, and
    leaves the user to guess why.
    """
    job_id = state["job_id"]
    info = state["final_model_info"]

    try:
        result = explain_model(
            state["final_model"],
            state["cleaned"],
            target=state["target"],
            task_type=state["task_type"],
            model_name=info.model_name,
        )
    except ExplainabilityError as exc:
        logger.warning("[job %s] no SHAP explanation: %s", job_id, exc)
        result = ExplainabilityResult(
            report=ExplainabilityReport(
                model_name=info.model_name,
                task_type=state["task_type"],
                target_column=state["target"],
                warnings=[str(exc)],
            )
        )

    with SessionLocal() as db:
        for chart in result.charts:
            register_bytes_artifact(
                db,
                job_id,
                chart.name,
                chart.png,
                content_type="image/png",
                kind=ArtifactKind.PLOT,
            )
        register_json_artifact(db, job_id, EXPLAINABILITY_ARTIFACT, result.report)
        db.commit()

    return {"explainability": result.report}


def _run_summary(state: PipelineState):
    """The shrunk view of the run that both Section 9 agents read (spec 6.3).

    Built once and shared. Doing it twice would double the token spend on a
    limit that is already the binding constraint, and would risk the critic and
    the report reasoning about subtly different views of the same run.
    """
    return summarise_run(
        budget_tokens=get_settings().llm_prompt_budget_tokens,
        filename=state.get("filename", ""),
        cleaning=state.get("cleaning_report"),
        eda=state.get("eda_report"),
        clustering=state.get("clustering_report"),
        features=state.get("feature_strategy"),
        preprocessing=state.get("preprocessing_spec"),
        leaderboard=state.get("leaderboard"),
        evaluation=state.get("evaluation"),
        explainability=state.get("explainability"),
        final_model=state.get("final_model_info"),
    )


def critic_node(state: PipelineState) -> dict:
    """Review the whole run and record what is wrong with it (spec 7.11).

    Runs after everything it reviews, and before the report that has to reflect
    it. The measured findings are computed from the *artifacts* rather than the
    summary, so a threshold check sees the real numbers even on a wide dataset
    where the model cannot -- see ``agents/critic.py``.
    """
    job_id = state["job_id"]
    summary = _run_summary(state)

    with SessionLocal() as db:
        review = review_run(
            summary,
            leaderboard=state.get("leaderboard"),
            evaluation=state.get("evaluation"),
            clustering=state.get("clustering_report"),
            explainability=state.get("explainability"),
            features=state.get("feature_strategy"),
            client=get_optional_llm(),
            on_usage=make_usage_recorder(db, job_id, CRITIC_AGENT),
        )
        register_json_artifact(db, job_id, CRITIC_ARTIFACT, review)
        db.commit()

    return {"critic": review, "run_summary": summary}


def report_node(state: PipelineState) -> dict:
    """Write the report: the model's prose around the code's numbers (spec 7.12).

    Three artifacts, deliberately. ``narrative_report.json`` is what the model
    wrote and nothing else, so a reader can see which sentences came from a model
    without diffing two documents; ``report.md`` is the assembled document; and
    ``report.pdf`` is the same document rendered.

    The PDF is best-effort. A rendering failure at the very end of a pipeline run
    must not discard the Markdown that is already correct -- and the failure is
    recorded rather than swallowed, because a missing PDF with no explanation
    looks like the pipeline stopping early.
    """
    job_id = state["job_id"]
    summary = state.get("run_summary") or _run_summary(state)

    with SessionLocal() as db:
        narrative = write_narrative(
            summary,
            critic=state.get("critic"),
            client=get_optional_llm(),
            on_usage=make_usage_recorder(db, job_id, REPORT_WRITER_AGENT),
        )
        register_json_artifact(db, job_id, NARRATIVE_ARTIFACT, narrative)
        db.commit()

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
        leaderboard=state.get("leaderboard"),
        final_model=state.get("final_model_info"),
        explainability=state.get("explainability"),
        narrative=narrative,
        critic=state.get("critic"),
        sampling_note=state.get("sampling_note", ""),
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

    _store_pdf(job_id, markdown, state.get("filename", "dataset.csv"))
    return {"report_markdown": markdown, "narrative": narrative}


def _store_pdf(job_id: int, markdown: str, filename: str) -> None:
    """Render and store the PDF, or log why there is none.

    Deliberately not allowed to fail the node. The Markdown report is the
    authoritative document and is already saved by the time this runs; losing a
    completed run because a rendering library could not find a font would be a
    poor trade.
    """
    try:
        data = render_pdf(markdown, title=f"AutoDS analysis of {filename}")
    except PdfError as exc:
        logger.warning("[job %s] the report PDF could not be rendered: %s", job_id, exc)
        return

    with SessionLocal() as db:
        register_bytes_artifact(
            db,
            job_id,
            REPORT_PDF_ARTIFACT,
            data,
            content_type="application/pdf",
            kind=ArtifactKind.REPORT,
        )
        db.commit()


def chat_index_node(state: PipelineState) -> dict:
    """Embed the run's written output so it can be asked about (spec 7.13).

    Runs last, over the artifacts every node before it produced. The passages are
    built from the *contracts* rather than re-read from storage, because they are
    already on the shared state -- and a chunk built from the object the report
    was written from cannot disagree with the report.

    Failure here is contained rather than propagated. Indexing is the last thing
    the pipeline does and nothing depends on it: a run whose embeddings failed
    still has a model, a report and a PDF, and losing all of that because a chat
    feature could not be prepared would be the wrong trade. The Chat page reports
    an unindexed run for itself, so the failure is visible where it matters.
    """
    job_id = state["job_id"]
    try:
        chunks = build_chunks(
            filename=state.get("filename", ""),
            narrative=state.get("narrative"),
            critic=state.get("critic"),
            explainability=state.get("explainability"),
            clustering=state.get("clustering_report"),
            evaluation=state.get("evaluation"),
            eda=state.get("eda_report"),
            cleaning=state.get("cleaning_report"),
            features=state.get("feature_strategy"),
        )
        with SessionLocal() as db:
            stored = index_run(db, job_id, chunks)
            db.commit()
    except Exception:
        logger.exception("[job %s] the run could not be indexed for chat", job_id)
        return {"chunks_indexed": 0}

    return {"chunks_indexed": stored}


NODE_FUNCTIONS: dict[str, Callable[[PipelineState], dict]] = {
    "planner": planner_node,
    "cleaning": cleaning_node,
    "eda": eda_node,
    "sampling": sampling_node,
    "feature_strategy": feature_strategy_node,
    "feature_selection": feature_selection_node,
    "preprocessing": preprocessing_node,
    "modeling": modeling_node,
    "evaluation": evaluation_node,
    "final_training": final_training_node,
    "explainability": explainability_node,
    "critic": critic_node,
    "report": report_node,
    "chat_index": chat_index_node,
}


# The optional steps, and the plan flag that decides each (spec 7.3, 11). A node
# named here gets a conditional edge instead of a plain one: the graph either
# enters it or routes past it to the next node, and a node routed past is marked
# SKIPPED rather than left looking unreached.
#
# Everything else in the pipeline is unconditional on purpose. A step that can be
# skipped has to be one the run is genuinely correct without -- cleaning and
# preprocessing are not, and making them optional would only mean more ways to
# produce a wrong answer.
OPTIONAL_NODES: dict[str, Callable[[PlannerPlan | None], bool]] = {
    "sampling": lambda plan: bool(plan and plan.run_sampling),
    "feature_selection": lambda plan: bool(plan and plan.run_feature_selection),
}

_SKIP_REASONS: dict[str, str] = {
    "sampling": "The planner judged the dataset small enough to use every row.",
    "feature_selection": "The planner kept all features rather than selecting a subset.",
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


def _router(optional: str, following: str):
    """Route into an optional node, or past it -- marking it SKIPPED on the way.

    The marking happens here, at the point the decision is actually made, which
    is the only place that knows a node was passed over rather than not yet
    reached. LangGraph evaluates a router once per traversal, so the row is
    written exactly once.
    """

    def route(state: PipelineState) -> str:
        if OPTIONAL_NODES[optional](state.get("plan")):
            return optional
        reason = _SKIP_REASONS.get(optional, "The planner did not ask for this step.")
        skip_node(state["job_id"], optional, reason)
        logger.info("[job %s] node %s skipped: %s", state["job_id"], optional, reason)
        return following

    return route


def build_pipeline_graph():
    """Compile the pipeline: START → planner → ... → report → END.

    Linear until Section 7, which adds the spec's conditional edges (7.3, 11).
    Two nodes are now optional, and the graph is built by walking the node list
    and giving each optional node a conditional edge *from its predecessor* --
    which either enters it or jumps to its successor.

    Building it from ``PIPELINE_NODES`` rather than wiring edges by hand means
    the roadmap the UI polls and the graph the worker executes cannot drift
    apart: they are the same list. It does assume no two optional nodes are
    adjacent, since a chain of skips would need to route past both -- asserted
    below rather than left as a comment, because the failure would be a
    silently-unreachable node.
    """
    graph = StateGraph(PipelineState)

    for name in PIPELINE_NODES:
        graph.add_node(name, _tracked(name, NODE_FUNCTIONS[name]))

    graph.add_edge(START, PIPELINE_NODES[0])

    for index, (earlier, later) in enumerate(zip(PIPELINE_NODES, PIPELINE_NODES[1:], strict=False)):
        if later not in OPTIONAL_NODES:
            graph.add_edge(earlier, later)
            continue

        following = PIPELINE_NODES[index + 2]
        if following in OPTIONAL_NODES:
            raise RuntimeError(
                f"Optional nodes {later!r} and {following!r} are adjacent; the "
                "router can only skip one at a time."
            )
        graph.add_conditional_edges(
            earlier, _router(later, following), {later: later, following: following}
        )

    graph.add_edge(PIPELINE_NODES[-1], END)

    return graph.compile()
