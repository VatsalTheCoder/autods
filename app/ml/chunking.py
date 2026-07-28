"""Turn a finished run into passages worth retrieving (spec 7.13).

RAG is only as good as its chunks, and the usual failure is mechanical: split the
report every 500 characters and hope a question lands inside one window. That
produces passages beginning mid-sentence and ending mid-table, and it embeds
layout as though it were meaning.

**These chunks are semantic units, one per thing a person might ask about.** One
per SHAP feature, one per cluster, one per critic finding. That is a deliberate
trade -- more, smaller passages rather than fewer, larger ones -- and it is made
because of the question the build plan uses as its own example: *"why was
transaction amount important?"* For that to retrieve well, there has to be a
passage that is *about* transaction amount rather than a passage about
explainability in general that happens to mention it.

**Every chunk is self-contained prose.** A fragment reading `0.23 · positive` has
no meaning to embed; "Support calls is the strongest driver, contributing 31% of
the model's decisions, and higher values push towards churn" does. Each passage
names its subject, so a retrieved chunk is intelligible on its own -- both to the
embedding model and to a reader checking where an answer came from.

Numbers appear here even though `agents/report_writer.py` forbids the model from
writing them. The distinction is the same one that section drew: code may state a
figure from an artifact, and a model may not invent one. These passages are built
by code from the artifacts, and the chat agent is required to quote them rather
than compute anything -- arithmetic goes to the pandas tool instead.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.ml.contracts import (
    CleaningReport,
    ClusteringReport,
    CriticReport,
    EdaReport,
    EvaluationReport,
    ExplainabilityReport,
    FeatureStrategy,
    NarrativeReport,
)

# How many of the ranked lists become their own passage. Beyond this the tail is
# noise: a feature carrying 0.4% of a model's decisions is not what anyone is
# asking about, and one passage per item would bury the ones that matter.
TOP_FEATURES = 12
TOP_CORRELATIONS = 6


@dataclass(frozen=True)
class Chunk:
    """One retrievable passage.

    ``source`` is the artifact it came from, so an answer can cite it. ``heading``
    is prepended to the embedded text: a passage about one feature is far easier
    to retrieve when its own subject line is part of what was embedded.
    """

    source: str
    heading: str
    content: str
    ordinal: int

    def embedding_text(self) -> str:
        """What actually gets embedded -- heading and body together."""
        return f"{self.heading}\n{self.content}"


def _percent(value: float) -> str:
    return f"{value * 100:.1f}%"


def build_chunks(
    *,
    filename: str = "",
    narrative: NarrativeReport | None = None,
    critic: CriticReport | None = None,
    explainability: ExplainabilityReport | None = None,
    clustering: ClusteringReport | None = None,
    evaluation: EvaluationReport | None = None,
    eda: EdaReport | None = None,
    cleaning: CleaningReport | None = None,
    features: FeatureStrategy | None = None,
) -> list[Chunk]:
    """Build every passage for a run, in a stable order.

    Every argument is optional. A run whose EDA failed still has a model to ask
    about, and the chat should work over whatever the run actually produced --
    the same tolerance the report has, for the same reason.
    """
    chunks: list[Chunk] = []

    def add(source: str, heading: str, content: str) -> None:
        # Blank bodies are dropped rather than stored: an empty passage is a
        # retrievable result that answers nothing, and it would compete for a
        # place in the top-k against one that does.
        if content and content.strip():
            chunks.append(Chunk(source, heading, content.strip(), len(chunks)))

    _add_narrative(add, narrative)
    _add_evaluation(add, evaluation, filename)
    _add_explainability(add, explainability)
    _add_clustering(add, clustering)
    _add_critic(add, critic)
    _add_data(add, eda, cleaning, features)
    return chunks


def _add_narrative(add, narrative: NarrativeReport | None) -> None:
    """The report's prose, which is already written for a human to read."""
    if narrative is None or narrative.is_empty():
        return
    add("narrative_report.json", "In short: what this analysis found", narrative.executive_summary)
    add("narrative_report.json", "About the data", narrative.data_story)
    add("narrative_report.json", "About the model", narrative.model_story)
    if narrative.recommendations:
        add(
            "narrative_report.json",
            "What to do next",
            "\n".join(f"- {item}" for item in narrative.recommendations),
        )


def _add_evaluation(add, evaluation: EvaluationReport | None, filename: str) -> None:
    if evaluation is None:
        return

    primary = evaluation.metrics.get(evaluation.primary_metric)
    scored = ""
    if primary is not None:
        scored = (
            f" It scored {primary.mean:.4f} on {evaluation.primary_metric}, "
            f"varying by {primary.std:.4f} across folds."
        )

    source = filename or "the uploaded dataset"
    add(
        "evaluation_report.json",
        f"The model chosen for {evaluation.target_column}",
        f"The pipeline trained a {evaluation.model_name} to predict "
        f"{evaluation.target_column} from {source}, a {evaluation.task_type} task."
        f"{scored} It used {evaluation.n_rows:,} rows and "
        f"{evaluation.n_features} features.",
    )

    add(
        "evaluation_report.json",
        "How the model was validated",
        f"Performance was measured with {evaluation.n_folds}-fold "
        f"{evaluation.cv_strategy} cross-validation. For each fold the whole "
        f"preparation pipeline was fitted on the training part only and scored on "
        f"the held-out part, so no information from the test rows reached the "
        f"model that scored them.",
    )

    others = [
        f"{name}: {summary.mean:.4f} (± {summary.std:.4f})"
        for name, summary in evaluation.metrics.items()
        if name != evaluation.primary_metric
    ]
    if others:
        add(
            "evaluation_report.json",
            "The other metrics the model was scored on",
            "Averaged across folds — " + "; ".join(others) + ".",
        )


def _add_explainability(add, explainability: ExplainabilityReport | None) -> None:
    """One passage per feature. The highest-value decision in this module.

    "Why was X important?" is the question the build plan names, and it retrieves
    well only if some passage is *about* X.
    """
    if explainability is None or not explainability.global_importance:
        return

    ranked = explainability.global_importance[:TOP_FEATURES]
    add(
        "explainability_report.json",
        "Which columns the model relies on most",
        "In order of influence: "
        + ", ".join(f"{item.feature} ({_percent(item.share)})" for item in ranked)
        + f". Measured as {explainability.aggregation.lower()} using "
        f"{explainability.explainer} over {explainability.n_rows_explained:,} rows.",
    )

    for position, item in enumerate(ranked, start=1):
        # ``direction`` is already a phrase written by ml/explain.py ("higher
        # values push the prediction up"), not a code to translate -- and it is
        # empty for features whose effect is not one-directional, which is
        # exactly when there is nothing truthful to say about direction. Using
        # it verbatim keeps one wording for this claim across the report, the
        # Results page and the chat.
        direction = f" For this model, {item.direction}." if item.direction else ""

        encoded = ""
        if len(item.encoded_features) > 1:
            encoded = (
                f" The model sees it as {len(item.encoded_features)} encoded "
                f"columns, whose contributions are added together here."
            )

        add(
            "explainability_report.json",
            f"Why {item.feature} matters to the model",
            f"{item.feature} is the number {position} most important column, "
            f"carrying {_percent(item.share)} of the model's total explained "
            f"influence.{direction}{encoded}",
        )


def _add_clustering(add, clustering: ClusteringReport | None) -> None:
    """One passage per group, because "describe the groups" asks about each."""
    if clustering is None or not clustering.profiles:
        return

    add(
        "clustering_report.json",
        "The natural groups found in the data",
        f"Clustering with {clustering.method} found {clustering.k} groups "
        f"(silhouette score {clustering.silhouette:.3f}). These describe the data "
        f"only — they were never used as model features, because they are computed "
        f"over every row including the ones held out for testing.",
    )

    for profile in clustering.profiles:
        distinguishing = ""
        if profile.distinguishing_features:
            distinguishing = " Set apart by: " + ", ".join(profile.distinguishing_features) + "."
        add(
            "clustering_report.json",
            f"Group {profile.cluster}: {profile.description or 'unlabelled group'}",
            f"Group {profile.cluster} holds {profile.size:,} rows "
            f"({_percent(profile.share)} of the data). "
            f"{profile.description}{distinguishing}",
        )


def _add_critic(add, critic: CriticReport | None) -> None:
    """One passage per finding, so a concern can be retrieved by its subject."""
    if critic is None:
        return

    if critic.verdict:
        add("critic_report.json", "The reviewer's overall verdict on this run", critic.verdict)

    if critic.strengths:
        add(
            "critic_report.json",
            "What held up well in this analysis",
            "\n".join(f"- {item}" for item in critic.strengths),
        )

    for finding in critic.by_severity():
        measured = (
            " This was measured directly from the artifacts rather than judged."
            if finding.measured
            else ""
        )
        add(
            "critic_report.json",
            f"A {finding.severity} the review raised about {finding.area}",
            f"{finding.finding} Recommendation: {finding.recommendation}{measured}",
        )

    if critic.recommended_next_steps:
        add(
            "critic_report.json",
            "What the reviewer suggests doing next",
            "\n".join(f"- {item}" for item in critic.recommended_next_steps),
        )

    if critic.omissions:
        # Retrievable on purpose. If someone asks the chat what the review
        # covered, the honest answer includes what it could not see.
        add(
            "critic_report.json",
            "What the review did not see",
            "The reviewer reads a summary sized to fit the model's token limit. "
            "On this run it was capped as follows:\n"
            + "\n".join(f"- {note}" for note in critic.omissions),
        )


def _add_data(
    add,
    eda: EdaReport | None,
    cleaning: CleaningReport | None,
    features: FeatureStrategy | None,
) -> None:
    """The dataset's own story: shape, cleaning, correlations, preparation.

    Deliberately *not* one passage per column. Per-column statistics are
    arithmetic, and arithmetic is the pandas tool's job -- indexing five hundred
    passages of means and medians would flood retrieval with numbers that the
    other tool answers exactly.
    """
    if eda is not None:
        add(
            "eda_report.json",
            "The shape of the dataset",
            f"The cleaned dataset has {eda.n_rows:,} rows and {eda.n_columns} "
            f"columns, predicting {eda.target_column}.",
        )

        if eda.class_balance and eda.class_balance.counts:
            counts = ", ".join(f"{k}: {v:,}" for k, v in eda.class_balance.counts.items())
            verdict = (
                "This is imbalanced enough to matter, and the pipeline resampled "
                "inside each training fold to compensate."
                if eda.class_balance.imbalanced
                else "The classes are balanced enough not to need resampling."
            )
            add(
                "eda_report.json",
                f"How the {eda.target_column} outcomes are distributed",
                f"{counts}. That is a ratio of "
                f"{eda.class_balance.imbalance_ratio:.2f} to 1. {verdict}",
            )

        if eda.top_correlations:
            pairs = "; ".join(
                f"{pair.left} and {pair.right} ({pair.correlation:+.2f})"
                for pair in eda.top_correlations[:TOP_CORRELATIONS]
            )
            add(
                "eda_report.json",
                "Which columns move together",
                f"The strongest correlations found were: {pairs}. Correlation is "
                "not causation, and these were not used to select features.",
            )

    if cleaning is not None:
        removed = cleaning.n_rows_before - cleaning.n_rows_after
        dropped = (
            f" It dropped {len(cleaning.dropped_columns)} columns: "
            + ", ".join(cleaning.dropped_columns)
            + "."
            if cleaning.dropped_columns
            else ""
        )
        add(
            "cleaning_report.json",
            "What cleaning did to the data",
            f"Cleaning removed {removed:,} of {cleaning.n_rows_before:,} rows "
            f"({cleaning.duplicate_rows_removed:,} duplicates, "
            f"{cleaning.missing_target_rows_removed:,} with no target value)."
            f"{dropped} Missing values in the remaining columns were left to the "
            "modelling pipeline, so they are filled inside each fold rather than "
            "from statistics over the whole dataset.",
        )

    if features is not None and features.columns:
        rationales = [
            f"{column.column}: {column.rationale}"
            for column in features.columns
            if column.rationale
        ]
        if rationales:
            add(
                "feature_report.json",
                "Why each column was prepared the way it was",
                "\n".join(f"- {line}" for line in rationales[:TOP_FEATURES]),
            )
