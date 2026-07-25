"""The shapes every Section 5 stage produces, and the JSON artifacts they become.

One module for all of them, and deliberately dependency-free -- pure Pydantic,
no pandas, no sklearn, no database. Cleaning, preprocessing, modelling, the
planner and the report all import from here, so nothing has to import from
anything else in the pipeline just to name its own output.

Each model serialises straight to its artifact (``cleaning_report.json``,
``evaluation_report.json``, ...), which means the file in S3 and the object the
code passed around can never drift apart -- the same guarantee ``SchemaReport``
gives the upload path.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

# Reused rather than redefined. There is exactly one definition of "is this
# classification or regression" in the codebase, and the schema the user
# confirmed is where it comes from; a parallel Literal here would be one
# rename away from silently disagreeing with it.
from app.agents.schema_models import ClassBalance, TaskType

# ---- The planner's output ---------------------------------------------------


ClusteringMethod = Literal["kmeans", "kprototypes"]


class PlannerPlan(BaseModel):
    """The plan the LLM writes, and the flags the pipeline actually obeys.

    Every field here is read by a later node -- there are no aspirational
    settings nothing consumes, because a plan field that changes no behaviour is
    just a lie in an artifact. Section 5 started with two booleans; Section 6 adds
    the clustering choice the spec gives the Planner (9). SMOTE and the model
    roster arrive in Sections 7 and 8, when there is something to switch between.
    """

    drop_duplicate_rows: bool = Field(
        default=True,
        description="Remove exactly-repeated rows before modelling.",
    )
    drop_high_null_columns: bool = Field(
        default=True,
        description="Drop columns that are mostly empty rather than imputing them.",
    )

    # Spec 9 gives the Planner this choice, but the rule behind it is mechanical
    # and the wrong answer is not merely suboptimal -- K-Means on categorical data
    # requires inventing Euclidean distances between labels that have none. So the
    # model states a preference and ``clustering.py`` overrides it when the data
    # cannot support it, recording that it did. Same shape as schema detection
    # validating the LLM's target against the columns that actually exist.
    clustering_method: ClusteringMethod = Field(
        default="kmeans",
        description="kmeans for all-numeric data; kprototypes when categories are present.",
    )
    run_clustering: bool = Field(
        default=True,
        description="Whether looking for natural groupings is worthwhile here.",
    )

    rationale: str = Field(default="", description="One or two sentences on why.")

    # Whether the LLM was actually consulted. The pipeline runs identically
    # either way, but the report says which -- an artifact should never imply a
    # model made a decision that a hardcoded default made.
    source: Literal["llm", "default"] = "default"


# ---- Cleaning ---------------------------------------------------------------


class DroppedColumn(BaseModel):
    name: str
    reason: str


class DtypeCorrection(BaseModel):
    name: str
    from_dtype: str
    to_dtype: str


class CleaningReport(BaseModel):
    """What cleaning did, in numbers a reader can check against the dataset."""

    n_rows_before: int
    n_rows_after: int
    n_columns_before: int
    n_columns_after: int

    duplicate_rows_removed: int = 0
    missing_target_rows_removed: int = 0
    dropped_columns: list[DroppedColumn] = Field(default_factory=list)
    dtype_corrections: list[DtypeCorrection] = Field(default_factory=list)

    # Missing values still present when cleaning finishes -- on purpose, and the
    # single most important thing this report communicates. Filling them here
    # would mean computing a median over rows that later land in a test fold,
    # which is exactly the leakage the project promises not to commit (spec 8).
    # Imputation is a step in the unfitted pipeline instead, fitted per fold.
    missing_values_left_to_the_pipeline: dict[str, int] = Field(default_factory=dict)


# ---- Preprocessing (the recipe, not the meal) -------------------------------


class PreprocessingSpec(BaseModel):
    """A description of the unfitted pipeline handed to cross-validation.

    This is the human-readable half of the "recipe, not a cooked meal" contract:
    the ``ColumnTransformer`` itself goes to S3 as a pickle, and this JSON says
    what is in it. Nothing here records whether the pipeline is fitted, because
    a self-reported flag would prove nothing -- the tests assert unfittedness
    against the real object instead.
    """

    numeric_columns: list[str] = Field(default_factory=list)
    categorical_columns: list[str] = Field(default_factory=list)
    # Columns cleaning kept but the Section 5 recipe has no strategy for
    # (datetime, free text). Named rather than silently ignored so the gap is
    # visible; Section 7 gives them real handling.
    unhandled_columns: list[DroppedColumn] = Field(default_factory=list)

    numeric_strategy: str = ""
    categorical_strategy: str = ""

    # "hardcoded" in Section 5 -- the LLM does not choose strategies until
    # Section 7 (build-plan), and the report should say so plainly.
    strategy_source: Literal["hardcoded", "llm"] = "hardcoded"


# ---- Evaluation -------------------------------------------------------------


class FoldScore(BaseModel):
    """One fold's metrics, plus the row counts that make the split auditable.

    ``n_train`` / ``n_test`` are recorded because they are the evidence that
    fitting happened on a *fold* and not the whole dataset -- a reader can check
    that n_train is roughly four fifths of the data, and the test suite asserts
    the transformer saw exactly that many rows.
    """

    fold: int
    n_train: int
    n_test: int
    metrics: dict[str, float] = Field(default_factory=dict)


class MetricSummary(BaseModel):
    """A metric across folds. The ``std`` is why cross-validation was worth it."""

    mean: float
    std: float


class EvaluationReport(BaseModel):
    """The cross-validated result -- the artifact Section 5 exists to produce."""

    task_type: TaskType
    target_column: str
    model_name: str

    n_folds: int
    cv_strategy: str
    n_rows: int
    # Feature columns entering the pipeline (before encoding expands them).
    n_features: int

    folds: list[FoldScore] = Field(default_factory=list)
    metrics: dict[str, MetricSummary] = Field(default_factory=dict)
    # The metric a reader should look at first, and later sections will rank on.
    primary_metric: str = ""

    # Metrics that could not be computed for this dataset (ROC-AUC with a class
    # missing from a fold, say). Surfaced rather than dropped, so an absent
    # number is explained instead of looking like an oversight.
    warnings: list[str] = Field(default_factory=list)

    def primary_score(self) -> float | None:
        summary = self.metrics.get(self.primary_metric)
        return summary.mean if summary else None


# ---- EDA (Section 6) --------------------------------------------------------


class NumericSummary(BaseModel):
    """The five-number summary plus mean and spread, for one numeric column."""

    mean: float
    std: float
    minimum: float
    q1: float
    median: float
    q3: float
    maximum: float
    # Rows beyond 1.5x the interquartile range from the quartiles. Reported, not
    # removed: an outlier is often the most interesting row in the dataset, and
    # deciding to drop one is a modelling choice the user should make knowingly.
    outlier_count: int = 0


class CategorySummary(BaseModel):
    """What a categorical column actually contains."""

    n_unique: int
    # The commonest values and their counts, capped -- enough to see the shape of
    # the distribution without embedding a whole column in a JSON report.
    top_values: dict[str, int] = Field(default_factory=dict)


class ColumnStatistics(BaseModel):
    """One column's summary. Exactly one of the two summaries is populated."""

    name: str
    semantic_type: str
    count: int
    missing: int
    missing_rate: float = Field(..., ge=0.0, le=1.0)
    numeric: NumericSummary | None = None
    categorical: CategorySummary | None = None


class CorrelationPair(BaseModel):
    """Two columns that move together, and how strongly."""

    left: str
    right: str
    correlation: float


class EdaReport(BaseModel):
    """``eda_report.json`` -- the descriptive picture of the cleaned dataset."""

    n_rows: int
    n_columns: int
    target_column: str
    columns: list[ColumnStatistics] = Field(default_factory=list)

    # Strongest absolute Pearson correlations between numeric columns. Ranked
    # because the interesting content of a correlation matrix is its extremes,
    # and a reader should not have to scan a grid to find them.
    top_correlations: list[CorrelationPair] = Field(default_factory=list)
    class_balance: ClassBalance | None = None

    # Artifact names of the charts, in display order. The bytes live in S3 and
    # are served by GET /jobs/{id}/artifacts/{name}/content.
    plots: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


# ---- Clustering (Section 6) -------------------------------------------------


class ClusterProfile(BaseModel):
    """One discovered group: how big it is, what marks it out, what it means."""

    cluster: int
    size: int
    share: float = Field(..., ge=0.0, le=1.0)
    # The few features on which this group departs most from the dataset average.
    # Computed in code; this is what the LLM is given to describe, so its summary
    # is grounded in measured differences rather than invented ones.
    distinguishing_features: dict[str, str] = Field(default_factory=dict)
    # The LLM's plain-language description. Empty when no model was available --
    # the clustering itself does not depend on it.
    description: str = ""


class ClusteringReport(BaseModel):
    """``clustering_report.json`` -- the groupings, and the guardrail restated.

    Clustering here is **EDA insight only**. Cluster labels are never added to the
    dataset as a feature: they are computed over every row, including the rows
    that later land in a test fold, so feeding them to the model would leak
    exactly like fitting a scaler up front (spec 9). The labels exist to colour a
    scatter plot and to be described in words, and nothing else.
    """

    method: ClusteringMethod
    k: int
    # Mean silhouette at the chosen k: roughly, how much better-separated the
    # groups are than chance. Near 0 means the "groups" are arbitrary slices, and
    # the report says so rather than presenting them as discoveries.
    silhouette: float
    # Every k tried, so a reader can see the choice was searched and not assumed.
    silhouette_by_k: dict[int, float] = Field(default_factory=dict)

    profiles: list[ClusterProfile] = Field(default_factory=list)
    scatter_plot: str | None = None
    # Set when the planner's method was overridden because the data could not
    # support it -- an artifact should never imply a choice that was not obeyed.
    method_override_reason: str | None = None
    warnings: list[str] = Field(default_factory=list)

    def is_weak(self) -> bool:
        """True when the groups are not separated enough to present as findings.

        The threshold is Kaufman & Rousseeuw's boundary between "reasonable
        structure" (above 0.5) and "weak, could be artificial" (below it). It is
        set here rather than at their lower 0.25 line -- "no substantial
        structure" -- for a measured reason: k-means on *pure Gaussian noise*
        scores about 0.36 on this pipeline, because partitioning a single blob
        produces respectable-looking silhouettes. Anything under 0.5 therefore
        has to be hedged, or the tool would report invented segments in random
        data as discoveries.
        """
        return self.silhouette < 0.5
