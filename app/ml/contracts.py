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

    # ---- Section 7: the optional steps the graph branches around -------------
    # These are the fields spec 7.3 has the Planner decide and LangGraph's
    # conditional edges consume. Each one can turn a node off entirely, and a
    # node that is turned off is marked SKIPPED rather than silently absent --
    # which is the visible half of the "dynamic orchestration" claim (spec 11).

    use_smote: bool = Field(
        default=False,
        description="Oversample the minority class inside each training fold.",
    )
    run_feature_selection: bool = Field(
        default=False,
        description="Keep only the strongest features rather than all of them.",
    )
    run_sampling: bool = Field(
        default=False,
        description="Train on a random subset because the dataset is very large.",
    )

    # Regression only, and off by default: removing rows changes what is being
    # predicted, which is the user's question to answer rather than a default to
    # assume. Bounded by construction to the outermost half-percent at each end
    # (``cleaning.py``), so an over-eager plan cannot delete a real population.
    trim_target_outliers: bool = Field(
        default=False,
        description=(
            "Drop rows whose numeric target sits in the extreme tail, when those "
            "values look like data-entry artefacts rather than genuine records."
        ),
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


class TargetOutliers(BaseModel):
    """The extreme tail of a numeric target, measured whether or not it was cut.

    Reported even when nothing was removed, because "1,044 listings priced above
    $500, against a median of $106" is the single most useful sentence a reader
    of a mediocre regression score can be given. Silence there is what leaves
    someone staring at R² 0.07 with no idea that a few hundred rows own most of
    the error.
    """

    column: str
    n_detected: int = 0
    n_removed: int = 0
    lower_bound: float | None = None
    upper_bound: float | None = None
    # What the tail is extreme *relative to*, so the bounds can be read without
    # the dataset to hand.
    median: float | None = None
    maximum: float | None = None
    note: str = ""


class CleaningReport(BaseModel):
    """What cleaning did, in numbers a reader can check against the dataset."""

    n_rows_before: int
    n_rows_after: int
    n_columns_before: int
    n_columns_after: int

    duplicate_rows_removed: int = 0
    missing_target_rows_removed: int = 0
    non_finite_target_rows_removed: int = 0
    dropped_columns: list[DroppedColumn] = Field(default_factory=list)
    dtype_corrections: list[DtypeCorrection] = Field(default_factory=list)
    target_outliers: TargetOutliers | None = None

    # Missing values still present when cleaning finishes -- on purpose, and the
    # single most important thing this report communicates. Filling them here
    # would mean computing a median over rows that later land in a test fold,
    # which is exactly the leakage the project promises not to commit (spec 8).
    # Imputation is a step in the unfitted pipeline instead, fitted per fold.
    missing_values_left_to_the_pipeline: dict[str, int] = Field(default_factory=dict)


# ---- Feature strategy (Section 7) -------------------------------------------
#
# The per-column decisions the LLM is allowed to make. These are deliberately
# closed ``Literal`` sets rather than free text: an LLM that can only answer with
# one of five words cannot ask for a transformation the code has no way to build,
# and Pydantic rejects anything outside the set before it reaches a pipeline.
# This is the whole reason the agent emits a table of choices instead of code.

# What the column *is*, which decides which recipe it gets built into.
ColumnRole = Literal["numeric", "categorical", "ordinal", "datetime", "text", "drop"]
ImputeStrategy = Literal["median", "mean", "most_frequent", "constant", "none"]
EncodeStrategy = Literal["onehot", "ordinal", "frequency", "tfidf", "none"]
ScaleStrategy = Literal["standard", "minmax", "none"]


class ColumnStrategy(BaseModel):
    """How one column should be prepared -- the LLM's answer for a single column."""

    column: str
    role: ColumnRole
    impute: ImputeStrategy = "none"
    encode: EncodeStrategy = "none"
    scale: ScaleStrategy = "none"

    # Only meaningful when ``role`` is ``ordinal``: the categories from lowest to
    # highest. This is the one place the LLM contributes knowledge the data does
    # not contain -- that "small" precedes "medium" precedes "large" is a fact
    # about English, not a fact recoverable from the column. Validated against the
    # values actually present before it is used.
    ordinal_order: list[str] = Field(default_factory=list)

    rationale: str = ""


class StrategyOverride(BaseModel):
    """A decision the code refused to carry out, and what it did instead.

    The counterpart to schema detection validating the LLM's target against the
    real columns, and to the planner's clustering method being overridden when the
    data cannot support it. Recorded rather than silently applied, because an
    artifact that showed the LLM's request while the pipeline ran something else
    would be a lie about how the model was built.
    """

    column: str
    field: str
    requested: str
    applied: str
    reason: str


class FeatureStrategy(BaseModel):
    """``feature_report.json`` -- what was asked for, what was allowed, what ran.

    Half of spec 7.6's output (the other half is the unfitted pickle). It exists
    as its own artifact rather than being folded into the preprocessing report
    because the two answer different questions: this one is *the decision*, and
    ``preprocessing_report.json`` is *what was built from it*. Keeping them apart
    is what makes the "LLM decides, code executes" split checkable from outside --
    a reader can diff the request against the result.
    """

    columns: list[ColumnStrategy] = Field(default_factory=list)

    # Columns the LLM invented. The spec's requirement is that these are rejected,
    # and naming them is how a reader can tell rejection happened rather than the
    # model simply not hallucinating on this run.
    rejected_columns: list[str] = Field(default_factory=list)
    # Real columns the LLM said nothing about; they fall back to the deterministic
    # dtype-based default, which is exactly the Section 5 behaviour.
    defaulted_columns: list[str] = Field(default_factory=list)
    overrides: list[StrategyOverride] = Field(default_factory=list)

    # "llm" only when the model was consulted *and* replied usably. A strategy
    # that fell back to defaults must not be recorded as an LLM decision.
    source: Literal["llm", "hardcoded"] = "hardcoded"
    rationale: str = ""

    def for_column(self, name: str) -> ColumnStrategy | None:
        return next((c for c in self.columns if c.column == name), None)


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
    # Section 7 gives these three real handling; before it they were listed under
    # ``unhandled_columns`` as a stated gap.
    ordinal_columns: list[str] = Field(default_factory=list)
    datetime_columns: list[str] = Field(default_factory=list)
    text_columns: list[str] = Field(default_factory=list)
    # Columns that still get no strategy -- now only the ones something explicitly
    # chose to drop, rather than whole dtypes the recipe could not express.
    unhandled_columns: list[DroppedColumn] = Field(default_factory=list)

    numeric_strategy: str = ""
    categorical_strategy: str = ""
    # Per-column detail, in the order the transformer applies it. The summary
    # strings above survive for the Markdown report and the Results page, which
    # want one line rather than a table.
    column_strategies: list[ColumnStrategy] = Field(default_factory=list)

    # "hardcoded" when the dtype-based defaults built the recipe, "llm" when the
    # feature strategy agent chose. Section 5 could only ever say the former.
    strategy_source: Literal["hardcoded", "llm"] = "hardcoded"

    # Set when the planner asked for feature selection: the selector is a *step in
    # the pipeline*, so it is fitted per fold like everything else. Recorded here
    # because "how many features survived" is otherwise invisible from the report.
    feature_selection: str = ""


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


# ---- The leaderboard (Section 7) --------------------------------------------


class LeaderboardEntry(BaseModel):
    """One model's cross-validated result, as it appears in the ranking."""

    rank: int
    model_name: str
    # The metric the ranking is on -- named per row so a reader never has to
    # infer which number the order came from.
    primary_metric: str
    score: float
    # Spread across folds. It belongs next to the score because it is what says
    # whether a lead is real: 0.81 ± 0.02 beating 0.79 ± 0.09 is a different
    # claim from the means alone.
    std: float
    metrics: dict[str, MetricSummary] = Field(default_factory=dict)
    fit_seconds: float = 0.0

    # A model that ignores the features entirely, included so a reader can tell a
    # mediocre score from a meaningless one. Ranked with everything else, but
    # never served -- ``modeling.run_leaderboard`` picks the best non-baseline.
    is_baseline: bool = False
    # Set when a candidate could not be trained at all. A named failure is more
    # useful than a model quietly missing from the table.
    error: str = ""


class Leaderboard(BaseModel):
    """``leaderboard.json`` -- every candidate, ranked, on identical folds.

    The comparison is only meaningful because every model in it saw exactly the
    same splits, built from the same seed, behind the same unfitted recipe. That
    is why the roster is cross-validated in one pass here rather than each model
    being run separately and the numbers collected afterwards.
    """

    task_type: TaskType
    target_column: str
    primary_metric: str
    n_folds: int
    cv_strategy: str
    entries: list[LeaderboardEntry] = Field(default_factory=list)

    # Whether resampling was part of every fold's fit, and why. Recorded on the
    # leaderboard rather than only in the plan because it changes what the scores
    # mean, and a reader comparing two runs needs to see it next to them.
    resampling: str = ""
    warnings: list[str] = Field(default_factory=list)

    def winner(self) -> LeaderboardEntry | None:
        return next((e for e in self.entries if not e.error), None)


# ---- Final training & explainability (Section 8) ----------------------------


class PredictorColumn(BaseModel):
    """One raw input column the served model expects, as a form can present it.

    Raw, not encoded: the caller of ``POST /jobs/{id}/predict`` sends ``city:
    "London"``, and the saved pipeline does the one-hot encoding itself. Anything
    else would make the endpoint's contract the *recipe's* internals, which
    change whenever the feature strategy does.
    """

    name: str
    dtype: str
    role: ColumnRole = "numeric"
    # One value observed in the training data, so a form can be pre-filled with
    # something plausible rather than leaving the user to guess the units.
    example: str = ""

    # False for a column the recipe drops -- excluded at the checkpoint, or given
    # ``role: drop`` by the strategy. It stays on the list because the fitted
    # ColumnTransformer was fitted against a frame that had it and will object to
    # one that does not, but nothing should *ask* a user for a value the model
    # then ignores, and a blank one must not be reported as a missing input.
    used: bool = True


class FinalModelInfo(BaseModel):
    """``final_model.json`` -- what the served pickle is, and what it is not.

    The pickle beside it is the *only* estimator in this project fitted on every
    row. That is correct here and nowhere else: cross-validation has already
    produced the honest score, and refitting on everything is how you get the
    strongest model to actually serve (spec 7.9). The distinction this file has to
    keep visible is that the final model therefore has **no held-out score of its
    own** -- ``cv_score`` is carried over from the cross-validated run of the same
    configuration, and is an estimate of this model's performance, not a
    measurement of it.
    """

    model_name: str
    task_type: TaskType
    target_column: str

    # Rows and raw feature columns the refit saw. ``n_rows`` matching the cleaned
    # dataset's row count is the check that "full dataset" means what it says.
    n_rows: int
    n_features: int
    # Raw input columns, in the order the recipe expects them.
    feature_columns: list[PredictorColumn] = Field(default_factory=list)
    # Class labels as the user uploaded them, in the model's own order. Empty for
    # regression.
    classes: list[str] = Field(default_factory=list)

    # The metric the leaderboard ranked on, and the winner's cross-validated
    # value. Named ``cv_`` throughout so no reader mistakes it for a score this
    # model earned on data it had not seen.
    primary_metric: str = ""
    cv_score: float | None = None
    resampling: str = ""

    artifact: str = ""
    warnings: list[str] = Field(default_factory=list)


class FeatureImportance(BaseModel):
    """How much one **source** column moved the model's output, on average.

    Source column, not encoded feature: SHAP explains ``city_London`` and
    ``city_Leeds``, and a user wants to know about ``city``. The importance here
    is the sum of the mean absolute SHAP value over every encoded feature the
    column produced, which is the aggregation that keeps a one-hot column's total
    influence comparable with a numeric column's (spec 7.10).
    """

    feature: str
    importance: float
    # The column's share of all importance, so a reader can see whether the top
    # feature dominates or the model is spreading its attention.
    share: float = Field(default=0.0, ge=0.0, le=1.0)
    # What the column expanded into. This is the audit trail for the mapping --
    # the part of Section 8 most likely to be quietly wrong.
    encoded_features: list[str] = Field(default_factory=list)

    # "higher values push the prediction up" and friends -- populated only for a
    # column that produced exactly one numeric feature, where the sign of the
    # value/SHAP relationship is well defined. A one-hot column has no such
    # direction and gets an empty string rather than an invented one.
    direction: str = ""


class LocalContribution(BaseModel):
    """One feature's push on one row's prediction, in the units of the output."""

    feature: str
    # The row's actual value, formatted for display. A string because the column
    # may be a label, a number or a timestamp, and the report only shows it.
    value: str
    contribution: float


class LocalExplanation(BaseModel):
    """Why *this row* got *this answer* -- SHAP's additive decomposition.

    ``base_value + sum(contributions) == prediction``, in the model's output
    units (log-odds for a classifier, the target's units for a regressor). The
    contributions listed are the largest few; ``other_contribution`` carries the
    rest so the sum still adds up rather than appearing to lose mass.

    ``predicted`` and ``explained_class`` are separate because they can honestly
    differ. A gradient-booster on a binary target produces **one** output -- the
    margin towards the positive class -- so a row predicted ``no`` is explained by
    contributions that push *away* from ``yes``. Collapsing the two would mean
    either mislabelling the answer or mislabelling the direction.
    """

    row_label: str
    # The model's answer for this row, and its confidence in that answer.
    predicted: str
    probability: float | None = None
    # The class the contributions below push towards. Empty for regression.
    explained_class: str = ""

    base_value: float
    contributions: list[LocalContribution] = Field(default_factory=list)
    other_contribution: float = 0.0
    # base + contributions + other, in output units. Stated so a reader can do
    # the addition themselves.
    output_value: float = 0.0


class ExplainabilityReport(BaseModel):
    """``explainability_report.json`` -- the SHAP account of the final model.

    The project's explainable-AI claim in one artifact (spec 7.10). Two things in
    it are worth more than the numbers:

    * ``feature_name_mapping`` -- every encoded feature and the source column it
      came from. The mapping is what makes the rest legible, and publishing it
      means a reader can check it rather than trust it.
    * ``additivity_max_error`` -- SHAP values are only meaningful if they add up
      to the model's output. This is the largest gap found across the explained
      rows, measured rather than assumed. A large number here invalidates every
      other number in the file, so it is recorded next to them.
    """

    model_name: str
    task_type: TaskType
    target_column: str

    # "TreeExplainer" or "LinearExplainer" -- which one, and why, is the whole of
    # the dispatch decision (see ``ml/explain.py``).
    explainer: str = ""
    n_rows_explained: int = 0
    n_encoded_features: int = 0
    # Set when the rows explained are a sample rather than the whole dataset.
    sampling_note: str = ""

    classes: list[str] = Field(default_factory=list)
    # How the per-class SHAP values were combined into one ranking. Recorded
    # because "importance" means something different for a 3-class model than a
    # regression, and the difference should not be silent.
    aggregation: str = ""

    global_importance: list[FeatureImportance] = Field(default_factory=list)
    examples: list[LocalExplanation] = Field(default_factory=list)

    feature_name_mapping: dict[str, str] = Field(default_factory=dict)
    additivity_max_error: float = 0.0

    plots: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    def top_features(self, limit: int = 5) -> list[FeatureImportance]:
        return self.global_importance[:limit]


# ---- The critic (Section 9) -------------------------------------------------

# Where a finding lands. A closed set for the same reason every other LLM-facing
# vocabulary here is closed: a critique filed under an invented heading cannot be
# grouped, counted or acted on.
CriticArea = Literal[
    "data_quality",
    "features",
    "modelling",
    "evaluation",
    "explainability",
    "deployment",
]

# How much the finding should worry a reader. Deliberately three levels: a scale
# of five invites the model to split hairs it has no basis for.
CriticSeverity = Literal["blocker", "concern", "note"]


class CriticFinding(BaseModel):
    """One thing the review found, and what to do about it.

    ``recommendation`` is required in spirit -- a finding with no suggested action
    is a complaint. The spec asks the critic to "recommend simpler models,
    alternative strategies, or additional validation" (7.11), all of which are
    actions, not observations.
    """

    area: CriticArea
    severity: CriticSeverity
    finding: str
    recommendation: str = ""

    # True when the finding came from a threshold in code rather than from the
    # model. Recorded because the two carry different weight: a measured finding
    # is checkable against the artifacts, and a written one is an opinion about
    # them. A reader deserves to know which they are looking at.
    measured: bool = False


class CriticReport(BaseModel):
    """``critic_report.json`` -- the review of the whole run (spec 7.11).

    The one agent whose job is to argue with the rest of the pipeline. Two design
    choices keep that useful rather than decorative:

    * **It is grounded in measurements.** Findings that code can derive from
      thresholds are computed first and handed to the model as established fact.
      The model may add to them and must not contradict them, which is the same
      arrangement that keeps the cluster profiler honest -- it describes measured
      differences rather than inventing them.
    * **It records what it could not see.** A wide dataset's summary is capped
      (``agents/summaries.py``), so a review of it is a review of part of the run.
      ``omissions`` carries that forward into the artifact, because a critique
      that looks complete and is not would be worse than no critique.
    """

    verdict: str = ""
    confidence: Literal["high", "medium", "low"] = "medium"

    strengths: list[str] = Field(default_factory=list)
    findings: list[CriticFinding] = Field(default_factory=list)
    recommended_next_steps: list[str] = Field(default_factory=list)

    # "llm" only when the model was consulted *and* replied usably; a review that
    # is entirely the measured findings must not be presented as a written one.
    source: Literal["llm", "default"] = "default"
    # Which summarisation tier fed the review, and what it left out.
    detail_level: str = ""
    omissions: list[str] = Field(default_factory=list)

    def blockers(self) -> list[CriticFinding]:
        return [f for f in self.findings if f.severity == "blocker"]

    def by_severity(self) -> list[CriticFinding]:
        order = {"blocker": 0, "concern": 1, "note": 2}
        return sorted(self.findings, key=lambda f: order.get(f.severity, 3))


class NarrativeReport(BaseModel):
    """The prose half of the final report (spec 7.12) -- and only the prose.

    The division here is the one thing that keeps an LLM-written report
    trustworthy: **the model writes sentences, the code supplies numbers.** Every
    figure in the finished document is formatted by ``ml/report.py`` from the
    artifacts, and these fields are woven around them. A report that hallucinates
    a metric is worse than no report, and the cheapest way to guarantee it cannot
    is to never ask a model for one.

    Section 5 built the whole report deterministically for exactly this reason
    and said the Report Agent would replace its prose. This is that replacement:
    the tables, folds and metrics are untouched.
    """

    executive_summary: str = ""
    # What the dataset is like, in prose, for a reader who will not read a table.
    data_story: str = ""
    # Why this model, and what its score does and does not establish.
    model_story: str = ""
    # What the reader should do next, in their own terms rather than the
    # pipeline's. Distinct from the critic's recommendations, which are about the
    # run; these are about the problem.
    recommendations: list[str] = Field(default_factory=list)

    source: Literal["llm", "default"] = "default"

    def is_empty(self) -> bool:
        return not (self.executive_summary or self.data_story or self.model_story)


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
