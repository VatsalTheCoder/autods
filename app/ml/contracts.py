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
from app.agents.schema_models import TaskType

# ---- The planner's output ---------------------------------------------------


class PlannerPlan(BaseModel):
    """The plan the LLM writes, and the flags the pipeline actually obeys.

    Deliberately tiny (build-plan Section 5: "one or two on/off switches").
    Every field here is read by a later node -- there are no aspirational
    settings nothing consumes, because a plan field that changes no behaviour is
    just a lie in an artifact. The model roster and SMOTE toggle the spec's
    Planner (7.3) also owns arrive with Sections 7 and 8, when something exists
    to switch between.
    """

    drop_duplicate_rows: bool = Field(
        default=True,
        description="Remove exactly-repeated rows before modelling.",
    )
    drop_high_null_columns: bool = Field(
        default=True,
        description="Drop columns that are mostly empty rather than imputing them.",
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
