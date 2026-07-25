"""Data Cleaning -- deterministic code, no LLM (spec 7.4, build-plan Section 5).

Removes what is structurally wrong with a dataset: columns the user excluded,
repeated rows, columns stored as the wrong type, columns that never change, and
columns that are mostly empty.

**What this module deliberately does not do is fill in missing values.** That
looks like cleaning's job and it is the single easiest way to destroy the
project's credibility. Computing a median or a modal category here means
computing it over every row -- including the rows that will later land in a test
fold -- so the training data quietly absorbs information about the test data and
every score afterwards is inflated (spec 8). Imputation is therefore a *step in
the unfitted pipeline* (``preprocessing.py``), fitted separately inside each
fold. Cleaning counts the gaps, reports them, and leaves them alone.

The operations that do happen here are all safe under that rule, because none of
them learns a value from the data that is then applied back to it:

* dropping a column is a decision about the column's *existence*, identical for
  every row and every fold;
* dropping duplicate or unlabelled rows removes records entirely, so nothing
  crosses from one fold to another;
* a dtype correction is a re-reading of values already present, not a new value
  derived from other rows.

Column types come from the schema report rather than being re-sniffed here, so
there is exactly one place in the codebase that decides what a column is
(``services/profiling.py``) and cleaning simply acts on its verdict.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass

import pandas as pd

from app.agents.schema_models import SchemaReport, TaskType
from app.core.config import get_settings
from app.ml.contracts import CleaningReport, DroppedColumn, DtypeCorrection, PlannerPlan

logger = logging.getLogger(__name__)

AGENT_NAME = "cleaning"

# Fraction of a column's values that must survive ``to_numeric`` / ``to_datetime``
# for the conversion to be accepted. Not 1.0: a genuinely numeric column often
# carries a couple of "n/a" or "unknown" strings, and turning those into NaN is
# the correct reading. Below this, the column is text that happens to contain
# some numbers and coercing it would silently delete most of its content.
_COERCION_THRESHOLD = 0.95


class CleaningError(RuntimeError):
    """Cleaning left nothing to model with. Raised with a reason a user can act on."""


@dataclass(slots=True)
class CleaningResult:
    """The cleaned frame and the record of what was done to it."""

    frame: pd.DataFrame
    report: CleaningReport


def clean_frame(
    frame: pd.DataFrame,
    schema: SchemaReport,
    *,
    target: str,
    task_type: TaskType,
    excluded: Sequence[str] = (),
    plan: PlannerPlan | None = None,
    max_null_rate: float | None = None,
) -> CleaningResult:
    """Clean ``frame`` for modelling, returning the frame and a report.

    ``excluded`` is the user's confirmed exclusion list (PII columns default into
    it at the checkpoint). The target column is never dropped by any rule here --
    if it were, the job would fail later with a confusing error about a missing
    column instead of an honest one about the target.
    """
    if target not in frame.columns:
        raise CleaningError(f"Target column {target!r} is not in the dataset.")

    plan = plan or PlannerPlan()
    if max_null_rate is None:
        max_null_rate = get_settings().max_null_column_rate

    n_rows_before, n_columns_before = frame.shape
    dropped: list[DroppedColumn] = []
    corrections: list[DtypeCorrection] = []

    frame = frame.copy()

    # ---- 1. Columns the user excluded --------------------------------------
    # First, so that later steps never spend effort on a column that is leaving
    # anyway -- and so duplicate detection below ignores excluded columns, which
    # is what makes two rows differing only in a dropped ID count as duplicates.
    for name in excluded:
        if name == target:
            logger.warning("Ignoring exclusion of the target column %r", name)
            continue
        if name in frame.columns:
            frame = frame.drop(columns=[name])
            dropped.append(DroppedColumn(name=name, reason="excluded at the schema checkpoint"))

    # ---- 2. Rows with no label ---------------------------------------------
    # A row with no target teaches nothing and cannot be scored. Dropping rows
    # is fold-safe: the record is gone from every split, not moved between them.
    missing_target = int(frame[target].isna().sum())
    if missing_target:
        frame = frame[frame[target].notna()]

    # ---- 3. Duplicate rows -------------------------------------------------
    duplicates_removed = 0
    if plan.drop_duplicate_rows:
        duplicates_removed = int(frame.duplicated().sum())
        if duplicates_removed:
            frame = frame.drop_duplicates()

    # ---- 4. Dtype corrections ----------------------------------------------
    frame, corrections = _correct_dtypes(frame, schema, target=target, task_type=task_type)

    # ---- 5. Constant columns -----------------------------------------------
    # A column with one distinct value (or none) cannot separate anything; it
    # only slows training and adds a meaningless entry to every explanation.
    for name in [c for c in frame.columns if c != target]:
        if int(frame[name].nunique(dropna=True)) <= 1:
            frame = frame.drop(columns=[name])
            dropped.append(DroppedColumn(name=name, reason="constant (one or no distinct value)"))

    # ---- 6. Mostly-empty columns -------------------------------------------
    if plan.drop_high_null_columns and len(frame):
        for name in [c for c in frame.columns if c != target]:
            null_rate = float(frame[name].isna().mean())
            if null_rate > max_null_rate:
                frame = frame.drop(columns=[name])
                dropped.append(
                    DroppedColumn(name=name, reason=f"{null_rate:.0%} of values missing")
                )

    # ---- 7. Report the gaps; do not fill them ------------------------------
    remaining = {
        str(name): int(count) for name, count in frame.isna().sum().items() if int(count) > 0
    }

    _guard_usable(frame, target=target)

    report = CleaningReport(
        n_rows_before=int(n_rows_before),
        n_rows_after=int(frame.shape[0]),
        n_columns_before=int(n_columns_before),
        n_columns_after=int(frame.shape[1]),
        duplicate_rows_removed=duplicates_removed,
        missing_target_rows_removed=missing_target,
        dropped_columns=dropped,
        dtype_corrections=corrections,
        missing_values_left_to_the_pipeline=remaining,
    )
    logger.info(
        "Cleaning: %d->%d rows, %d->%d columns, %d gaps left for the fold-fitted imputer",
        report.n_rows_before,
        report.n_rows_after,
        report.n_columns_before,
        report.n_columns_after,
        sum(remaining.values()),
    )
    return CleaningResult(frame=frame, report=report)


def _correct_dtypes(
    frame: pd.DataFrame,
    schema: SchemaReport,
    *,
    target: str,
    task_type: TaskType,
) -> tuple[pd.DataFrame, list[DtypeCorrection]]:
    """Re-read columns that CSV forced into strings but are really dates or numbers.

    CSV has no types, so a numeric column containing one stray ``"unknown"``
    arrives as ``object`` and every model downstream would treat it as a category.
    Profiling already decided what each column is; this applies that verdict, with
    one deliberate restriction.

    Numeric coercion is attempted only on columns profiling called **text**, i.e.
    high-cardinality ones. A column of hundreds of distinct numeric strings is a
    number. A *low*-cardinality one -- profiling calls those categorical -- is
    left alone even if every value parses, because a five-value code column like
    a postal district or a rating band is genuinely a category, and one-hot
    encoding is the better treatment for it either way. Guessing wrong in that
    direction is cheap; guessing wrong in the other silently imposes an ordering
    on labels that have none.
    """
    corrections: list[DtypeCorrection] = []

    for name in frame.columns:
        profile = schema.column(name)
        if profile is None or not _is_object_like(frame[name]):
            continue

        if name == target:
            # The target's type is the user's decision, not profiling's: a
            # numeric-looking target is only a number if they said regression.
            # Coercing a confirmed classification target would turn its labels
            # into floats and change what is being predicted.
            if task_type == "regression":
                converted = pd.to_numeric(frame[name], errors="coerce")
                if _kept_enough(frame[name], converted):
                    corrections.append(_record(name, frame[name], converted))
                    frame[name] = converted
            continue

        if profile.semantic_type == "datetime":
            converted = pd.to_datetime(frame[name], errors="coerce", format="mixed")
        elif profile.semantic_type == "text":
            converted = pd.to_numeric(frame[name], errors="coerce")
        else:
            continue

        if _kept_enough(frame[name], converted):
            corrections.append(_record(name, frame[name], converted))
            frame[name] = converted

    return frame, corrections


def _is_object_like(series: pd.Series) -> bool:
    """True for a column pandas parsed as strings/objects, i.e. one worth re-reading."""
    return pd.api.types.is_object_dtype(series) or isinstance(series.dtype, pd.StringDtype)


def _kept_enough(original: pd.Series, converted: pd.Series) -> bool:
    """Accept a conversion only if it did not quietly blank most of the column."""
    had = int(original.notna().sum())
    if had == 0:
        return False
    return (int(converted.notna().sum()) / had) >= _COERCION_THRESHOLD


def _record(name: str, before: pd.Series, after: pd.Series) -> DtypeCorrection:
    return DtypeCorrection(name=name, from_dtype=str(before.dtype), to_dtype=str(after.dtype))


def _guard_usable(frame: pd.DataFrame, *, target: str) -> None:
    """Fail now, with a readable reason, rather than deep inside scikit-learn.

    A dataset that cleaning empties out is a real outcome (a tiny file, every
    column constant, every row unlabelled). Spec 10 asks for failures the user
    can act on, and "no usable feature columns" is far more actionable than the
    shape error sklearn would raise three stages later.
    """
    features = [c for c in frame.columns if c != target]
    if not features:
        raise CleaningError(
            "No usable feature columns remain after cleaning -- every column was "
            "excluded, constant, or almost entirely empty."
        )
    if frame.shape[0] < 2:
        raise CleaningError(
            f"Only {frame.shape[0]} row(s) remain after cleaning; not enough to train on."
        )
    if int(frame[target].nunique(dropna=True)) < 2:
        raise CleaningError(
            f"The target column {target!r} has only one distinct value after cleaning, "
            "so there is nothing to predict."
        )
