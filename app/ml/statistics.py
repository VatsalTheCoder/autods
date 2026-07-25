"""Descriptive statistics -- the deterministic half of EDA (spec 7.5, Section 6).

Pure functions over a DataFrame: no database, no storage, no LLM. Everything here
is exact and directly checkable against the data, which is where the build plan
wants test coverage concentrated.

This runs on the **cleaned** frame, not the raw upload, so the numbers describe
what the model actually saw. That is the more useful reading -- a correlation
computed over rows that cleaning went on to drop describes a dataset that was
never modelled.

Two deliberate non-actions:

* **Outliers are counted, never removed.** The spec lists outlier detection under
  cleaning, but an outlier is frequently the most interesting row in a dataset --
  the fraudulent transaction, the failing sensor. Removing them is a modelling
  decision with real consequences, so this reports how many there are and leaves
  the decision visible rather than quietly making it.
* **Correlation is computed between numeric columns only.** Pearson's r on
  one-hot encoded categories measures something, but not anything a reader would
  take it to mean.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from app.agents.schema_models import ClassBalance, TaskType
from app.core.config import get_settings
from app.ml.contracts import (
    CategorySummary,
    ColumnStatistics,
    CorrelationPair,
    EdaReport,
    NumericSummary,
)

logger = logging.getLogger(__name__)

AGENT_NAME = "eda"

# How many of the commonest values to record per categorical column.
_TOP_VALUES = 10
# How many correlation pairs to surface. The extremes are the story; the middle
# of a correlation matrix is noise.
_TOP_CORRELATIONS = 10
# Below this absolute r, two columns are not meaningfully related and listing
# them would pad the report with nothing.
_MIN_CORRELATION = 0.1
# The conventional Tukey fence. Not tuned -- it is a stated convention, and a
# bespoke threshold here would be a number nobody could interpret.
_IQR_MULTIPLIER = 1.5


def compute_statistics(
    frame: pd.DataFrame,
    *,
    target: str,
    task_type: TaskType,
) -> EdaReport:
    """Summarise every column, plus correlations and class balance."""
    warnings: list[str] = []
    columns = [_summarise_column(frame[name]) for name in frame.columns]

    report = EdaReport(
        n_rows=int(frame.shape[0]),
        n_columns=int(frame.shape[1]),
        target_column=target,
        columns=columns,
        top_correlations=top_correlations(frame),
        class_balance=(_class_balance(frame, target) if task_type == "classification" else None),
        warnings=warnings,
    )
    logger.info(
        "EDA statistics: %d columns summarised, %d correlation pair(s) above %.1f",
        len(columns),
        len(report.top_correlations),
        _MIN_CORRELATION,
    )
    return report


def _summarise_column(series: pd.Series) -> ColumnStatistics:
    """One column's statistics, branching on whether it is a number."""
    n_rows = int(len(series))
    missing = int(series.isna().sum())
    numeric = None
    categorical = None

    if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
        numeric = _numeric_summary(series)
        semantic = "numeric"
    elif pd.api.types.is_bool_dtype(series):
        # Booleans are summarised as categories: a mean of 0.62 over True/False
        # is harder to read than the two counts.
        categorical = _category_summary(series)
        semantic = "boolean"
    elif pd.api.types.is_datetime64_any_dtype(series):
        categorical = _category_summary(series)
        semantic = "datetime"
    else:
        categorical = _category_summary(series)
        semantic = "categorical"

    return ColumnStatistics(
        name=str(series.name),
        semantic_type=semantic,
        count=n_rows - missing,
        missing=missing,
        missing_rate=(missing / n_rows) if n_rows else 0.0,
        numeric=numeric,
        categorical=categorical,
    )


def _numeric_summary(series: pd.Series) -> NumericSummary:
    values = series.dropna()
    if values.empty:
        zero = float("nan")
        return NumericSummary(
            mean=zero, std=zero, minimum=zero, q1=zero, median=zero, q3=zero, maximum=zero
        )

    q1 = float(values.quantile(0.25))
    q3 = float(values.quantile(0.75))
    return NumericSummary(
        mean=float(values.mean()),
        # ddof=0: describing the column in hand, not inferring about a population.
        std=float(values.std(ddof=0)),
        minimum=float(values.min()),
        q1=q1,
        median=float(values.median()),
        q3=q3,
        maximum=float(values.max()),
        outlier_count=_count_outliers(values, q1, q3),
    )


def _count_outliers(values: pd.Series, q1: float, q3: float) -> int:
    """Rows outside the Tukey fences. Counted only -- see the module docstring."""
    iqr = q3 - q1
    if iqr <= 0:
        # A column where at least half the values are identical has no meaningful
        # spread to measure outliers against.
        return 0
    low = q1 - _IQR_MULTIPLIER * iqr
    high = q3 + _IQR_MULTIPLIER * iqr
    return int(((values < low) | (values > high)).sum())


def _category_summary(series: pd.Series) -> CategorySummary:
    counts = series.dropna().astype(str).value_counts().head(_TOP_VALUES)
    return CategorySummary(
        n_unique=int(series.nunique(dropna=True)),
        top_values={str(k): int(v) for k, v in counts.items()},
    )


def top_correlations(frame: pd.DataFrame) -> list[CorrelationPair]:
    """The strongest absolute Pearson correlations between numeric columns.

    Each unordered pair appears once. Self-correlations are excluded, as are
    pairs below a floor where "correlated" would overstate the relationship.
    """
    numeric = frame.select_dtypes(include=[np.number])
    if numeric.shape[1] < 2:
        return []

    matrix = numeric.corr(numeric_only=True)
    pairs: list[CorrelationPair] = []
    seen: set[frozenset[str]] = set()

    for left in matrix.columns:
        for right in matrix.columns:
            if left == right:
                continue
            key = frozenset((str(left), str(right)))
            if key in seen:
                continue
            value = matrix.loc[left, right]
            if pd.isna(value) or abs(float(value)) < _MIN_CORRELATION:
                continue
            seen.add(key)
            pairs.append(
                CorrelationPair(left=str(left), right=str(right), correlation=float(value))
            )

    pairs.sort(key=lambda p: abs(p.correlation), reverse=True)
    return pairs[:_TOP_CORRELATIONS]


def _class_balance(frame: pd.DataFrame, target: str) -> ClassBalance | None:
    """Recomputed on the cleaned frame, not reused from the upload-time report.

    Cleaning drops unlabelled and duplicated rows, so the balance the model faced
    can genuinely differ from the one profiling saw at upload.
    """
    if target not in frame.columns:
        return None
    counts = frame[target].value_counts(dropna=True)
    if counts.empty:
        return None
    largest, smallest = int(counts.max()), int(counts.min())
    ratio = largest / smallest if smallest else float("inf")
    return ClassBalance(
        counts={str(k): int(v) for k, v in counts.items()},
        imbalance_ratio=ratio,
        imbalanced=ratio > 1.5,
    )


def heatmap_columns(frame: pd.DataFrame) -> list[str]:
    """Numeric columns worth putting on a correlation heatmap.

    Capped: a grid of eighty columns is an unreadable smear, and the caller says
    so in the report rather than emitting something nobody can use. The most
    variable columns are kept, since a near-constant column's correlations are
    dominated by noise.
    """
    numeric = frame.select_dtypes(include=[np.number])
    limit = get_settings().max_heatmap_columns
    if numeric.shape[1] <= limit:
        return [str(c) for c in numeric.columns]
    spread = numeric.std(ddof=0).sort_values(ascending=False)
    return [str(c) for c in spread.head(limit).index]
