"""Deterministic dataset profiling -- the half of schema detection that needs no LLM.

Pure functions over a DataFrame: no database, no storage, no network. This runs
on every upload, synchronously, before the LLM is ever consulted (spec 7.1), and
it is what makes schema detection robust -- if the LLM is unavailable or slow,
the user still gets a complete, editable schema built entirely from these rules.

The LLM later *enriches* what is produced here (column meanings, semantic PII);
it never replaces it. Everything here is exact and directly testable, which is
where the build plan wants test coverage concentrated.
"""

from __future__ import annotations

import io
import re

import pandas as pd

from app.agents.schema_models import (
    ClassBalance,
    ColumnProfile,
    SchemaReport,
    SemanticType,
    TaskType,
)

# Regexes for the PII the spec calls out (7.1). Deliberately conservative:
# better to miss a borderline case and let the user tick the box than to flag
# half the dataset and train them to ignore the warning.
_EMAIL = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
_PHONE = re.compile(r"^\+?[\d\s().-]{7,}$")
_SSN = re.compile(r"^\d{3}-\d{2}-\d{4}$")

# How many non-null values to test per column. A column is what it is in its
# first few hundred rows; scanning a million to confirm "these are emails" is
# wasted work in a synchronous request.
_PII_SAMPLE = 200
# Fraction of sampled values that must match for the column to count as PII.
_PII_THRESHOLD = 0.8
_SAMPLE_VALUES = 5

# A low-cardinality object column is categorical; a high-cardinality one is free
# text. This is the boundary. Also used to decide numeric target task type.
_MAX_CATEGORICAL_UNIQUE = 20


def read_csv_frame(data: bytes) -> pd.DataFrame:
    """Parse validated CSV bytes into a DataFrame for profiling.

    The upload route has already run ``inspect_csv`` on these bytes, so parsing
    is known to succeed; this just re-materialises the frame that validation
    discarded, keeping profiling decoupled from the validation step.
    """
    return pd.read_csv(io.BytesIO(data))


def profile_dataset(frame: pd.DataFrame) -> SchemaReport:
    """Build the deterministic half of the schema report."""
    n_rows = int(frame.shape[0])
    columns = [_profile_column(frame[name], n_rows) for name in frame.columns]

    target = _suggest_target(columns)
    task_type = _infer_task_type(columns, target)
    balance = _class_balance(frame, target) if target and task_type == "classification" else None

    return SchemaReport(
        n_rows=n_rows,
        n_columns=int(frame.shape[1]),
        columns=columns,
        suggested_target=target,
        task_type=task_type,
        class_balance=balance,
        llm_enriched=False,
    )


def _profile_column(series: pd.Series, n_rows: int) -> ColumnProfile:
    name = str(series.name)
    null_count = int(series.isna().sum())
    semantic = _semantic_type(series)
    pii_type = _detect_pii(series) if semantic in ("text", "categorical", "numeric") else None

    return ColumnProfile(
        name=name,
        semantic_type=semantic,
        pandas_dtype=str(series.dtype),
        n_unique=int(series.nunique(dropna=True)),
        null_count=null_count,
        null_rate=(null_count / n_rows) if n_rows else 0.0,
        sample_values=_sample_values(series),
        is_pii=pii_type is not None,
        pii_type=pii_type,
        # Exclude PII from modelling by default -- the leakage-and-ethics-safe
        # choice is the one the user has to actively undo.
        exclude=pii_type is not None,
    )


def _semantic_type(series: pd.Series) -> SemanticType:
    """Map a pandas column to one of five modelling-relevant kinds."""
    non_null = series.dropna()
    if pd.api.types.is_bool_dtype(series):
        return "boolean"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    if pd.api.types.is_numeric_dtype(series):
        # A 0/1 or single-value integer column is really a boolean flag.
        uniques = set(non_null.unique().tolist())
        if uniques <= {0, 1}:
            return "boolean"
        return "numeric"

    # Object/string. Distinguish a categorical (few repeated values) from free
    # text (mostly unique). An empty column defaults to categorical.
    n_unique = int(non_null.nunique())
    if n_unique == 0:
        return "categorical"
    if _looks_like_datetime(non_null):
        return "datetime"
    if n_unique <= _MAX_CATEGORICAL_UNIQUE or n_unique / max(1, len(non_null)) < 0.5:
        return "categorical"
    return "text"


def _looks_like_datetime(non_null: pd.Series) -> bool:
    """True when a string column parses cleanly as dates on a sample."""
    sample = non_null.astype(str).str.strip().head(_PII_SAMPLE)
    if sample.empty:
        return False

    # A column of bare digits is not a date. ``pd.to_datetime`` reads "1000" as
    # the year 1000 and "20240101" as a date, so without this guard any column of
    # four- to eight-digit numbers stored as text -- an amount, a postcode, a
    # product code -- is classified as a datetime. From Section 5 that is not
    # cosmetic: cleaning acts on this verdict and converts the column, and
    # preprocessing then drops datetimes, so the column's signal is lost
    # silently. Requiring a separator costs the compact "20240101" form, which is
    # rarer than integer columns by a wide margin and would be caught as numeric.
    if not sample.str.contains(r"[-/:\s.]").any():
        return False

    parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
    return parsed.notna().mean() >= 0.9


def _detect_pii(series: pd.Series) -> str | None:
    """Return the PII kind a column's values match, or None."""
    sample = series.dropna().astype(str).head(_PII_SAMPLE)
    if sample.empty:
        return None
    for label, pattern in (("email", _EMAIL), ("ssn", _SSN), ("phone", _PHONE)):
        if (sample.str.match(pattern)).mean() >= _PII_THRESHOLD:
            # Phone is the loosest pattern and would swallow plain integers; only
            # trust it on genuinely string-typed data, not numeric columns.
            if label == "phone" and pd.api.types.is_numeric_dtype(series):
                continue
            return label
    return None


def _sample_values(series: pd.Series) -> list:
    """A few representative non-null values, JSON-safe, for the UI and the LLM."""
    values = series.dropna().unique()[:_SAMPLE_VALUES]
    out = []
    for value in values:
        out.append(value.item() if hasattr(value, "item") else value)
    return out


def _suggest_target(columns: list[ColumnProfile]) -> str | None:
    """Heuristic target guess, later refined by the LLM.

    Convention puts the label last, so the last non-PII, non-constant column is
    a good first guess -- the user (and the LLM) can override it. Constant and
    PII columns are skipped because neither is ever the thing you predict.
    """
    for col in reversed(columns):
        if col.is_pii or col.n_unique <= 1:
            continue
        return col.name
    return columns[-1].name if columns else None


def _infer_task_type(columns: list[ColumnProfile], target: str | None) -> TaskType | None:
    """Classification vs regression, from the target column's shape."""
    if target is None:
        return None
    profile = next((c for c in columns if c.name == target), None)
    if profile is None:
        return None
    if profile.semantic_type in ("categorical", "boolean", "text"):
        return "classification"
    # Numeric target: few distinct values reads as classes (e.g. a 1-5 rating),
    # many distinct values reads as a continuous quantity.
    if profile.n_unique <= _MAX_CATEGORICAL_UNIQUE:
        return "classification"
    return "regression"


def _class_balance(frame: pd.DataFrame, target: str) -> ClassBalance | None:
    """Class counts and an imbalance ratio for a classification target."""
    counts = frame[target].value_counts(dropna=True)
    if counts.empty:
        return None
    as_dict = {str(k): int(v) for k, v in counts.items()}
    largest, smallest = int(counts.max()), int(counts.min())
    ratio = largest / smallest if smallest else float("inf")
    # A 1.5:1 majority is normal; beyond it SMOTE (Section 7) starts to matter.
    return ClassBalance(counts=as_dict, imbalance_ratio=ratio, imbalanced=ratio > 1.5)
