"""The schema-report contract, shared by profiling, the agent, the API and the UI.

One representation, used everywhere: deterministic profiling fills most of it,
the LLM enriches a few fields, it serialises straight to the ``schema_report.json``
artifact, and it is the body of the upload response the confirmation screen
renders. Being Pydantic means the artifact on disk and the API payload can never
drift apart.

The user's *confirmed* edits come back as ``ConfirmedSchema`` -- deliberately a
separate, smaller shape, because what the user may change (target, task type,
PII flags, exclusions) is a subset of everything profiling reports.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

SemanticType = Literal["numeric", "categorical", "boolean", "datetime", "text"]
TaskType = Literal["classification", "regression"]
# Kept as an open string rather than an enum: regex covers email/phone/ssn, the
# LLM may name others (e.g. "full_name", "address"), and a new category should
# not require a code change to store.
PIIType = str


class ColumnProfile(BaseModel):
    """Everything known about a single column."""

    name: str
    semantic_type: SemanticType
    pandas_dtype: str
    n_unique: int
    null_count: int
    null_rate: float = Field(..., ge=0.0, le=1.0)
    sample_values: list[Any] = Field(default_factory=list)

    # PII: regex sets these deterministically; the LLM may additionally flag
    # semantic PII (a "name" column no regex catches). ``exclude`` defaults on
    # for PII so the safe choice is the default -- the user opts back in.
    is_pii: bool = False
    pii_type: PIIType | None = None
    exclude: bool = False

    # Filled by the LLM pass; ``None`` when detection ran deterministic-only.
    meaning: str | None = None


class ClassBalance(BaseModel):
    """Target class distribution, for the imbalance check (spec 7.1)."""

    counts: dict[str, int]
    # max class / min class. 1.0 is perfectly balanced; large is skewed.
    imbalance_ratio: float
    imbalanced: bool


class SchemaReport(BaseModel):
    """The full picture shown at the human checkpoint."""

    n_rows: int
    n_columns: int
    columns: list[ColumnProfile]
    suggested_target: str | None = None
    task_type: TaskType | None = None
    class_balance: ClassBalance | None = None
    # True when the LLM pass ran; False when it was skipped or failed and the
    # report is deterministic-only. Surfaced so the UI can say which it is.
    llm_enriched: bool = False

    def column_names(self) -> list[str]:
        return [c.name for c in self.columns]

    def column(self, name: str) -> ColumnProfile | None:
        return next((c for c in self.columns if c.name == name), None)


# ---- The LLM's slice of the work -------------------------------------------
# Kept small and separate (spec 6.2: "LLM decides, code executes"). The model
# only assigns meanings, names a target, picks the task type, and flags semantic
# PII -- it never sees or transforms data.


class LLMColumnInsight(BaseModel):
    name: str
    meaning: str
    is_pii: bool = False
    pii_type: PIIType | None = None


class LLMSchemaInference(BaseModel):
    columns: list[LLMColumnInsight]
    suggested_target: str
    task_type: TaskType


# ---- What the user confirms ------------------------------------------------


class ConfirmedColumn(BaseModel):
    # Unknown keys are rejected rather than ignored. ``exclude`` decides whether
    # a column reaches the model, so a caller who guesses the name -- ``include:
    # false`` is the obvious guess, and inverted -- would otherwise get a 200 and
    # silently train on the PII they meant to withhold. A 422 naming the field is
    # the only safe failure here.
    model_config = ConfigDict(extra="forbid")

    name: str
    is_pii: bool = False
    exclude: bool = False


class ConfirmedSchema(BaseModel):
    """The user's approved schema, posted to launch the (future) pipeline."""

    model_config = ConfigDict(extra="forbid")

    target_column: str
    task_type: TaskType
    columns: list[ConfirmedColumn] = Field(default_factory=list)
    # Naming a column here switches cross-validation to time-ordered folds, so
    # each fold is scored on rows that came *after* the ones it trained on. It is
    # a decision rather than a detection: a dataset can hold a date that is not an
    # event time -- a date of birth, a renewal date -- and ordering by one of
    # those would be worse than not ordering at all. Left unset, folds are random
    # exactly as before.
    time_column: str | None = None

    def excluded(self) -> list[str]:
        return [c.name for c in self.columns if c.exclude]
