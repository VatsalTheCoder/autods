"""Shrinking a run's artifacts into something a prompt can afford (spec 6.3, 10).

The Critic and the Report agents are the only ones in this project that want to
see *everything* — cleaning, EDA, clustering, feature decisions, the leaderboard,
the evaluation, the explanation. Handing them the artifacts is the obvious
implementation and it does not work: the free tier's binding constraint is
**15,000 input tokens per minute**, the two agents run back to back, and the
model's large context window is irrelevant because the limit is on the wire, not
in the model.

Measured rather than assumed: on the nine-column example the raw artifacts come
to about 4,300 tokens, which fits — and is exactly why a demo dataset proves
nothing here. Artifact size grows with *column count*, through the per-column
tables in the EDA, feature and preprocessing reports, and with *encoded* feature
count through the SHAP name mapping. A hundred-column dataset multiplies the
parts that scale and blows the cap.

**The rule this module exists to enforce: shrinking is lossy, and the loss must
be stated.** Cap a 121-row feature table at 20 rows and a critic reading it will
tell you the dataset has 20 features and that you should have used more of them.
It is not wrong to have sent 20 — it is wrong to have sent 20 that *look* like
all of them. So every cap here leaves behind a count of what it dropped, in the
payload the model actually reads.

Degradation is progressive and in a fixed order. Detail is given up before
substance: per-column tables go first, because "here are all 121 columns" is
worth less to a reviewer than the scores, the leakage guarantees and the top
features. What survives to the last tier is what a critic could still write a
useful review from.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Literal

from app.core.llm.rate_limit import estimate_input_tokens
from app.ml.contracts import (
    CleaningReport,
    ClusteringReport,
    EdaReport,
    EvaluationReport,
    ExplainabilityReport,
    FeatureStrategy,
    FinalModelInfo,
    Leaderboard,
    PreprocessingSpec,
)

logger = logging.getLogger(__name__)

# How many decimal places any float keeps on its way into a prompt. A score of
# 0.8123456789012345 costs about ten tokens more than 0.8123 and tells a reader
# nothing further -- the fourth decimal of a cross-validated mean is noise.
_PLACES = 4

DetailLevel = Literal["full", "reduced", "minimal"]

# The order detail is surrendered in. Each tier is a complete, honest summary --
# not a truncation of the one above -- so a critic reading the smallest one is
# working from less, never from something misleading.
_LEVELS: tuple[DetailLevel, ...] = ("full", "reduced", "minimal")

# Per-tier caps on the tables that grow with the dataset's width.
_CAPS: dict[DetailLevel, dict[str, int]] = {
    "full": {"columns": 40, "features": 20, "clusters": 8, "correlations": 6, "models": 6},
    "reduced": {"columns": 12, "features": 12, "clusters": 4, "correlations": 3, "models": 6},
    "minimal": {"columns": 0, "features": 8, "clusters": 0, "correlations": 0, "models": 4},
}


@dataclass(slots=True)
class RunSummary:
    """The compact view of a run, and an honest account of what it cost to get.

    ``notes`` is not decoration. It is the list of things the model is *not*
    being shown, and it goes into the prompt alongside the payload so the model
    can qualify its own conclusions rather than draw confident ones from a
    truncated table.
    """

    payload: dict[str, Any]
    estimated_tokens: int
    detail_level: DetailLevel
    notes: list[str] = field(default_factory=list)

    def as_prompt_block(self) -> str:
        """The payload as JSON, with the omissions stated above it."""
        import json

        header = ""
        if self.notes:
            header = (
                "Some detail was omitted to fit the prompt budget. Treat the "
                "following as known limits of what you can see:\n"
                + "\n".join(f"- {note}" for note in self.notes)
                + "\n\n"
            )
        return header + json.dumps(self.payload, separators=(",", ":"), default=str)


def summarise_run(
    *,
    budget_tokens: int,
    filename: str = "",
    cleaning: CleaningReport | None = None,
    eda: EdaReport | None = None,
    clustering: ClusteringReport | None = None,
    features: FeatureStrategy | None = None,
    preprocessing: PreprocessingSpec | None = None,
    leaderboard: Leaderboard | None = None,
    evaluation: EvaluationReport | None = None,
    explainability: ExplainabilityReport | None = None,
    final_model: FinalModelInfo | None = None,
) -> RunSummary:
    """Build the smallest faithful summary of a run that fits ``budget_tokens``.

    Tries each detail tier in turn and returns the first that fits, so a narrow
    dataset gets the full picture and a wide one degrades rather than failing.
    The last tier is returned even if it overruns: a prompt that is slightly too
    large is throttled by the rate limiter and still succeeds, whereas returning
    nothing would cost the run its critic entirely.
    """
    summary = None
    for level in _LEVELS:
        summary = _summarise_at(
            level,
            filename=filename,
            cleaning=cleaning,
            eda=eda,
            clustering=clustering,
            features=features,
            preprocessing=preprocessing,
            leaderboard=leaderboard,
            evaluation=evaluation,
            explainability=explainability,
            final_model=final_model,
        )
        if summary.estimated_tokens <= budget_tokens:
            break
        logger.info(
            "Run summary at %s detail is ~%d tokens, over the %d budget; reducing",
            level,
            summary.estimated_tokens,
            budget_tokens,
        )

    assert summary is not None  # noqa: S101 - _LEVELS is never empty
    if summary.estimated_tokens > budget_tokens:
        # Nothing left to give up. Said out loud rather than passed off as a fit.
        summary.notes.append(
            "This summary is still larger than the prompt budget even at minimum "
            "detail; the dataset is unusually wide."
        )
        logger.warning(
            "Run summary is ~%d tokens at minimum detail, over the %d budget",
            summary.estimated_tokens,
            budget_tokens,
        )
    logger.info(
        "Run summary: %s detail, ~%d tokens, %d omission(s) stated",
        summary.detail_level,
        summary.estimated_tokens,
        len(summary.notes),
    )
    return summary


def _summarise_at(
    level: DetailLevel,
    *,
    filename: str,
    cleaning: CleaningReport | None,
    eda: EdaReport | None,
    clustering: ClusteringReport | None,
    features: FeatureStrategy | None,
    preprocessing: PreprocessingSpec | None,
    leaderboard: Leaderboard | None,
    evaluation: EvaluationReport | None,
    explainability: ExplainabilityReport | None,
    final_model: FinalModelInfo | None,
) -> RunSummary:
    """One tier's worth of summary, with the omissions it caused recorded."""
    caps = _CAPS[level]
    notes: list[str] = []
    payload: dict[str, Any] = {}

    if filename:
        payload["dataset"] = filename
    if cleaning is not None:
        payload["cleaning"] = _cleaning(cleaning)
    if eda is not None:
        payload["data"] = _eda(eda, caps, notes)
    if clustering is not None:
        payload["clusters"] = _clustering(clustering, caps, notes)
    if features is not None:
        payload["feature_decisions"] = _features(features, caps, notes)
    if preprocessing is not None:
        payload["preparation"] = _preprocessing(preprocessing)
    if leaderboard is not None:
        payload["models_compared"] = _leaderboard(leaderboard, caps, notes)
    if evaluation is not None:
        payload["evaluation"] = _evaluation(evaluation)
    if explainability is not None:
        payload["explanation"] = _explainability(explainability, caps, notes)
    if final_model is not None:
        payload["served_model"] = _final_model(final_model)

    summary = RunSummary(payload=payload, estimated_tokens=0, detail_level=level, notes=notes)
    summary.estimated_tokens = estimate_input_tokens(summary.as_prompt_block())
    return summary


# ---- Per-artifact summaries --------------------------------------------------
#
# Each of these answers "what would a reviewer need from this artifact", not
# "how do I make this artifact smaller". The difference shows in what is kept:
# the cleaning summary keeps the *count* of remaining gaps and drops the
# per-column breakdown, because "12 columns still have gaps, filled inside each
# fold" is the reviewable claim and the column names are not.


def _cleaning(report: CleaningReport) -> dict[str, Any]:
    return _drop_empty(
        {
            "rows_before": report.n_rows_before,
            "rows_after": report.n_rows_after,
            "columns_before": report.n_columns_before,
            "columns_after": report.n_columns_after,
            "duplicate_rows_removed": report.duplicate_rows_removed,
            "rows_dropped_for_missing_target": report.missing_target_rows_removed,
            "columns_dropped": [d.name for d in report.dropped_columns],
            "dtype_corrections": len(report.dtype_corrections),
            "columns_with_gaps_left_to_the_pipeline": len(
                report.missing_values_left_to_the_pipeline
            ),
        }
    )


def _eda(report: EdaReport, caps: dict[str, int], notes: list[str]) -> dict[str, Any]:
    limit = caps["columns"]
    out: dict[str, Any] = {
        "rows": report.n_rows,
        "columns": report.n_columns,
        "target": report.target_column,
    }

    if report.class_balance is not None:
        out["class_balance"] = _drop_empty(
            {
                "counts": report.class_balance.counts,
                "imbalanced": report.class_balance.imbalanced,
                "ratio": _round(report.class_balance.imbalance_ratio),
            }
        )

    correlations = report.top_correlations[: caps["correlations"]]
    if correlations:
        out["strongest_correlations"] = [
            {"between": [p.left, p.right], "r": _round(p.correlation)} for p in correlations
        ]

    # Per-column statistics are the first thing to go: they are the widest table
    # here and the least reviewable. What replaces them is the shape of the
    # problem -- how many columns, how many have gaps, how many hold outliers --
    # which is what a critic actually reasons about.
    gaps = sum(1 for c in report.columns if c.missing)
    outliers = sum(1 for c in report.columns if c.numeric and c.numeric.outlier_count)
    out["columns_with_missing_values"] = gaps
    out["columns_with_outliers"] = outliers

    if limit and report.columns:
        shown = report.columns[:limit]
        out["column_detail"] = [
            _drop_empty(
                {
                    "name": c.name,
                    "type": c.semantic_type,
                    "missing_rate": _round(c.missing_rate),
                    "unique": c.categorical.n_unique if c.categorical else None,
                }
            )
            for c in shown
        ]
        _note_cap(notes, "column statistics", len(shown), len(report.columns))
    elif report.columns:
        notes.append(
            f"Per-column statistics were omitted entirely ({len(report.columns)} columns); "
            "only the totals above are shown."
        )
    return out


def _clustering(report: ClusteringReport, caps: dict[str, int], notes: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {
        "method": report.method,
        "groups": report.k,
        "silhouette": _round(report.silhouette),
        # Carried because it changes how the finding should be read, and a model
        # given only the number may not know where the threshold sits.
        "well_separated": not report.is_weak(),
        "used_as_features": False,
    }
    limit = caps["clusters"]
    if limit and report.profiles:
        shown = report.profiles[:limit]
        out["profiles"] = [
            _drop_empty(
                {
                    "cluster": p.cluster,
                    "share": _round(p.share),
                    "description": p.description,
                }
            )
            for p in shown
        ]
        _note_cap(notes, "cluster profiles", len(shown), len(report.profiles))
    return out


def _features(strategy: FeatureStrategy, caps: dict[str, int], notes: list[str]) -> dict[str, Any]:
    limit = caps["columns"]
    roles: dict[str, int] = {}
    for column in strategy.columns:
        roles[column.role] = roles.get(column.role, 0) + 1

    out: dict[str, Any] = {
        "chosen_by": strategy.source,
        "columns_decided": len(strategy.columns),
        "roles": roles,
        # The two facts that make the "LLM proposes, code checks" claim
        # reviewable. Kept at every tier: they are small and they are the point.
        "invented_columns_rejected": len(strategy.rejected_columns),
        "columns_left_to_defaults": len(strategy.defaulted_columns),
        "decisions_overruled_by_code": [
            {"column": o.column, "asked": o.requested, "used": o.applied, "why": o.reason}
            for o in strategy.overrides[: caps["features"]]
        ],
    }
    if limit and strategy.columns:
        shown = strategy.columns[:limit]
        out["column_detail"] = [
            _drop_empty(
                {
                    "column": c.column,
                    "role": c.role,
                    "impute": c.impute,
                    "encode": c.encode,
                    "scale": c.scale,
                }
            )
            for c in shown
        ]
        _note_cap(notes, "per-column feature decisions", len(shown), len(strategy.columns))
    elif strategy.columns:
        notes.append(
            f"Per-column feature decisions were omitted ({len(strategy.columns)} columns); "
            "the role counts above describe all of them."
        )
    return out


def _preprocessing(spec: PreprocessingSpec) -> dict[str, Any]:
    """Counts, not names. Which columns are numeric matters far less than how many."""
    return _drop_empty(
        {
            "numeric": len(spec.numeric_columns),
            "categorical": len(spec.categorical_columns),
            "ordinal": len(spec.ordinal_columns),
            "datetime": len(spec.datetime_columns),
            "text": len(spec.text_columns),
            "dropped": len(spec.unhandled_columns),
            "strategy_source": spec.strategy_source,
            "feature_selection": spec.feature_selection,
        }
    )


def _leaderboard(board: Leaderboard, caps: dict[str, int], notes: list[str]) -> dict[str, Any]:
    shown = board.entries[: caps["models"]]
    out = {
        "metric": board.primary_metric,
        "folds": board.n_folds,
        "cv_strategy": board.cv_strategy,
        "resampling": board.resampling,
        "ranking": [
            _drop_empty(
                {
                    "rank": e.rank,
                    "model": e.model_name,
                    "score": None if e.error else _round(e.score),
                    "spread": None if e.error else _round(e.std),
                    "error": e.error,
                }
            )
            for e in shown
        ],
    }
    _note_cap(notes, "leaderboard entries", len(shown), len(board.entries))
    return out


def _evaluation(report: EvaluationReport) -> dict[str, Any]:
    return _drop_empty(
        {
            "task": report.task_type,
            "target": report.target_column,
            "model": report.model_name,
            "folds": report.n_folds,
            "cv_strategy": report.cv_strategy,
            "rows": report.n_rows,
            "features": report.n_features,
            "primary_metric": report.primary_metric,
            "metrics": {
                name: {"mean": _round(s.mean), "std": _round(s.std)}
                for name, s in report.metrics.items()
            },
            # Per-fold scores are dropped everywhere: the mean and the spread
            # carry the same information for a reviewer, in a tenth of the space.
            "warnings": report.warnings,
        }
    )


def _explainability(
    report: ExplainabilityReport, caps: dict[str, int], notes: list[str]
) -> dict[str, Any]:
    limit = caps["features"]
    shown = report.global_importance[:limit]
    out: dict[str, Any] = {
        "explainer": report.explainer,
        "rows_explained": report.n_rows_explained,
        "encoded_features": report.n_encoded_features,
        # The number that says whether any of the rest can be trusted.
        "additivity_error": _round(report.additivity_max_error, 6),
        "top_features": [
            _drop_empty(
                {
                    "feature": f.feature,
                    "share": _round(f.share),
                    "direction": f.direction,
                }
            )
            for f in shown
        ],
        "warnings": report.warnings,
    }
    _note_cap(notes, "feature importances", len(shown), len(report.global_importance))
    # The name mapping is the biggest single table in a wide run -- one row per
    # encoded feature -- and a reviewer needs to know it exists, not to read it.
    if report.feature_name_mapping:
        notes.append(
            f"The encoded-to-source feature mapping ({len(report.feature_name_mapping)} "
            "entries) is recorded in the artifact but omitted here."
        )
    return out


def _final_model(info: FinalModelInfo) -> dict[str, Any]:
    return _drop_empty(
        {
            "model": info.model_name,
            "refitted_on_rows": info.n_rows,
            "input_columns": info.n_features,
            "cv_score": _round(info.cv_score) if info.cv_score is not None else None,
            "metric": info.primary_metric,
            # Stated explicitly because it is the thing a critic is most likely
            # to get wrong about this project: the served model has no held-out
            # score, and its number is inherited from cross-validation.
            "note": "Refitted on every row; the score above is the cross-validated estimate.",
            "warnings": info.warnings,
        }
    )


# ---- Shrinking primitives ----------------------------------------------------


def _round(value: float | None, places: int = _PLACES) -> float | None:
    if value is None:
        return None
    try:
        return round(float(value), places)
    except (TypeError, ValueError):  # pragma: no cover - defensive
        return None


def _note_cap(notes: list[str], what: str, shown: int, total: int) -> None:
    """Record a truncation in the words the model will read.

    The whole reason this function exists rather than a bare slice: a capped list
    that does not announce itself is indistinguishable from a complete one, and a
    reviewer told "here are the features" will reason about the set as if it were
    all of them.
    """
    if total > shown:
        notes.append(f"Showing {shown} of {total} {what}.")


def _drop_empty(data: dict[str, Any]) -> dict[str, Any]:
    """Remove keys whose values carry nothing, to stop paying tokens for them.

    ``0`` and ``False`` are kept: "zero columns were dropped" and "the classes
    are not imbalanced" are both findings a reviewer needs, and dropping them
    would leave the model to guess from an absence.
    """
    return {k: v for k, v in data.items() if v is not None and v != "" and v != [] and v != {}}
