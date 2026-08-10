"""Evaluation -- the metric set for the task type, and the cross-fold summary (spec 7.8).

Two jobs, kept apart on purpose. ``fold_metrics`` scores a single fold's held-out
predictions and is called from inside the cross-validation loop;
``build_evaluation_report`` folds those per-fold numbers into the
``evaluation_report.json`` artifact. Splitting them this way keeps this module
free of any dependency on the modelling code (so there is no import cycle) and
means the metric definitions can be tested directly on arrays, with no model
involved at all.

The metric sets are the spec's: accuracy, precision, recall, F1, ROC-AUC and
PR-AUC for classification; MAE, MSE, RMSE and R² for regression.

Two choices worth stating plainly, because they change how a number reads:

* **Macro averaging** for precision/recall/F1, not weighted. Macro treats every
  class as equally important, so a model that ignores a rare class is punished
  for it. Weighted averaging would hide exactly the failure this project cares
  about -- the imbalanced-target case SMOTE exists to address (Section 7). The
  metric names say ``_macro`` so nobody has to guess which was used.
* **Metrics that cannot be computed are omitted and explained**, never silently
  reported as zero. ROC-AUC needs both classes present in the held-out fold; on
  a tiny or badly skewed dataset that can fail. A missing number with a stated
  reason is honest, a zero is a lie about the model's performance.

The regression set has since gained two more, median absolute error and MAPE,
for a reason worth stating up front: the spec's four are all absolute or squared
and therefore all describe the same thing on a heavy-tailed target -- its tail.
See ``_regression_metrics``.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)

from app.agents.schema_models import TaskType
from app.ml.contracts import (
    ConcentratedFoldError,
    EvaluationReport,
    FoldScore,
    MetricSummary,
)

logger = logging.getLogger(__name__)

AGENT_NAME = "evaluation"

# The metric a reader should look at first, per task type. F1 rather than
# accuracy for classification: on a 95/5 split, accuracy rewards predicting the
# majority class every time, and macro F1 does not.
PRIMARY_METRIC: dict[str, str] = {
    "classification": "f1_macro",
    "regression": "r2",
}


def fold_metrics(
    *,
    y_true: Any,
    y_pred: Any,
    y_proba: np.ndarray | None,
    classes: Sequence[Any],
    task_type: TaskType,
) -> tuple[dict[str, float], list[str]]:
    """Score one fold's held-out predictions.

    Returns the metrics that could be computed and a list of warnings for the
    ones that could not.
    """
    if task_type == "classification":
        return _classification_metrics(y_true, y_pred, y_proba, classes)
    return _regression_metrics(y_true, y_pred)


def _classification_metrics(
    y_true: Any,
    y_pred: Any,
    y_proba: np.ndarray | None,
    classes: Sequence[Any],
) -> tuple[dict[str, float], list[str]]:
    warnings: list[str] = []
    metrics: dict[str, float] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        # zero_division=0: a class the model never predicts has undefined
        # precision. Scoring that as 0 is the correct reading -- the model found
        # none of them -- and it keeps the fold from raising instead of scoring.
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }

    if y_proba is None or len(classes) < 2:
        warnings.append("ROC-AUC and PR-AUC need class probabilities, which this model lacks.")
        return metrics, warnings

    if len(classes) == 2:
        # Column 1 of predict_proba is the probability of classes_[1], by
        # scikit-learn's ordering. Binarising against that class explicitly
        # avoids relying on label ordering being inferred from strings.
        positive = np.asarray(y_true) == classes[1]
        scores = y_proba[:, 1]
        _try_metric(metrics, warnings, "roc_auc", lambda: roc_auc_score(positive, scores))
        _try_metric(metrics, warnings, "pr_auc", lambda: average_precision_score(positive, scores))
        return metrics, warnings

    _try_metric(
        metrics,
        warnings,
        "roc_auc",
        lambda: roc_auc_score(
            y_true, y_proba, multi_class="ovr", average="macro", labels=list(classes)
        ),
    )
    # Average precision has no unambiguous multiclass definition without
    # choosing a positive class, so it is reported for binary targets only.
    warnings.append("PR-AUC is reported for binary targets only; this target has more classes.")
    return metrics, warnings


def _regression_metrics(y_true: Any, y_pred: Any) -> tuple[dict[str, float], list[str]]:
    """MAE, MSE, RMSE and R² -- plus two the spec's four cannot express.

    All four of the originals are absolute and squared-error-dominated, which on
    a heavy-tailed target means all four describe the tail. A run on listing
    prices reported RMSE 202 against MAE 81 and R² 0.07, and there was no number
    anywhere in the report that let a reader work out whether a typical
    prediction was any good.

    * **Median absolute error** is the typical miss. Where MAE is pulled upward
      by a handful of large errors, this is not, so the gap between the two is
      itself the diagnosis.
    * **MAPE** puts the error on the target's own scale. "Out by 81" means
      nothing without knowing that the median listing costs 106; "out by 43%"
      means something on its own.

    Neither replaces R², which stays the primary metric -- they sit alongside it.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    errors = np.abs(y_true - y_pred)

    mse = float(mean_squared_error(y_true, y_pred))
    metrics = {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "median_ae": float(np.median(errors)),
        "mse": mse,
        "rmse": math.sqrt(mse),
        "r2": float(r2_score(y_true, y_pred)),
    }

    warnings: list[str] = []
    # A percentage of zero is not a large error, it is an undefined one, and
    # scikit-learn's own MAPE quietly substitutes an epsilon that turns a single
    # zero row into a value in the billions. Those rows are excluded and counted
    # instead -- a percentage over 99% of the data, honestly labelled, beats a
    # meaningless number over all of it.
    usable = y_true != 0
    if usable.any():
        metrics["mape"] = float(np.mean(errors[usable] / np.abs(y_true[usable])))
        if not usable.all():
            # Deliberately without the row count. This runs per fold, and each
            # fold holds a different number of zero-valued rows, so a counted
            # message defeats the caller's deduplication and the report grows one
            # near-identical caveat per fold. The qualification is what matters;
            # the exact count per fold is not something a reader can act on.
            warnings.append(
                "MAPE excludes held-out rows whose true value is zero, since a "
                "percentage of zero is undefined. Every other metric covers them."
            )
    else:
        warnings.append("MAPE could not be computed: every held-out target value was zero.")

    return metrics, warnings


def _try_metric(
    metrics: dict[str, float],
    warnings: list[str],
    name: str,
    compute,
) -> None:
    """Record a metric, or explain why it is absent.

    ROC-AUC raises when a fold's held-out rows happen to contain a single class.
    That is a property of the data, not a bug, so it is reported as a warning and
    the metric is left out of the report rather than being invented.
    """
    try:
        value = float(compute())
    except ValueError as exc:
        warnings.append(f"{name} could not be computed for at least one fold: {exc}")
        return
    if math.isfinite(value):
        metrics[name] = value
    else:
        warnings.append(f"{name} was not a finite number for at least one fold.")


def build_evaluation_report(
    folds: Sequence[FoldScore],
    *,
    task_type: TaskType,
    target_column: str,
    model_name: str,
    n_folds: int,
    cv_strategy: str,
    n_rows: int,
    n_features: int,
    warnings: Sequence[str] = (),
    concentrated_fold_error: ConcentratedFoldError | None = None,
    shuffled: bool = False,
    random_seed: int | None = None,
) -> EvaluationReport:
    """Aggregate per-fold metrics into the report artifact.

    A metric is summarised over the folds that actually produced it, so one fold
    failing to yield ROC-AUC does not discard the other four.
    """
    if not folds:
        raise ValueError("Cannot build an evaluation report from zero folds.")

    metrics = _summarise(folds)
    primary = PRIMARY_METRIC.get(task_type, "")
    if primary and primary not in metrics:
        # Should not happen -- the primary metric is always computable -- but a
        # report whose headline metric is missing must not point at a gap.
        primary = next(iter(metrics), "")

    report = EvaluationReport(
        task_type=task_type,
        target_column=target_column,
        model_name=model_name,
        n_folds=n_folds,
        cv_strategy=cv_strategy,
        n_rows=n_rows,
        n_features=n_features,
        folds=list(folds),
        metrics=metrics,
        primary_metric=primary,
        warnings=list(warnings),
        concentrated_fold_error=concentrated_fold_error,
        shuffled=shuffled,
        random_seed=random_seed,
    )
    logger.info(
        "Evaluation: %s = %.4f across %d folds",
        report.primary_metric,
        report.primary_score() or float("nan"),
        n_folds,
    )
    return report


def _summarise(folds: Sequence[FoldScore]) -> dict[str, MetricSummary]:
    """Mean and spread of each metric across folds.

    The standard deviation is the reason cross-validation was worth running: a
    mean F1 of 0.80 that swings ±0.15 between folds is a far weaker result than
    the same mean at ±0.01, and a single train/test split would not have shown
    the difference. Population standard deviation (ddof=0) -- these five folds
    are the whole set being described, not a sample of a larger one.
    """
    names = sorted({name for fold in folds for name in fold.metrics})
    summary: dict[str, MetricSummary] = {}
    for name in names:
        values = [fold.metrics[name] for fold in folds if name in fold.metrics]
        if not values:
            continue
        summary[name] = MetricSummary(mean=float(np.mean(values)), std=float(np.std(values)))
    return summary
