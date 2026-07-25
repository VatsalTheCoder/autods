"""Tests for the metric definitions and the cross-fold summary.

Pure arrays in, numbers out -- no model, no pipeline. The behaviour worth pinning
down hardest is what happens when a metric *cannot* be computed: it must be
omitted with an explanation, never reported as zero, because a zero ROC-AUC reads
as "the model is useless" when the truth is "this fold had one class in it".
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from app.ml.contracts import FoldScore
from app.ml.evaluation import build_evaluation_report, fold_metrics

BINARY_CLASSES = ["no", "yes"]


def binary_case():
    y_true = np.array(["no", "yes", "no", "yes", "no", "yes"])
    y_pred = np.array(["no", "yes", "yes", "yes", "no", "no"])
    # Column 1 is P(classes[1]) == P("yes"), matching sklearn's ordering.
    proba = np.array([[0.9, 0.1], [0.2, 0.8], [0.4, 0.6], [0.3, 0.7], [0.8, 0.2], [0.6, 0.4]])
    return y_true, y_pred, proba


def score(y_true, y_pred, proba, classes=BINARY_CLASSES, task_type="classification"):
    return fold_metrics(
        y_true=y_true, y_pred=y_pred, y_proba=proba, classes=classes, task_type=task_type
    )


class TestClassificationMetrics:
    def test_the_spec_metric_set_is_reported(self):
        metrics, _ = score(*binary_case())
        assert {
            "accuracy",
            "precision_macro",
            "recall_macro",
            "f1_macro",
            "roc_auc",
            "pr_auc",
        } <= set(metrics)

    def test_accuracy_is_correct(self):
        y_true, y_pred, proba = binary_case()
        metrics, _ = score(y_true, y_pred, proba)
        assert metrics["accuracy"] == pytest.approx(4 / 6)

    def test_a_perfect_prediction_scores_one(self):
        y_true = np.array(["no", "yes", "no", "yes"])
        proba = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
        metrics, _ = score(y_true, y_true, proba)
        assert metrics["accuracy"] == 1.0
        assert metrics["f1_macro"] == 1.0
        assert metrics["roc_auc"] == 1.0

    def test_macro_averaging_punishes_ignoring_a_rare_class(self):
        """The reason macro is used instead of weighted (see ml/evaluation.py)."""
        y_true = np.array(["no"] * 9 + ["yes"])
        all_majority = np.array(["no"] * 10)
        proba = np.tile([0.9, 0.1], (10, 1))
        metrics, _ = score(y_true, all_majority, proba)
        assert metrics["accuracy"] == pytest.approx(0.9)
        # Macro F1 averages a perfect majority class with a missed minority one.
        assert metrics["f1_macro"] < 0.6

    def test_probabilities_are_read_against_the_models_class_order(self):
        """Column 1 must be the probability of ``classes[1]``, not of "the first"."""
        y_true = np.array(["no", "yes", "no", "yes"])
        y_pred = y_true
        # Confidently correct on the "yes" rows via column 1.
        proba = np.array([[0.95, 0.05], [0.05, 0.95], [0.9, 0.1], [0.1, 0.9]])
        metrics, _ = score(y_true, y_pred, proba)
        assert metrics["roc_auc"] == 1.0


class TestMissingMetricsAreExplained:
    def test_without_probabilities_the_threshold_free_metrics_are_skipped(self):
        y_true, y_pred, _ = binary_case()
        metrics, warnings = score(y_true, y_pred, None)
        assert "roc_auc" not in metrics
        assert "pr_auc" not in metrics
        assert any("probabilities" in w for w in warnings)

    def test_a_single_class_fold_omits_roc_auc_with_a_reason(self):
        """A property of the data, not a bug -- so it is explained, not invented."""
        y_true = np.array(["no", "no", "no", "no"])
        y_pred = np.array(["no", "no", "no", "no"])
        proba = np.tile([0.8, 0.2], (4, 1))
        metrics, warnings = score(y_true, y_pred, proba)
        assert "roc_auc" not in metrics
        assert any("roc_auc" in w for w in warnings)

    def test_multiclass_reports_roc_auc_but_not_pr_auc(self):
        y_true = np.array(["a", "b", "c", "a", "b", "c"])
        y_pred = np.array(["a", "b", "c", "a", "b", "b"])
        proba = np.array(
            [
                [0.8, 0.1, 0.1],
                [0.1, 0.8, 0.1],
                [0.1, 0.1, 0.8],
                [0.7, 0.2, 0.1],
                [0.2, 0.7, 0.1],
                [0.2, 0.6, 0.2],
            ]
        )
        metrics, warnings = score(y_true, y_pred, proba, classes=["a", "b", "c"])
        assert "roc_auc" in metrics
        assert "pr_auc" not in metrics
        assert any("binary targets only" in w for w in warnings)

    def test_a_metric_is_never_reported_as_zero_instead_of_absent(self):
        y_true = np.array(["no", "no", "no"])
        proba = np.tile([0.8, 0.2], (3, 1))
        metrics, _ = score(y_true, y_true, proba)
        assert metrics.get("roc_auc") != 0.0


class TestRegressionMetrics:
    def test_the_spec_metric_set_is_reported(self):
        y_true = np.array([10.0, 20.0, 30.0, 40.0])
        y_pred = np.array([12.0, 18.0, 33.0, 39.0])
        metrics, warnings = fold_metrics(
            y_true=y_true, y_pred=y_pred, y_proba=None, classes=[], task_type="regression"
        )
        assert set(metrics) == {"mae", "mse", "rmse", "r2"}
        assert warnings == []

    def test_rmse_is_the_root_of_mse(self):
        y_true = np.array([10.0, 20.0, 30.0, 40.0])
        y_pred = np.array([12.0, 18.0, 33.0, 39.0])
        metrics, _ = fold_metrics(
            y_true=y_true, y_pred=y_pred, y_proba=None, classes=[], task_type="regression"
        )
        assert metrics["rmse"] == pytest.approx(math.sqrt(metrics["mse"]))

    def test_a_perfect_fit_scores_r2_of_one(self):
        y = np.array([10.0, 20.0, 30.0, 40.0])
        metrics, _ = fold_metrics(
            y_true=y, y_pred=y, y_proba=None, classes=[], task_type="regression"
        )
        assert metrics["r2"] == 1.0
        assert metrics["mae"] == 0.0


def folds_of(*values: float, metric: str = "f1_macro") -> list[FoldScore]:
    return [
        FoldScore(fold=i, n_train=80, n_test=20, metrics={metric: v})
        for i, v in enumerate(values, start=1)
    ]


def report_of(folds, task_type="classification"):
    return build_evaluation_report(
        folds,
        task_type=task_type,
        target_column="churn",
        model_name="RandomForestClassifier",
        n_folds=len(folds),
        cv_strategy="StratifiedKFold",
        n_rows=100,
        n_features=3,
    )


class TestCrossFoldSummary:
    def test_the_mean_is_averaged_over_the_folds(self):
        report = report_of(folds_of(0.6, 0.7, 0.8))
        assert report.metrics["f1_macro"].mean == pytest.approx(0.7)

    def test_identical_folds_have_no_spread(self):
        report = report_of(folds_of(0.7, 0.7, 0.7))
        assert report.metrics["f1_macro"].std == pytest.approx(0.0)

    def test_varying_folds_have_spread(self):
        """The std is the whole reason cross-validation beats one split."""
        assert report_of(folds_of(0.2, 0.9, 0.5)).metrics["f1_macro"].std > 0.1

    def test_a_metric_missing_from_one_fold_still_summarises_the_others(self):
        folds = folds_of(0.6, 0.8)
        folds[0].metrics["roc_auc"] = 0.9  # only fold 1 produced it
        report = report_of(folds)
        assert report.metrics["roc_auc"].mean == pytest.approx(0.9)
        assert report.metrics["f1_macro"].mean == pytest.approx(0.7)

    def test_the_primary_metric_is_f1_for_classification(self):
        report = report_of(folds_of(0.6, 0.8))
        assert report.primary_metric == "f1_macro"
        assert report.primary_score() == pytest.approx(0.7)

    def test_the_primary_metric_is_r2_for_regression(self):
        report = report_of(folds_of(0.5, 0.7, metric="r2"), task_type="regression")
        assert report.primary_metric == "r2"

    def test_warnings_are_carried_into_the_report(self):
        report = build_evaluation_report(
            folds_of(0.7),
            task_type="classification",
            target_column="churn",
            model_name="m",
            n_folds=1,
            cv_strategy="KFold",
            n_rows=10,
            n_features=2,
            warnings=["something was reduced"],
        )
        assert report.warnings == ["something was reduced"]

    def test_zero_folds_is_an_error(self):
        with pytest.raises(ValueError, match="zero folds"):
            report_of([])
