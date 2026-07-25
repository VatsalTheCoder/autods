"""Cross-validation -- where the unfitted recipe is finally fitted, per fold (spec 7.7).

**This is the file to read if you want to check the project's methodology.**

The rule, in one sentence: the pipeline handed in is never fitted; a *clone* of it
is fitted inside each fold, on that fold's training rows only, and then scored on
rows it has never seen. Four things make that true here, and each is load-bearing:

1. ``clone()`` on every iteration. The template that arrives from
   ``preprocessing.py`` leaves this function in exactly the state it entered --
   unfitted. Each fold starts from a fresh, equally unfitted copy, so no fold can
   inherit anything another fold learned.
2. ``fit`` is called on ``X.iloc[train]`` only. The held-out rows are not passed
   to ``fit`` in any form -- not to the imputer, not to the scaler, not to the
   encoder. This is the step that most student projects get wrong by scaling the
   whole dataset first, and the reason the fold's row counts are recorded in the
   artifact where a reader can check them.
3. The whole thing -- preprocessing *and* model -- is one ``Pipeline``, so
   ``fit`` cannot accidentally be called on the preprocessor at a different time
   from the model. There is one fit call per fold, on one object.
4. That object is **imblearn's** ``Pipeline``, not scikit-learn's. Identical
   behaviour today; the difference is that when Section 7 adds SMOTE, the
   resampler slots in as a step and is therefore applied to training folds only.
   Adopting the right pipeline class before there is a resampler means there is
   never a version of this code with SMOTE outside the fold (spec 8).

Section 5 trains one model. Section 7 turns the single estimator below into a
roster and a leaderboard; the fold discipline above does not change.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import KFold, StratifiedKFold

from app.agents.schema_models import TaskType
from app.core.config import get_settings
from app.ml.contracts import FoldScore
from app.ml.evaluation import fold_metrics

logger = logging.getLogger(__name__)

AGENT_NAME = "modeling"

# The single Section 5 model. A random forest is the right default for a first
# vertical slice: it copes with mixed numeric and one-hot data, needs no tuning
# to give a sane number, and does not emit convergence warnings on awkward
# datasets the way an unturned linear model does. Section 7 adds the rest of the
# spec's roster (logistic/linear regression, XGBoost, LightGBM) and picks between
# them on the leaderboard.
_N_ESTIMATORS = 100


class ModelingError(RuntimeError):
    """Cross-validation could not be run at all, with a reason for the user."""


@dataclass(slots=True)
class CrossValidationResult:
    """Per-fold scores and the metadata the evaluation report needs.

    ``pipeline_template`` is handed back deliberately: it is the same object that
    came in, still unfitted, and returning it lets callers (and the test suite)
    assert that running cross-validation did not fit it. The fitted per-fold
    copies are thrown away -- fitting on the full dataset for serving is a
    separate, explicit step (spec 7.9, Section 8).
    """

    folds: list[FoldScore]
    model_name: str
    cv_strategy: str
    n_folds: int
    n_rows: int
    n_features: int
    pipeline_template: ImbPipeline
    warnings: list[str] = field(default_factory=list)


def build_estimator(task_type: TaskType, *, random_seed: int):
    """The estimator for a task type, seeded so a reported score is reproducible."""
    if task_type == "classification":
        return RandomForestClassifier(n_estimators=_N_ESTIMATORS, random_state=random_seed)
    return RandomForestRegressor(n_estimators=_N_ESTIMATORS, random_state=random_seed)


def build_pipeline(preprocessor: ColumnTransformer, task_type: TaskType, *, random_seed: int):
    """Glue the unfitted recipe and the estimator into one unfitted pipeline.

    imblearn's ``Pipeline`` -- see this module's docstring, point 4. The returned
    object has not been fitted and must not be fitted by the caller.
    """
    return ImbPipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", build_estimator(task_type, random_seed=random_seed)),
        ]
    )


def cross_validate_model(
    frame: pd.DataFrame,
    *,
    target: str,
    task_type: TaskType,
    preprocessor: ColumnTransformer,
    cv_folds: int | None = None,
    random_seed: int | None = None,
) -> CrossValidationResult:
    """Run k-fold cross-validation, fitting the pipeline only inside each fold."""
    settings = get_settings()
    if cv_folds is None:
        cv_folds = settings.cv_folds
    if random_seed is None:
        random_seed = settings.random_seed

    features = [c for c in frame.columns if c != target]
    if not features:
        raise ModelingError("No feature columns to train on.")

    X = frame[features]
    y = frame[target]

    warnings: list[str] = []
    n_folds, splitter, strategy = _make_splitter(
        y, task_type=task_type, cv_folds=cv_folds, random_seed=random_seed, warnings=warnings
    )

    template = build_pipeline(preprocessor, task_type, random_seed=random_seed)

    folds: list[FoldScore] = []
    for index, (train_idx, test_idx) in enumerate(splitter.split(X, y), start=1):
        # A fresh unfitted copy per fold. Without this, fold 2 would be scored by
        # a pipeline that had already seen fold 2's rows during fold 1's fit.
        pipeline = clone(template)

        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]

        # The only fit call. Training rows only -- the held-out rows below are
        # not passed to it, which is the entire point of the exercise.
        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)
        y_proba = _predict_proba(pipeline, X_test)
        classes = list(getattr(pipeline.named_steps["model"], "classes_", []))

        metrics, fold_warnings = fold_metrics(
            y_true=y_test,
            y_pred=y_pred,
            y_proba=y_proba,
            classes=classes,
            task_type=task_type,
        )
        warnings.extend(w for w in fold_warnings if w not in warnings)

        folds.append(
            FoldScore(
                fold=index,
                n_train=int(len(train_idx)),
                n_test=int(len(test_idx)),
                metrics=metrics,
            )
        )
        logger.info(
            "Fold %d/%d: fitted on %d rows, scored on %d held-out rows",
            index,
            n_folds,
            len(train_idx),
            len(test_idx),
        )

    return CrossValidationResult(
        folds=folds,
        model_name=type(template.named_steps["model"]).__name__,
        cv_strategy=strategy,
        n_folds=n_folds,
        n_rows=int(frame.shape[0]),
        n_features=len(features),
        pipeline_template=template,
        warnings=warnings,
    )


def _make_splitter(
    y: pd.Series,
    *,
    task_type: TaskType,
    cv_folds: int,
    random_seed: int,
    warnings: list[str],
) -> tuple[int, KFold | StratifiedKFold, str]:
    """Choose the splitter and a fold count the data can actually support.

    Stratified for classification, so every fold holds roughly the class
    proportions of the whole dataset -- without it, a rare class can be absent
    from a training fold and the fold's score becomes noise (spec 7.7).

    The fold count is reduced rather than the run abandoned when a small or
    skewed dataset cannot support five: a stratified split needs at least as many
    members of the rarest class as there are folds. Every reduction is recorded
    as a warning so the report never implies five folds it did not run.
    """
    n_rows = int(len(y))
    if n_rows < 4:
        raise ModelingError(f"Only {n_rows} rows available; cross-validation needs at least 4.")

    n_folds = min(cv_folds, n_rows)

    if task_type == "classification":
        smallest_class = int(y.value_counts(dropna=True).min())
        if smallest_class < 2:
            # Cannot stratify around a class with a single member. Plain k-fold
            # is the honest fallback, and the warning says the folds are not
            # class-balanced so nobody reads more into the score than is there.
            warnings.append(
                "The rarest class has only one row, so folds could not be "
                "stratified; scores are less stable than usual."
            )
            n_folds = max(2, min(n_folds, n_rows))
            return n_folds, KFold(n_splits=n_folds, shuffle=True, random_state=random_seed), "KFold"

        if smallest_class < n_folds:
            warnings.append(
                f"Reduced to {smallest_class} folds: the rarest class has only "
                f"{smallest_class} rows, too few for {n_folds}-fold stratification."
            )
            n_folds = smallest_class

        n_folds = max(2, n_folds)
        return (
            n_folds,
            StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_seed),
            "StratifiedKFold",
        )

    if n_folds < cv_folds:
        warnings.append(f"Reduced to {n_folds} folds: the dataset has only {n_rows} rows.")
    n_folds = max(2, n_folds)
    return n_folds, KFold(n_splits=n_folds, shuffle=True, random_state=random_seed), "KFold"


def _predict_proba(pipeline: ImbPipeline, X_test: pd.DataFrame) -> np.ndarray | None:
    """Class probabilities when the model offers them, for ROC-AUC and PR-AUC.

    Regressors have no ``predict_proba``; returning None lets the metric layer
    skip the threshold-free metrics rather than the caller having to know which
    estimators support what.
    """
    if not hasattr(pipeline, "predict_proba"):
        return None
    try:
        return np.asarray(pipeline.predict_proba(X_test))
    except (AttributeError, ValueError):  # pragma: no cover - estimator-specific
        return None
