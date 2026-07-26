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
3. The whole thing -- preprocessing, resampling *and* model -- is one
   ``Pipeline``, so ``fit`` cannot accidentally be called on the preprocessor at
   a different time from the model. There is one fit call per fold, on one
   object.
4. That object is **imblearn's** ``Pipeline``, not scikit-learn's. This is what
   makes SMOTE correct rather than catastrophic. imblearn's pipeline applies a
   resampler during ``fit`` and *skips it during* ``predict`` -- so the minority
   class is oversampled in the training portion of each fold and never in the
   part being scored. Section 5 adopted this class before there was a resampler
   to put in it, precisely so that there is no version of this file where SMOTE
   sits outside the fold (spec 8).

Section 5 trained one model. Section 7 turns that into the spec's roster and
ranks it on a leaderboard -- every candidate over the *same* splits from the same
seed, which is the only reason the ranking means anything. The fold discipline
above is unchanged; there are simply four of them now.

On SMOTE and one-hot columns, since it is a fair objection: resampling happens
after the ColumnTransformer, so SMOTE interpolates between encoded rows and can
produce a dummy of 0.4. The alternative, SMOTENC before encoding, needs
categorical *positions* in the raw frame and would tie this module to the
recipe's internals. The tree ensembles that dominate the roster split on such a
value without trouble, and the resampling is confined to training rows either
way -- so the cost is some synthetic rows being less crisp than real ones, not a
score that is wrong.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier, XGBRegressor

from app.agents.schema_models import TaskType
from app.core.config import get_settings
from app.ml.contracts import (
    FoldScore,
    Leaderboard,
    LeaderboardEntry,
    MetricSummary,
)
from app.ml.evaluation import PRIMARY_METRIC, fold_metrics

logger = logging.getLogger(__name__)

AGENT_NAME = "modeling"

_N_ESTIMATORS = 100

# SMOTE needs at least this many members of the minority class to interpolate
# between: its k nearest neighbours default is 5, and it cannot find 5 neighbours
# among 3 rows. Below the threshold the request is refused with a warning rather
# than silently reduced -- synthesising a class from two examples produces noise
# that looks like data.
_MIN_MINORITY_FOR_SMOTE = 6


class ModelingError(RuntimeError):
    """Cross-validation could not be run at all, with a reason for the user."""


@dataclass(slots=True)
class Candidate:
    """One entry in the roster: a name and the estimator behind it."""

    name: str
    estimator: object


class LabelEncodedClassifier(BaseEstimator, ClassifierMixin):
    """Wraps a classifier that insists on 0..n-1 integer labels.

    XGBoost's scikit-learn wrapper refuses the string targets this pipeline
    carries ("yes"/"no"), so the labels are encoded on the way in and decoded on
    the way out. Doing it here rather than converting the target upstream keeps
    the change local to the one model that needs it: every other candidate, the
    metrics and the report all continue to see the labels the user actually
    uploaded.

    The encoder is fitted per fold along with everything else, which is correct
    for a further reason -- a fold whose training rows happen to lack a class
    genuinely has fewer classes, and encoding against the whole dataset's label
    set would paper over that.
    """

    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, X, y):  # noqa: N803 - sklearn's parameter name
        self.encoder_ = LabelEncoder().fit(y)
        self.classes_ = self.encoder_.classes_
        self.estimator_ = clone(self.estimator).fit(X, self.encoder_.transform(y))
        return self

    def predict(self, X):  # noqa: N803 - sklearn's parameter name
        return self.encoder_.inverse_transform(self.estimator_.predict(X))

    def predict_proba(self, X):  # noqa: N803 - sklearn's parameter name
        return self.estimator_.predict_proba(X)


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
    """The default single estimator -- the first entry of the roster.

    A random forest: it copes with mixed numeric and one-hot data, needs no
    tuning to give a sane number, and does not emit convergence warnings on
    awkward datasets the way an untuned linear model does.
    """
    return build_roster(task_type, random_seed=random_seed)[0].estimator


def build_roster(task_type: TaskType, *, random_seed: int) -> list[Candidate]:
    """The spec's four model families for this task type (7.7).

    Deliberately untuned. Every one is at its library default beyond the seed,
    because the leaderboard's claim is "these four families, compared fairly",
    and a hand-tuned XGBoost against a default logistic regression would be a
    claim about how much effort each got instead. Hyperparameter search is not in
    this project's scope, and pretending otherwise with a few hand-picked
    settings would be worse than not doing it.

    The linear model is in the roster as a baseline as much as a contender: if a
    logistic regression matches the gradient-boosted trees, the honest reading is
    that the problem is linear and the ensembles are buying nothing.
    """
    seed = random_seed
    if task_type == "classification":
        return [
            Candidate("RandomForest", RandomForestClassifier(_N_ESTIMATORS, random_state=seed)),
            Candidate(
                "LogisticRegression",
                # The default 100 iterations is not enough for the scaled,
                # one-hot-widened matrices this pipeline produces, and the result
                # is a ConvergenceWarning and an under-fitted baseline that makes
                # the ensembles look better than they are.
                LogisticRegression(max_iter=1000, random_state=seed),
            ),
            Candidate(
                "XGBoost",
                # Wrapped because XGBoost's sklearn API requires 0..n-1 integer
                # labels and this pipeline's targets are whatever the user
                # uploaded -- see ``LabelEncodedClassifier``.
                LabelEncodedClassifier(
                    XGBClassifier(n_estimators=_N_ESTIMATORS, random_state=seed, verbosity=0)
                ),
            ),
            Candidate(
                "LightGBM",
                LGBMClassifier(n_estimators=_N_ESTIMATORS, random_state=seed, verbose=-1),
            ),
        ]
    return [
        Candidate("RandomForest", RandomForestRegressor(_N_ESTIMATORS, random_state=seed)),
        Candidate("LinearRegression", LinearRegression()),
        Candidate(
            "XGBoost", XGBRegressor(n_estimators=_N_ESTIMATORS, random_state=seed, verbosity=0)
        ),
        Candidate(
            "LightGBM", LGBMRegressor(n_estimators=_N_ESTIMATORS, random_state=seed, verbose=-1)
        ),
    ]


def build_pipeline(
    preprocessor: ColumnTransformer,
    task_type: TaskType,
    *,
    random_seed: int,
    estimator=None,
    resampler=None,
):
    """Glue the unfitted recipe, an optional resampler and the estimator together.

    imblearn's ``Pipeline`` -- see this module's docstring, point 4. The order of
    the steps is the whole safety property: the resampler sits *between*
    preprocessing and the model, inside the object that gets cloned and fitted
    per fold, so it only ever sees a training fold. The returned object has not
    been fitted and must not be fitted by the caller.
    """
    steps = [("preprocess", preprocessor)]
    if resampler is not None:
        steps.append(("resample", resampler))
    steps.append(
        (
            "model",
            estimator
            if estimator is not None
            else build_estimator(task_type, random_seed=random_seed),
        )
    )
    return ImbPipeline(steps=steps)


def build_resampler(
    y: pd.Series,
    *,
    task_type: TaskType,
    random_seed: int,
    warnings: list[str],
):
    """SMOTE, or ``None`` with a stated reason (spec 7.7).

    Refused rather than attempted in three cases, each of which would otherwise
    fail from inside a fold where the error is least legible: a regression target
    has no minority class to oversample, a balanced target has nothing to gain,
    and a class with fewer than a handful of members cannot be interpolated
    between at all.
    """
    if task_type != "classification":
        warnings.append("SMOTE was requested but the target is continuous, so it was skipped.")
        return None

    counts = y.value_counts(dropna=True)
    if len(counts) < 2:
        warnings.append("SMOTE was requested but the target has only one class.")
        return None

    smallest = int(counts.min())
    if smallest < _MIN_MINORITY_FOR_SMOTE:
        warnings.append(
            f"SMOTE was requested but the rarest class has only {smallest} rows; "
            "synthesising from so few examples would invent structure rather than "
            "balance it, so it was skipped."
        )
        return None

    # k_neighbours must be under the minority count *within a training fold*,
    # which is smaller than the whole-dataset count -- hence the extra headroom.
    k = max(1, min(5, smallest - 2))
    return SMOTE(random_state=random_seed, k_neighbors=k)


def cross_validate_model(
    frame: pd.DataFrame,
    *,
    target: str,
    task_type: TaskType,
    preprocessor: ColumnTransformer,
    cv_folds: int | None = None,
    random_seed: int | None = None,
    candidate: Candidate | None = None,
    resampler=None,
    splitter=None,
) -> CrossValidationResult:
    """Run k-fold cross-validation, fitting the pipeline only inside each fold.

    ``splitter`` is accepted so the leaderboard can hand every candidate the
    *same* splits -- see ``run_leaderboard``. Left out, one is chosen from the
    data as before.
    """
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
    if splitter is None:
        n_folds, splitter, strategy = _make_splitter(
            y, task_type=task_type, cv_folds=cv_folds, random_seed=random_seed, warnings=warnings
        )
    else:
        n_folds, strategy = splitter.get_n_splits(), type(splitter).__name__

    template = build_pipeline(
        preprocessor,
        task_type,
        random_seed=random_seed,
        estimator=candidate.estimator if candidate else None,
        resampler=resampler,
    )

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
        model_name=candidate.name if candidate else type(template.named_steps["model"]).__name__,
        cv_strategy=strategy,
        n_folds=n_folds,
        n_rows=int(frame.shape[0]),
        n_features=len(features),
        pipeline_template=template,
        warnings=warnings,
    )


def run_leaderboard(
    frame: pd.DataFrame,
    *,
    target: str,
    task_type: TaskType,
    preprocessor: ColumnTransformer,
    use_smote: bool = False,
    cv_folds: int | None = None,
    random_seed: int | None = None,
) -> tuple[Leaderboard, CrossValidationResult]:
    """Cross-validate the whole roster and rank it (spec 7.7, 7.8).

    Returns the ranking and the winner's full result, because the evaluation
    report is still about one model -- the best one -- and rebuilding its folds
    from the leaderboard would mean running it twice.

    **Every candidate gets the same splits.** The splitter is built once, before
    the loop, and handed to each run. Letting each model derive its own from the
    same seed would *probably* produce identical folds, and "probably" is not
    something a ranking should rest on: comparing scores computed over different
    partitions is comparing two different questions.

    A candidate that cannot be trained is recorded with its error rather than
    dropped. A leaderboard silently missing LightGBM looks like a leaderboard of
    three models, not like a failure worth investigating.
    """
    settings = get_settings()
    if cv_folds is None:
        cv_folds = settings.cv_folds
    if random_seed is None:
        random_seed = settings.random_seed

    y = frame[target]
    warnings: list[str] = []
    n_folds, splitter, strategy = _make_splitter(
        y, task_type=task_type, cv_folds=cv_folds, random_seed=random_seed, warnings=warnings
    )

    resampler = None
    resampling = "none"
    if use_smote:
        resampler = build_resampler(
            y, task_type=task_type, random_seed=random_seed, warnings=warnings
        )
        resampling = (
            "SMOTE, applied inside each training fold only" if resampler is not None else "none"
        )

    primary = PRIMARY_METRIC.get(task_type, "")
    scored: list[tuple[LeaderboardEntry, CrossValidationResult | None]] = []

    for candidate in build_roster(task_type, random_seed=random_seed):
        started = time.perf_counter()
        try:
            result = cross_validate_model(
                frame,
                target=target,
                task_type=task_type,
                preprocessor=preprocessor,
                cv_folds=cv_folds,
                random_seed=random_seed,
                candidate=candidate,
                resampler=resampler,
                splitter=splitter,
            )
        except Exception as exc:  # noqa: BLE001 - one model failing must not end the run
            logger.warning("Candidate %s failed: %s", candidate.name, exc)
            scored.append(
                (
                    LeaderboardEntry(
                        rank=0,
                        model_name=candidate.name,
                        primary_metric=primary,
                        score=float("nan"),
                        std=0.0,
                        error=str(exc),
                    ),
                    None,
                )
            )
            continue

        summary = _summarise_folds(result.folds)
        warnings.extend(w for w in result.warnings if w not in warnings)
        scored.append(
            (
                LeaderboardEntry(
                    rank=0,
                    model_name=candidate.name,
                    primary_metric=primary,
                    score=summary[primary].mean if primary in summary else float("nan"),
                    std=summary[primary].std if primary in summary else 0.0,
                    metrics=summary,
                    fit_seconds=round(time.perf_counter() - started, 2),
                ),
                result,
            )
        )

    # Higher is better for every primary metric in use (macro F1, R²). Failed
    # candidates sort last on their NaN score rather than being interleaved.
    scored.sort(key=lambda pair: (pair[0].error != "", -_sortable(pair[0].score)))
    for rank, (entry, _) in enumerate(scored, start=1):
        entry.rank = rank

    winner = next((result for entry, result in scored if result is not None), None)
    if winner is None:
        raise ModelingError(
            "No model could be trained on this dataset. "
            + "; ".join(entry.error for entry, _ in scored if entry.error)
        )

    leaderboard = Leaderboard(
        task_type=task_type,
        target_column=target,
        primary_metric=primary,
        n_folds=n_folds,
        cv_strategy=strategy,
        entries=[entry for entry, _ in scored],
        resampling=resampling,
        warnings=warnings,
    )
    logger.info(
        "Leaderboard: %s wins on %s = %.4f (%d candidates)",
        leaderboard.entries[0].model_name,
        primary,
        leaderboard.entries[0].score,
        len(leaderboard.entries),
    )
    return leaderboard, winner


def _sortable(score: float) -> float:
    """NaN sorts as the worst possible score rather than poisoning the order."""
    return -float("inf") if score != score else score


def _summarise_folds(folds: list[FoldScore]) -> dict[str, MetricSummary]:
    """Mean and spread per metric, over the folds that produced each one."""
    names = sorted({name for fold in folds for name in fold.metrics})
    summary: dict[str, MetricSummary] = {}
    for name in names:
        values = [fold.metrics[name] for fold in folds if name in fold.metrics]
        if values:
            summary[name] = MetricSummary(mean=float(np.mean(values)), std=float(np.std(values)))
    return summary


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
