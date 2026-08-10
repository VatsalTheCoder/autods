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
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import KFold, StratifiedKFold, TimeSeriesSplit
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier, XGBRegressor

from app.agents.schema_models import TaskType
from app.core.config import get_settings
from app.ml.contracts import (
    ConcentratedFoldError,
    FoldScore,
    Leaderboard,
    LeaderboardEntry,
    MetricSummary,
)
from app.ml.evaluation import PRIMARY_METRIC, fold_metrics
from app.ml.target import TargetTransform, choose_target_transform

logger = logging.getLogger(__name__)

AGENT_NAME = "modeling"

_N_ESTIMATORS = 100

# SMOTE needs at least this many members of the minority class to interpolate
# between: its k nearest neighbours default is 5, and it cannot find 5 neighbours
# among 3 rows. Below the threshold the request is refused with a warning rather
# than silently reduced -- synthesising a class from two examples produces noise
# that looks like data.
_MIN_MINORITY_FOR_SMOTE = 6

# A scratch column used only to sort rows into time order before splitting. It is
# added and dropped inside cross-validation, so it never reaches the recipe and
# cannot become a feature. The name is deliberately one no CSV would carry.
_ORDER_KEY = "__autods_time_order__"

# How much of a time column may be unreadable before ordering by it is abandoned
# rather than repaired. Undated rows have no place in a chronological split (see
# ``order_by_time``); below this fraction they are set aside and the score
# describes the dated remainder, above it that remainder is no longer the
# dataset and random folds over everything are the more honest answer.
_MAX_TIME_COLUMN_NULL_RATE = 0.10


class ModelingError(RuntimeError):
    """Cross-validation could not be run at all, with a reason for the user."""


def order_by_time(
    frame: pd.DataFrame, time_column: str, warnings: list[str]
) -> tuple[pd.DataFrame, bool]:
    """Sort ``frame`` oldest-first by ``time_column``, or decline to.

    Returns the frame and whether time ordering was actually applied. A ``False``
    means the caller must fall back to its normal splitter: the frame handed back
    is then the original, untouched.

    ``TimeSeriesSplit`` slices by *position*, so this has to happen before any
    splitting or the folds are ordered by whatever sequence the CSV arrived in --
    which would look time-aware in the report and be nothing of the sort.

    The sort is stable, so rows sharing a timestamp keep their original order
    rather than being shuffled arbitrarily between folds.

    **Rows with no readable value are set aside, not sorted to one end.** All
    three options are lossy and the least bad one is not obvious, so it is worth
    saying why this is it. Sorting them last -- what this function used to do --
    hands the final fold or two a validation set made entirely of undated rows,
    which is a different population from anything the model trained on; the NYC
    Airbnb listings dataset fails exactly this way, where a fifth of the rows have
    no ``last_review`` and the last fold scores negative R² against a wall of
    them. Sorting them first puts them in every training fold instead, which
    hides the problem rather than removing it. Excluding them leaves a score that
    can be described honestly -- it is the score over the rows that carry a date
    -- and the count is warned about so the report never implies otherwise.

    Past ``_MAX_TIME_COLUMN_NULL_RATE`` that description stops being worth much,
    because the dated remainder is no longer the dataset the user uploaded. There
    the whole idea is abandoned and the caller reverts to random folds.

    **A numeric column counts as a clock.** Plenty of real datasets carry time as
    a counter rather than a date -- an hour index, epoch seconds, a day number.
    PaySim, the dataset that prompted this whole feature, numbers its hours in a
    ``step`` column and has no date anywhere. Refusing those would have meant
    shipping time-ordered validation that could not be applied to the case that
    motivated it. Ordering only needs the values to be *comparable*, which a
    counter is.
    """
    if time_column not in frame.columns:
        raise ModelingError(f"Time column {time_column!r} is not in the dataset.")

    column = frame[time_column]
    if pd.api.types.is_numeric_dtype(column):
        ordered = pd.to_numeric(column, errors="coerce")
        kind = "numbers"
    else:
        ordered = pd.to_datetime(column, errors="coerce")
        kind = "dates"

    # A column with nothing readable in it is a wrong column, not a gappy one.
    # The checkpoint already rejects a categorical or free-text choice, so
    # reaching here means the choice cannot be repaired by setting rows aside --
    # there would be none left. Refused loudly rather than quietly downgraded.
    if ordered.isna().all():
        raise ModelingError(
            f"Time column {time_column!r} holds no readable {kind}, so folds "
            "cannot be ordered by it."
        )

    missing = int(ordered.isna().sum())
    if missing:
        null_rate = missing / len(ordered)
        if null_rate > _MAX_TIME_COLUMN_NULL_RATE:
            warnings.append(
                f"Folds were not ordered by time: {missing:,} of {len(ordered):,} rows "
                f"({null_rate:.0%}) have no readable value in {time_column!r}. Ordering "
                "would have scored whole folds against undated rows alone, so the run "
                "used random folds over every row instead."
            )
            return frame, False
        warnings.append(
            f"{missing:,} rows have no readable value in {time_column!r} and were left "
            "out of cross-validation; scores describe the rows that carry a value."
        )
        frame = frame.loc[ordered.notna()]
        ordered = ordered.loc[ordered.notna()]

    return (
        frame.assign(**{_ORDER_KEY: ordered})
        .sort_values(_ORDER_KEY, kind="stable")
        .drop(columns=[_ORDER_KEY]),
        True,
    )


@dataclass(slots=True)
class Candidate:
    """One entry in the roster: a name and the estimator behind it."""

    name: str
    estimator: object
    # A baseline is scored and ranked alongside the contenders but is never the
    # model that gets served -- see ``run_leaderboard``.
    is_baseline: bool = False


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
    # The log-scale treatment, when the target's skew called for one. Carried
    # here so the report can say the models were fitted on a transformed target
    # -- the scores themselves are already back in the original units.
    target_transform: TargetTransform | None = None
    # Why one fold scored far below the others, when a few of its rows explain
    # it. None when the folds agree, or when no small set of rows is responsible.
    concentrated_fold_error: ConcentratedFoldError | None = None
    # Read off the splitter rather than assumed, so the report states what
    # actually happened even when a caller supplies its own.
    shuffled: bool = False
    random_seed: int | None = None


# How far below the median of the other folds the worst one must sit before it
# is worth explaining rather than treating as ordinary spread. In R² terms 0.05
# is roughly "a grade worse", and comfortably outside the fold-to-fold wobble a
# well-behaved dataset shows.
_FOLD_GAP_TO_EXPLAIN = 0.05

# The share of a fold's squared error that counts as "most of it".
_DOMINANT_ERROR_SHARE = 0.5

# The largest fraction of a fold's rows that can be called a *few* rows. Above
# this the error is diffuse, which is a different finding and not this one.
_MAX_DOMINANT_FRACTION = 0.05


def _diagnose_concentrated_error(
    folds: list[FoldScore],
    predictions: list[tuple[np.ndarray, np.ndarray]],
    *,
    metric: str,
) -> ConcentratedFoldError | None:
    """Find the worst fold, and say whether a few rows account for its score.

    Returns None unless both halves hold: one fold is materially worse than the
    others, *and* a small number of its rows own most of its squared error. Those
    two together are what distinguish "this dataset is small" from "these
    particular records are unlike the training data" -- advice that differs
    completely, and which the report previously collapsed into a single sentence
    recommending more data.

    Deliberately silent when the error is spread evenly. A fold that is simply
    harder is not a finding, and a diagnosis that fires on every run teaches a
    reader to skip it.
    """
    scores = [fold.metrics.get(metric) for fold in folds]
    if len(folds) < 3 or any(score is None for score in scores):
        return None

    worst = int(np.argmin(scores))
    others = [s for i, s in enumerate(scores) if i != worst]
    median = float(np.median(others))
    if median - scores[worst] < _FOLD_GAP_TO_EXPLAIN:
        return None

    y_true, y_pred = predictions[worst]
    errors = np.asarray((y_true - y_pred) ** 2, dtype=float)
    total = float(errors.sum())
    if not np.isfinite(total) or total <= 0:
        return None

    # The smallest set of rows accounting for most of the error, largest first.
    order = np.argsort(-errors)
    cumulative = np.cumsum(errors[order]) / total
    n_dominant = int(np.searchsorted(cumulative, _DOMINANT_ERROR_SHARE) + 1)
    if n_dominant > max(1, int(_MAX_DOMINANT_FRACTION * len(errors))):
        return None

    share = float(cumulative[n_dominant - 1])
    without = _score_without(y_true, y_pred, drop=order[:n_dominant], metric=metric)
    return ConcentratedFoldError(
        score_without_dominant=without,
        fold=folds[worst].fold,
        metric=metric,
        score=float(scores[worst]),
        median_score=median,
        n_test_rows=int(len(errors)),
        n_dominant_rows=n_dominant,
        dominant_error_share=share,
        note=(
            f"Fold {folds[worst].fold} scored {scores[worst]:.4f} against a median of "
            f"{median:.4f} across the other folds. {n_dominant} of its "
            f"{len(errors):,} held-out rows produce {share:.0%} of its squared "
            "error, so the gap is those records rather than the size of the "
            "dataset -- more rows would not close it."
            + (f" Without them the fold scores {without:.4f}." if without is not None else "")
        ),
    )


def _score_without(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    drop: np.ndarray,
    metric: str,
) -> float | None:
    """The fold's score with the dominant rows removed.

    Not a result and never presented as one -- it is how the report states the
    *size* of the effect those rows are having. Reported only for R², where
    "0.69 becomes 0.90" is immediately readable; an RMSE recomputed on a
    different row set invites comparison with the RMSE above it, which would be
    measured over different data.
    """
    if metric != "r2":
        return None
    keep = np.ones(len(y_true), dtype=bool)
    keep[drop] = False
    if keep.sum() < 2:
        return None
    kept_true, kept_pred = np.asarray(y_true)[keep], np.asarray(y_pred)[keep]
    ss_tot = float(((kept_true - kept_true.mean()) ** 2).sum())
    if ss_tot <= 0:
        return None
    return float(1.0 - ((kept_true - kept_pred) ** 2).sum() / ss_tot)


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

    **And below all of them, a model that does not look at the features at all.**
    A leaderboard without one gives a reader no way to tell a mediocre score from
    a meaningless one: R² 0.07 reads as a catastrophe and macro F1 0.62 reads as
    respectable, and neither impression survives learning what predicting the
    same answer every time would have scored. It is one extra fit per fold of a
    model with no parameters, which is the cheapest row on the board.

    ``median`` rather than ``mean`` for the regression baseline. R² is *defined*
    against the mean, so a mean-predictor scores 0.000 by construction and the
    row tells the reader nothing they did not already know from the metric's
    definition. The median predictor scores slightly below zero, and its MAE is
    the number that actually matters: the error you get for free.
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
            Candidate(
                "Baseline (most frequent class)",
                DummyClassifier(strategy="most_frequent"),
                is_baseline=True,
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
        Candidate(
            "Baseline (always the median)",
            DummyRegressor(strategy="median"),
            is_baseline=True,
        ),
    ]


def build_pipeline(
    preprocessor: ColumnTransformer,
    task_type: TaskType,
    *,
    random_seed: int,
    estimator=None,
    resampler=None,
    target_transform: TargetTransform | None = None,
):
    """Glue the unfitted recipe, an optional resampler and the estimator together.

    imblearn's ``Pipeline`` -- see this module's docstring, point 4. The order of
    the steps is the whole safety property: the resampler sits *between*
    preprocessing and the model, inside the object that gets cloned and fitted
    per fold, so it only ever sees a training fold. The returned object has not
    been fitted and must not be fitted by the caller.

    **A recipe that is itself a Pipeline is spliced in, not nested.** When the
    planner turns feature selection on, ``build_preprocessor`` returns a
    ``Pipeline`` of the ColumnTransformer plus a ``SelectKBest``, and imblearn
    rejects a Pipeline as an intermediate step outright ("All intermediate steps
    of the chain should not be Pipelines"). Nesting one therefore produces an
    object that cannot be fitted at all -- and, because ``run_leaderboard``
    catches each candidate's failure individually, the symptom is every model in
    the roster failing for the same opaque reason. Flattening keeps the step
    order, and so the leakage property, exactly as it was.

    **A target transform wraps the estimator rather than becoming a step**, since
    a pipeline step transforms ``X`` and this one transforms ``y``. Wrapping keeps
    it inside the object cloned per fold, so the log is taken and undone within
    the fold like everything else (``target.py``).
    """
    steps: list[tuple[str, object]] = []
    if isinstance(preprocessor, SklearnPipeline):
        steps.extend(preprocessor.steps)
    else:
        steps.append(("preprocess", preprocessor))
    if resampler is not None:
        steps.append(("resample", resampler))
    model = (
        estimator if estimator is not None else build_estimator(task_type, random_seed=random_seed)
    )
    if target_transform is not None:
        model = target_transform.wrap(model)
    steps.append(("model", model))
    return ImbPipeline(steps=steps)


def unwrap_estimator(model):
    """The estimator underneath any wrappers this module puts around one.

    Two wrappers exist, for unrelated reasons -- ``LabelEncodedClassifier`` so
    XGBoost can be handed integer labels, and ``TransformedTargetRegressor`` so a
    skewed target can be modelled on a log scale (``target.py``). Anything that
    wants the real model, to name its family or to hand it to SHAP, wants this
    rather than ``named_steps["model"]``.

    Fitted and unfitted pipelines both work: scikit-learn parks the fitted inner
    estimator on a trailing-underscore attribute and leaves the constructor
    argument in place, so each is tried in turn. Applied repeatedly, because the
    two wrappers can legitimately nest.
    """
    for _ in range(4):  # more nesting than exists; a guard against a cycle
        if isinstance(model, LabelEncodedClassifier):
            model = getattr(model, "estimator_", None) or model.estimator
        elif isinstance(model, TransformedTargetRegressor):
            model = getattr(model, "regressor_", None) or model.regressor
        else:
            return model
    return model


def preprocessing_of(pipeline: ImbPipeline):
    """Everything a fitted pipeline does to the data before the model sees it.

    Needed because the recipe is not always one step: with feature selection on
    it is spliced in as two (see ``build_pipeline``), so "the preprocessing" has
    to be identified by what it is rather than by a fixed step name. Any
    resampler is left out -- it has no ``transform`` and does not run at
    prediction time, which is exactly what makes SMOTE safe here.

    The result is the fitted transform half, ready to hand SHAP the matrix the
    model actually sees along with the names of its columns.
    """
    steps = [
        (name, step)
        for name, step in pipeline.steps
        if name != "model" and not hasattr(step, "fit_resample")
    ]
    if len(steps) == 1:
        return steps[0][1]
    return SklearnPipeline(steps=steps)


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
    time_column: str | None = None,
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

    warnings: list[str] = []

    # Time-ordered validation, when the user named a time column at the
    # checkpoint. TimeSeriesSplit slices by *position*, so the rows have to be in
    # time order before it sees them or the split means nothing at all -- it
    # would happily "validate on the future" using whatever order the CSV
    # happened to arrive in. ``order_by_time`` may decline, in which case the
    # ordinary splitter below is the correct one and says so in the report.
    time_ordered = False
    if time_column:
        frame, time_ordered = order_by_time(frame, time_column, warnings)

    X = frame[features]
    y = frame[target]

    if splitter is None:
        n_folds, splitter, strategy = _make_splitter(
            y,
            task_type=task_type,
            cv_folds=cv_folds,
            random_seed=random_seed,
            warnings=warnings,
            time_ordered=time_ordered,
        )
    else:
        n_folds, strategy = splitter.get_n_splits(), type(splitter).__name__

    # Decided here rather than passed in, and safe to decide twice: it is a pure
    # function of the target column, so the leaderboard's copy (recorded for the
    # report) and this one cannot disagree about the same rows.
    transform = choose_target_transform(y, task_type=task_type)
    if transform is not None and transform.rationale not in warnings:
        warnings.append(transform.rationale)

    template = build_pipeline(
        preprocessor,
        task_type,
        random_seed=random_seed,
        estimator=candidate.estimator if candidate else None,
        resampler=resampler,
        target_transform=transform,
    )

    folds: list[FoldScore] = []
    # Held-out actuals and predictions per fold, kept only for regression and
    # only so the diagnosis below can ask *which rows* a bad fold's error came
    # from. Two float arrays per fold; discarded with this function's frame.
    predictions: list[tuple[np.ndarray, np.ndarray]] = []
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

        if task_type == "regression":
            predictions.append((np.asarray(y_test, dtype=float), np.asarray(y_pred, dtype=float)))

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

    concentrated = (
        _diagnose_concentrated_error(folds, predictions, metric="r2")
        if task_type == "regression" and len(predictions) == len(folds)
        else None
    )
    if concentrated is not None:
        logger.info("Fold variance: %s", concentrated.note)

    return CrossValidationResult(
        folds=folds,
        concentrated_fold_error=concentrated,
        shuffled=bool(getattr(splitter, "shuffle", False)),
        random_seed=getattr(splitter, "random_state", None),
        model_name=(
            candidate.name
            if candidate
            else type(unwrap_estimator(template.named_steps["model"])).__name__
        ),
        cv_strategy=strategy,
        n_folds=n_folds,
        n_rows=int(frame.shape[0]),
        n_features=len(features),
        pipeline_template=template,
        warnings=warnings,
        target_transform=transform,
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
    time_column: str | None = None,
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

    warnings: list[str] = []
    # Ordered before the splitter is built, not after: the empty-fold check below
    # inspects which rows land in which fold, and on an unordered frame it would
    # be inspecting the wrong ones.
    time_ordered = False
    if time_column:
        frame, time_ordered = order_by_time(frame, time_column, warnings)

    y = frame[target]
    n_folds, splitter, strategy = _make_splitter(
        y,
        task_type=task_type,
        cv_folds=cv_folds,
        random_seed=random_seed,
        warnings=warnings,
        time_ordered=time_ordered,
    )

    # Every candidate gets the same treatment of the target, for the same reason
    # they get the same splits: a roster where one model was fitted on log dollars
    # and another on dollars is not a comparison.
    transform = choose_target_transform(y, task_type=task_type)
    if transform is not None:
        warnings.append(transform.rationale)

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
    scored: list[tuple[LeaderboardEntry, CrossValidationResult | None, bool]] = []

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
                        is_baseline=candidate.is_baseline,
                    ),
                    None,
                    candidate.is_baseline,
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
                    is_baseline=candidate.is_baseline,
                ),
                result,
                candidate.is_baseline,
            )
        )

    # Higher is better for every primary metric in use (macro F1, R²). Failed
    # candidates sort last on their NaN score rather than being interleaved.
    #
    # The baseline sorts on its score like everything else rather than being
    # pinned to the bottom: where it lands *is* the information. A board whose
    # top row predicts the same answer every time has said something no other
    # arrangement of the same numbers would have.
    scored.sort(key=lambda row: (row[0].error != "", -_sortable(row[0].score)))
    for rank, (entry, _, _) in enumerate(scored, start=1):
        entry.rank = rank

    # The served model is the best *real* one. A no-information model that
    # happens to top the board is a finding to report, not an artifact to deploy
    # -- and SHAP cannot explain an estimator that never looked at a feature.
    winner = next(
        (result for _, result, baseline in scored if result is not None and not baseline), None
    )
    if winner is None:
        raise ModelingError(
            "No model could be trained on this dataset. "
            + "; ".join(entry.error for entry, _, _ in scored if entry.error)
        )

    warnings.extend(_baseline_verdict(scored, primary=primary))

    leaderboard = Leaderboard(
        task_type=task_type,
        target_column=target,
        primary_metric=primary,
        n_folds=n_folds,
        cv_strategy=strategy,
        entries=[entry for entry, _, _ in scored],
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


def _baseline_verdict(
    scored: list[tuple[LeaderboardEntry, CrossValidationResult | None, bool]],
    *,
    primary: str,
) -> list[str]:
    """Say what the featureless baseline proved, in the two cases where it matters.

    Reading a score is guesswork without this. R² 0.07 on listing prices sounds
    like nothing at all, and it is in fact a real improvement on the 0.00 a
    median-predictor gets; macro F1 0.62 sounds respectable and can be *worse*
    than always answering with the commonest class. Only the comparison
    distinguishes them, so the comparison is stated rather than left as two
    numbers a reader has to subtract.

    Silent when nothing interesting happened -- a baseline comfortably beaten is
    the expected outcome and needs no paragraph.
    """
    baseline = next((entry for entry, _, is_base in scored if is_base), None)
    best = next((entry for entry, _, is_base in scored if not is_base and not entry.error), None)
    if baseline is None or best is None or baseline.error:
        return []

    gap = best.score - baseline.score
    if gap <= 0:
        return [
            f"No model beat the featureless baseline: the best {primary} was "
            f"{best.score:.4f} ({best.model_name}) against {baseline.score:.4f} for "
            f"{baseline.model_name.lower()}. The features carry little or no signal "
            "about this target, and the ranking above is between models that are all "
            "failing at the same task."
        ]
    # A lead this thin is not a lead. The threshold is one fold's worth of
    # spread: if the winner's advantage over the baseline is inside the noise
    # that separates its own folds, the advantage is not established.
    if gap < best.std:
        return [
            f"The best model beats the featureless baseline by only {gap:.4f} on "
            f"{primary}, which is less than its own spread across folds "
            f"(±{best.std:.4f}). Treat the improvement as unproven rather than small."
        ]
    return []


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
    time_ordered: bool = False,
) -> tuple[int, KFold | StratifiedKFold | TimeSeriesSplit, str]:
    """Choose the splitter and a fold count the data can actually support.

    ``time_ordered`` overrides everything below it. When the user has named a
    time column, folds must respect it: each fold trains on rows that came before
    the ones it is scored on, so a model is never asked to predict the past from
    the future. That is a *different* leak from the one the rest of this module
    guards -- preprocessing fitted across a split -- and neither implies the
    other.

    The cost is that time-ordered folds cannot be stratified, because which rows
    fall in which fold is decided by the clock rather than by the label. On a
    rare-event dataset an early fold can therefore contain no positive cases at
    all. That is warned about rather than prevented: it is a real property of
    validating a rare event chronologically, and hiding it by silently reverting
    to random folds would be the worse answer.

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

    if time_ordered:
        # TimeSeriesSplit needs one more row than folds: fold k trains on
        # everything before its test window, so the first window must have
        # something behind it.
        n_folds = max(2, min(n_folds, n_rows - 1))
        splitter = TimeSeriesSplit(n_splits=n_folds)

        if task_type == "classification":
            empty = [
                index
                for index, (_, test_idx) in enumerate(splitter.split(y), start=1)
                if y.iloc[test_idx].nunique(dropna=True) < 2
            ]
            if empty:
                warnings.append(
                    f"Folds ordered by time cannot be class-balanced. "
                    f"{len(empty)} of {n_folds} validation folds contain a single "
                    "class, so their threshold-free metrics are unreliable; the "
                    "averages across folds are still reported."
                )
        return n_folds, splitter, "TimeSeriesSplit"

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
