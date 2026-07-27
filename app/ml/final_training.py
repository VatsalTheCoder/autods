"""Final training -- the one place fitting on the whole dataset is correct.

Every other module in ``app/ml`` exists to keep a fitted thing away from the
rows it will be scored on. This one deliberately fits on all of them, and it is
worth being precise about why that is not a contradiction of ``modeling.py``:

* Cross-validation has already answered "how well does this configuration
  generalise". That question is settled, the number is in
  ``evaluation_report.json``, and nothing here changes it.
* The remaining question is "which estimator should we actually serve", and the
  answer is the one trained on the most data available. Holding a slice back
  now would produce a *weaker* served model in exchange for a second score that
  is less reliable than the cross-validated one it duplicates (spec 7.9).

The line between the two is that the score is never recomputed here. ``cv_score``
on ``FinalModelInfo`` is copied from the cross-validated run of the same
configuration and named so it cannot be read as a measurement of this model. If
this module ever grows a ``pipeline.score(X, y)`` call, that number would be the
training score of a model fitted on those exact rows -- the single most flattering
and least meaningful figure the project could publish.

The winner is rebuilt from the roster by name rather than being carried over as a
fitted object from the fold loop. Those per-fold copies each saw four fifths of
the data; reusing one would mean serving a model trained on less data than the
leaderboard implies, and re-deriving it from ``build_roster`` at the same seed is
one line.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.base import clone

from app.agents.schema_models import TaskType
from app.core.config import get_settings
from app.ml.contracts import FeatureStrategy, FinalModelInfo, PredictorColumn
from app.ml.modeling import build_pipeline, build_resampler, build_roster

logger = logging.getLogger(__name__)

AGENT_NAME = "final_training"


class FinalTrainingError(RuntimeError):
    """The winning configuration could not be refitted on the full dataset."""


@dataclass(slots=True)
class FinalModel:
    """The fitted pipeline destined for S3, and the JSON that describes it."""

    pipeline: ImbPipeline
    info: FinalModelInfo


def train_final_model(
    frame: pd.DataFrame,
    *,
    target: str,
    task_type: TaskType,
    preprocessor,
    model_name: str,
    use_smote: bool = False,
    strategy: FeatureStrategy | None = None,
    primary_metric: str = "",
    cv_score: float | None = None,
    random_seed: int | None = None,
) -> FinalModel:
    """Refit the leaderboard's winner on every row and describe the result.

    ``preprocessor`` is the same unfitted recipe cross-validation was handed. It
    is cloned before use, so the object on the pipeline state leaves this function
    exactly as unfitted as it arrived -- the artifact written in the preprocessing
    node stays the evidence it was registered to be.
    """
    settings = get_settings()
    if random_seed is None:
        random_seed = settings.random_seed

    features = [c for c in frame.columns if c != target]
    if not features:
        raise FinalTrainingError("No feature columns to train on.")

    warnings: list[str] = []
    candidate = _winning_candidate(model_name, task_type, random_seed, warnings)

    resampler = None
    resampling = "none"
    if use_smote:
        resampler = build_resampler(
            frame[target], task_type=task_type, random_seed=random_seed, warnings=warnings
        )
        if resampler is not None:
            # Worth stating in the artifact: on the full-dataset refit SMOTE has
            # no held-out rows to stay clear of, so the caveat that made it safe
            # in cross-validation does not arise. It is applied because the model
            # being served should be the configuration that was ranked, not a
            # different one that happens to share its name.
            resampling = "SMOTE, applied to the full training set"

    pipeline = build_pipeline(
        clone(preprocessor),
        task_type,
        random_seed=random_seed,
        estimator=candidate.estimator,
        resampler=resampler,
    )

    X = frame[features]
    y = frame[target]
    # The deliberate one. Every row, on purpose -- see the module docstring.
    pipeline.fit(X, y)

    classes = [str(c) for c in getattr(pipeline.named_steps["model"], "classes_", [])]
    logger.info(
        "Final model: %s refitted on all %d rows (%d raw features)",
        candidate.name,
        len(frame),
        len(features),
    )

    info = FinalModelInfo(
        model_name=candidate.name,
        task_type=task_type,
        target_column=target,
        n_rows=int(len(frame)),
        n_features=len(features),
        feature_columns=[_describe_column(frame[name], name, strategy) for name in features],
        classes=classes,
        primary_metric=primary_metric,
        cv_score=cv_score,
        resampling=resampling,
        warnings=warnings,
    )
    return FinalModel(pipeline=pipeline, info=info)


def _winning_candidate(model_name: str, task_type: TaskType, random_seed: int, warnings: list[str]):
    """Look the winner up in the roster by name, at the same seed it was ranked at.

    A name with no match should be impossible -- the leaderboard's names come from
    this same roster -- so the fallback exists to keep a rename from taking the
    whole pipeline down, and records itself rather than quietly serving a
    different model from the one the report names.
    """
    roster = build_roster(task_type, random_seed=random_seed)
    for candidate in roster:
        if candidate.name == model_name:
            return candidate

    warnings.append(
        f"The leaderboard's winner {model_name!r} is not in the current model "
        f"roster, so {roster[0].name} was trained instead."
    )
    logger.warning("Unknown winning model %r; falling back to %s", model_name, roster[0].name)
    return roster[0]


def _describe_column(
    values: pd.Series, name: str, strategy: FeatureStrategy | None
) -> PredictorColumn:
    """One input column as the prediction form needs to present it.

    The example value comes from the data rather than being invented, which is
    what makes a pre-filled form show plausible units -- a user guessing what
    ``tenure_months`` wants is a user who submits a bad prediction and blames the
    model.

    A column the strategy dropped is listed but marked unused. It is still part
    of the frame the recipe was fitted against, so it cannot simply be omitted --
    but asking someone for a customer ID the model discards would be worse than
    not asking at all.
    """
    chosen = strategy.for_column(name) if strategy is not None else None
    present = values.dropna()
    example = "" if present.empty else str(present.iloc[0])
    role = chosen.role if chosen is not None else "numeric"
    return PredictorColumn(
        name=name,
        dtype=str(values.dtype),
        role=role,
        example=example,
        used=role != "drop",
    )
