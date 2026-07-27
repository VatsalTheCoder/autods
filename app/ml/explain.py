"""SHAP over the final model, in the user's own column names (spec 7.10).

This is the project's explainability claim, and almost all of its difficulty is
in one place that sounds trivial: **the model does not think in the user's
columns.** By the time an estimator sees the data, ``city`` has become
``city_London``, ``city_Leeds`` and three more; ``signed_up`` has become five
calendar numbers; ``email`` has become ``email_frequency``. SHAP faithfully
explains those. A user asked to read ``city_Glasgow: +0.03`` five times over has
not been given an explanation, they have been given the recipe's internals.

So the mapping back is the substance of this module, and it is done by walking
the fitted ``ColumnTransformer``'s branches rather than by parsing names:

1. Every branch knows the source columns it was given, and each branch's own
   ``get_feature_names_out`` knows what it produced from them.
2. Each output is attributed to the **longest** source column that prefixes it.
   Longest, not first: with ``city`` and ``city_tier`` both present,
   ``city_tier_London`` prefix-matches both, and only the longer one is right.
3. The mapping is published in the artifact (``feature_name_mapping``), so a
   reader can check it instead of trusting it. It is the part of Section 8 most
   likely to be quietly wrong, and quietly wrong is the worst way for an
   explanation to fail -- a mislabelled feature is more damaging than a missing
   one, because it gets believed.

**Four things about the SHAP API this module has to get right**, each established
by a compatibility spike against shap 0.52 rather than assumed:

* ``TreeExplainer`` rejects ``LabelEncodedClassifier`` (the XGBoost wrapper) but
  accepts the estimator inside it, which ``fit`` exposes as ``estimator_``. The
  wrapper does not need to change; this module unwraps it.
* Multiclass SHAP values come back as ``(rows, features, classes)`` where the
  class axis is in the **model's own encoded order**. Index ``k`` means
  ``classes_[k]``, and anything user-facing has to decode through it or every
  explanation is labelled with the wrong class.
* ``expected_value`` must be read *after* ``shap_values``. On a fresh
  ``TreeExplainer`` it is still the booster's ``base_score``; the call replaces it
  with the dataset's mean margin. Reading it early breaks additivity by about
  0.05 -- small enough to look like rounding, which is exactly why it is worth a
  comment.
* Dispatch is on model family. Linear models need ``LinearExplainer`` with a
  background matrix; ``TreeExplainer`` refuses them outright.

Everything here is descriptive, like EDA: nothing downstream consumes it, so a
dataset that defeats the explainer costs its explanation and not its model.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from xgboost import XGBClassifier, XGBRegressor

from app.agents.schema_models import TaskType
from app.core.config import get_settings
from app.ml.contracts import (
    ExplainabilityReport,
    FeatureImportance,
    LocalContribution,
    LocalExplanation,
)
from app.ml.modeling import LabelEncodedClassifier, preprocessing_of
from app.ml.plots import Chart, render_shap_charts
from app.ml.sampling import sample_frame

logger = logging.getLogger(__name__)

AGENT_NAME = "explainability"

# The roster, split by which explainer each family needs. Written as concrete
# classes rather than name matching so that adding a model to ``build_roster``
# without deciding how it is explained fails loudly here instead of silently
# producing no explanation.
_TREE_MODELS = (
    RandomForestClassifier,
    RandomForestRegressor,
    XGBClassifier,
    XGBRegressor,
    LGBMClassifier,
    LGBMRegressor,
)
_LINEAR_MODELS = (LogisticRegression, LinearRegression)

# How many features one local explanation lists individually before the rest are
# summed into "everything else". Ten lines is a readable waterfall; forty is a
# table nobody reads, and dropping the remainder silently would make the
# contributions stop adding up to the prediction.
_LOCAL_FEATURE_LIMIT = 10


class ExplainabilityError(RuntimeError):
    """SHAP could not be run on this model at all, with a reason for the user."""


@dataclass(slots=True)
class ExplainabilityResult:
    """The report and the charts that go with it; the node stores both."""

    report: ExplainabilityReport
    charts: list[Chart] = field(default_factory=list)


# ---- The feature-name mapping -----------------------------------------------


def column_transformer_of(preprocess) -> ColumnTransformer:
    """Find the ``ColumnTransformer`` inside whatever the recipe turned out to be.

    ``build_preprocessor`` returns a bare ``ColumnTransformer``, or -- when the
    planner asked for feature selection -- a ``Pipeline`` of one wrapped around a
    ``SelectKBest``. Both are handed around as "the unfitted recipe", so the one
    place that needs to see inside does the unwrapping.
    """
    if isinstance(preprocess, ColumnTransformer):
        return preprocess
    steps = getattr(preprocess, "named_steps", {})
    if "columns" in steps and isinstance(steps["columns"], ColumnTransformer):
        return steps["columns"]
    raise ExplainabilityError(
        "The fitted recipe has no ColumnTransformer, so encoded features cannot "
        "be traced back to the columns they came from."
    )


def source_columns(preprocess) -> dict[str, str]:
    """Map every encoded feature name to the source column it came from.

    Built from the fitted transformer rather than from the strategy, because the
    strategy says what each column was *asked* to become and this has to describe
    what it actually did -- a one-hot encoder's output width is a property of the
    data it saw, not of the request.
    """
    transformer = column_transformer_of(preprocess)
    mapping: dict[str, str] = {}

    for name, branch, columns in transformer.transformers_:
        if name == "remainder" or branch in ("drop", "passthrough"):
            # ``remainder="drop"`` is the recipe's default and produces nothing;
            # a passthrough remainder would produce columns under their own
            # names, which need no mapping.
            continue

        sources = [str(c) for c in columns]
        try:
            produced = [str(c) for c in branch.get_feature_names_out(sources)]
        except (AttributeError, ValueError, TypeError) as exc:  # pragma: no cover
            # A branch that cannot name its own output leaves its columns
            # unmapped rather than taking the whole explanation down.
            logger.warning("Branch %s could not name its output: %s", name, exc)
            continue

        for encoded in produced:
            mapping[encoded] = _longest_source(encoded, sources)

    return mapping


def _longest_source(encoded: str, sources: list[str]) -> str:
    """Attribute one encoded name to its source column by longest prefix.

    Exact match first, since a numeric branch passes names through untouched.
    Then the longest ``<column>_`` prefix, which is the disambiguating rule: both
    ``city`` and ``city_tier`` prefix ``city_tier_London``, and the longer one is
    the column that actually produced it.

    An unmatched name maps to itself. That happens only if a branch invents a
    name unrelated to its inputs, in which case showing the encoded name is more
    honest than attributing it to a column at random.
    """
    if encoded in sources:
        return encoded
    matches = [c for c in sources if encoded.startswith(f"{c}_")]
    if not matches:
        return encoded
    return max(matches, key=len)


# ---- Dispatch ----------------------------------------------------------------


def _unwrap(model):
    """The estimator SHAP can actually read.

    ``LabelEncodedClassifier`` exists so XGBoost can be given integer labels
    (``modeling.py``); ``TreeExplainer`` sees a class it does not recognise and
    refuses it. The wrapper keeps the fitted booster on ``estimator_``, so
    unwrapping here is enough and the wrapper needs no SHAP-specific code.
    """
    if isinstance(model, LabelEncodedClassifier):
        return model.estimator_
    return model


def _build_explainer(model, background: np.ndarray):
    """Pick the explainer for this model family, or say why there is none.

    ``TreeExplainer`` is chosen wherever it applies because it is *exact*: it
    reads the fitted trees and computes the decomposition, rather than sampling
    perturbations and approximating it. ``LinearExplainer`` is likewise exact for
    a linear model given the background distribution its coefficients are
    centred against -- which is why it needs the training matrix and the tree
    explainer does not.
    """
    import shap  # imported lazily: a heavy dependency only this module needs

    if isinstance(model, _TREE_MODELS):
        return shap.TreeExplainer(model), "TreeExplainer"
    if isinstance(model, _LINEAR_MODELS):
        return shap.LinearExplainer(model, background), "LinearExplainer"
    raise ExplainabilityError(
        f"No SHAP explainer is defined for {type(model).__name__}; the model was "
        "trained and saved, but it could not be explained."
    )


def _margin_reference(model, matrix: np.ndarray) -> np.ndarray | None:
    """The model output SHAP values are supposed to add up to, per family.

    Each explainer decomposes a specific output, and it is not the same one
    everywhere: ``TreeExplainer`` on a random forest works in probability space,
    on a gradient-booster in raw margin (log-odds), and ``LinearExplainer`` on a
    logistic regression in the decision function. Getting this reference right is
    the whole point of checking additivity at all -- comparing against the wrong
    output would report a failure on a perfectly good explanation.

    Returns ``None`` for a family with no defined reference, which is recorded as
    "not checked" rather than reported as an error of zero.
    """
    try:
        if isinstance(model, RandomForestClassifier):
            return np.asarray(model.predict_proba(matrix), dtype=float)
        if isinstance(model, XGBClassifier):
            return np.asarray(model.predict(matrix, output_margin=True), dtype=float)
        if isinstance(model, LGBMClassifier):
            return np.asarray(model.predict_proba(matrix, raw_score=True), dtype=float)
        if isinstance(model, LogisticRegression):
            return np.asarray(model.decision_function(matrix), dtype=float)
        if isinstance(model, LGBMRegressor):
            return np.asarray(model.predict(matrix, raw_score=True), dtype=float)
        if isinstance(model, (RandomForestRegressor, XGBRegressor, LinearRegression)):
            return np.asarray(model.predict(matrix), dtype=float)
    except (AttributeError, TypeError, ValueError) as exc:  # pragma: no cover
        logger.warning("Additivity reference unavailable for %s: %s", type(model).__name__, exc)
    return None


# ---- The main entry point ----------------------------------------------------


def explain_model(
    pipeline,
    frame: pd.DataFrame,
    *,
    target: str,
    task_type: TaskType,
    model_name: str,
    max_rows: int | None = None,
    top_features: int | None = None,
    n_examples: int | None = None,
) -> ExplainabilityResult:
    """Explain a **fitted** pipeline over ``frame``, in the user's column names.

    Takes the fitted final model rather than refitting anything: the explanation
    has to be of the model that will actually be served, or it is an explanation
    of something else.
    """
    settings = get_settings()
    max_rows = settings.shap_max_rows if max_rows is None else max_rows
    top_features = settings.shap_top_features if top_features is None else top_features
    n_examples = settings.shap_local_examples if n_examples is None else n_examples

    warnings: list[str] = []
    features = [c for c in frame.columns if c != target]
    if not features:
        raise ExplainabilityError("No feature columns to explain.")

    sampled = sample_frame(frame, target=target, task_type=task_type, limit=max_rows)
    sampling_note = ""
    if len(sampled) < len(frame):
        sampling_note = (
            f"SHAP was computed on a sample of {len(sampled):,} rows drawn from "
            f"{len(frame):,}. Mean absolute contributions are stable well below "
            "this many rows; the model itself was trained on every row."
        )

    X = sampled[features]
    # Not ``named_steps["preprocess"]``: with feature selection on, the recipe is
    # spliced into the pipeline as two steps rather than one (``modeling.py``).
    preprocess = preprocessing_of(pipeline)
    model = pipeline.named_steps["model"]

    matrix = np.asarray(_dense(preprocess.transform(X)), dtype=float)
    encoded_names = [str(n) for n in preprocess.get_feature_names_out()]
    if matrix.shape[1] != len(encoded_names):  # pragma: no cover - defensive
        raise ExplainabilityError(
            f"The recipe produced {matrix.shape[1]} columns but named "
            f"{len(encoded_names)}; feature names cannot be trusted."
        )

    mapping = source_columns(preprocess)
    unmapped = [n for n in encoded_names if n not in mapping]
    if unmapped:
        warnings.append(
            f"{len(unmapped)} encoded feature(s) could not be traced back to a "
            "source column and are shown under their encoded names: " + ", ".join(unmapped[:5])
        )
    origins = [mapping.get(name, name) for name in encoded_names]

    estimator = _unwrap(model)
    explainer, explainer_name = _build_explainer(estimator, matrix)

    values = _normalise(explainer.shap_values(matrix), n_rows=matrix.shape[0])
    # After ``shap_values``, never before -- see the module docstring.
    expected = np.atleast_1d(np.asarray(explainer.expected_value, dtype=float))
    if expected.size != values.shape[2]:
        # A single base value against several output columns is legitimate for
        # some families; broadcasting keeps the arithmetic below uniform.
        expected = np.resize(expected, values.shape[2])

    classes = [str(c) for c in getattr(model, "classes_", [])]
    additivity, additivity_note = _check_additivity(estimator, matrix, values, expected)
    if additivity_note:
        warnings.append(additivity_note)

    importance = _global_importance(values, origins, encoded_names, matrix, top_features)
    examples = _local_examples(
        pipeline,
        sampled,
        X,
        values=values,
        expected=expected,
        origins=origins,
        classes=classes,
        task_type=task_type,
        n_examples=n_examples,
    )

    focus = _focus_class(values.shape[2])
    report = ExplainabilityReport(
        model_name=model_name,
        task_type=task_type,
        target_column=target,
        explainer=explainer_name,
        n_rows_explained=int(matrix.shape[0]),
        n_encoded_features=len(encoded_names),
        sampling_note=sampling_note,
        classes=classes,
        aggregation=_aggregation_note(values.shape[2], classes),
        global_importance=importance,
        examples=examples,
        feature_name_mapping={name: mapping.get(name, name) for name in encoded_names},
        additivity_max_error=additivity,
        warnings=warnings,
    )

    charts = render_shap_charts(
        importance=importance,
        shap_values=values[:, :, focus],
        feature_values=matrix,
        encoded_names=encoded_names,
        origins=origins,
        examples=examples,
        class_label=_class_label(focus, values.shape[2], classes),
        target=target,
    )
    report.plots = [chart.name for chart in charts]

    logger.info(
        "Explained %s with %s over %d rows; top feature %s",
        model_name,
        explainer_name,
        matrix.shape[0],
        importance[0].feature if importance else "(none)",
    )
    return ExplainabilityResult(report=report, charts=charts)


# ---- Shapes and arithmetic ---------------------------------------------------


def _dense(matrix):
    """SHAP wants a dense array; a one-hot branch may hand back a sparse one."""
    if hasattr(matrix, "toarray"):
        return matrix.toarray()
    return np.asarray(matrix)


def _normalise(values, *, n_rows: int) -> np.ndarray:
    """Give every case the same ``(rows, features, outputs)`` shape.

    SHAP returns three different things depending on the model: a 2-D array for
    regression and for gradient-boosted binary classification, a 3-D array for
    multiclass, and -- on older versions -- a list of 2-D arrays, one per class.
    Normalising once here means nothing downstream has to branch on it, and the
    single-output case becomes a class axis of length one rather than a special
    case.
    """
    if isinstance(values, list):
        values = np.stack([np.asarray(v, dtype=float) for v in values], axis=-1)
    values = np.asarray(values, dtype=float)
    if values.ndim == 2:
        values = values[:, :, np.newaxis]
    if values.shape[0] != n_rows:  # pragma: no cover - defensive
        raise ExplainabilityError(
            f"SHAP returned {values.shape[0]} rows of values for {n_rows} rows of data."
        )
    return values


def _focus_class(n_outputs: int) -> int:
    """Which output the value-level charts show.

    For a binary classifier the last class is the conventional "positive" one and
    the two columns are mirror images, so it makes no difference which is drawn
    beyond the label. For a multiclass model the first class is shown and the
    chart says which -- a beeswarm cannot show five classes at once, and the
    global ranking above it is already aggregated across all of them.
    """
    return n_outputs - 1 if n_outputs <= 2 else 0


def _class_label(output_index: int, n_outputs: int, classes: list[str]) -> str:
    """The class an output column is *about*, which is not always its index.

    The trap: a gradient-booster or a logistic regression on a binary target
    produces a single output -- the margin towards the **positive** class -- while
    ``classes_`` still has two entries. Labelling that output ``classes_[0]``
    reverses the meaning of every chart drawn from it, and nothing about the
    result looks wrong. A random forest, which produces one column per class,
    does not have the problem, so the two cases have to be distinguished rather
    than handled by one index.
    """
    if not classes:
        return ""
    if n_outputs == 1:
        return classes[-1] if len(classes) == 2 else ""
    return classes[output_index] if output_index < len(classes) else ""


def _aggregation_note(n_outputs: int, classes: list[str]) -> str:
    """Say in words how per-class values became one ranking."""
    if n_outputs <= 1:
        return "Mean absolute SHAP value per feature, over the explained rows."
    return (
        "Mean absolute SHAP value per feature, averaged across the "
        f"{len(classes) or n_outputs} classes and over the explained rows -- so a "
        "feature that matters for one class only is not hidden by the others."
    )


def _check_additivity(
    estimator,
    matrix: np.ndarray,
    values: np.ndarray,
    expected: np.ndarray,
) -> tuple[float, str]:
    """Confirm the contributions actually add up to the model's output.

    SHAP's guarantee is additive: base value plus every feature's contribution
    equals the model's output for that row. If that does not hold, the numbers
    are not a decomposition of anything and the whole report is decoration. It
    costs one extra forward pass to check, so it is checked, and the largest gap
    found goes in the artifact next to the numbers it validates.
    """
    reference = _margin_reference(estimator, matrix)
    if reference is None:
        return 0.0, (
            f"Additivity could not be verified for {type(estimator).__name__}: "
            "the model exposes no raw output to compare the contributions against."
        )

    reference = reference.reshape(matrix.shape[0], -1)
    if reference.shape[1] != values.shape[2]:
        return 0.0, (
            "Additivity could not be verified: the model's output has "
            f"{reference.shape[1]} columns and SHAP produced {values.shape[2]}."
        )

    reconstructed = values.sum(axis=1) + expected[np.newaxis, :]
    error = float(np.max(np.abs(reconstructed - reference)))

    # A tenth of a percent of the output's own scale. Absolute tolerances are
    # useless here: log-odds run to +-10 while probabilities live in [0, 1].
    tolerance = max(1e-3, 1e-3 * float(np.max(np.abs(reference))))
    if error > tolerance:
        return error, (
            f"SHAP contributions do not fully reconstruct the model's output "
            f"(largest gap {error:.4f}). Treat the magnitudes below as indicative "
            "rather than exact."
        )
    return error, ""


def _global_importance(
    values: np.ndarray,
    origins: list[str],
    encoded_names: list[str],
    matrix: np.ndarray,
    limit: int,
) -> list[FeatureImportance]:
    """Rank **source columns** by mean absolute contribution.

    Summed over the encoded features a column produced, not averaged: a
    five-level one-hot column influences a prediction through all five of its
    outputs at once, and averaging would make a wide column look weak precisely
    because it is expressive. Averaging *across classes* first (inside
    ``per_feature``) is the opposite case -- there the classes are alternative
    views of the same prediction, not separate contributions.
    """
    per_feature = np.abs(values).mean(axis=0).mean(axis=1)

    totals: dict[str, float] = {}
    members: dict[str, list[str]] = {}
    for name, origin, score in zip(encoded_names, origins, per_feature, strict=True):
        totals[origin] = totals.get(origin, 0.0) + float(score)
        members.setdefault(origin, []).append(name)

    grand_total = sum(totals.values())
    ranked = sorted(totals.items(), key=lambda pair: pair[1], reverse=True)[:limit]

    columns = {name: index for index, name in enumerate(encoded_names)}
    return [
        FeatureImportance(
            feature=origin,
            importance=score,
            share=(score / grand_total) if grand_total > 0 else 0.0,
            encoded_features=members[origin],
            direction=_direction(origin, members[origin], columns, matrix, values),
        )
        for origin, score in ranked
    ]


def _direction(
    origin: str,
    encoded: list[str],
    columns: dict[str, int],
    matrix: np.ndarray,
    values: np.ndarray,
) -> str:
    """Which way a column pushes, but only where that question has an answer.

    Well defined for a column that survived preprocessing as a single number of
    its own: the sign of the correlation between its values and its SHAP values
    says whether more of it raises or lowers the output. Not well defined for a
    one-hot column, where each level pushes its own way, or for a frequency
    encoding, where "higher" means "more common" rather than "more" -- so those
    get an empty string instead of a sentence that would read as a finding.
    """
    if len(encoded) != 1 or encoded[0] != origin:
        return ""

    index = columns[encoded[0]]
    column = matrix[:, index]
    contribution = values[:, index, _focus_class(values.shape[2])]
    if np.std(column) == 0 or np.std(contribution) == 0:
        return ""

    correlation = float(np.corrcoef(column, contribution)[0, 1])
    if abs(correlation) < 0.3:
        return "its effect is not consistently in one direction"
    return (
        "higher values push the prediction up"
        if correlation > 0
        else "higher values push the prediction down"
    )


# ---- Individual predictions --------------------------------------------------


def _local_examples(
    pipeline,
    sampled: pd.DataFrame,
    X: pd.DataFrame,
    *,
    values: np.ndarray,
    expected: np.ndarray,
    origins: list[str],
    classes: list[str],
    task_type: TaskType,
    n_examples: int,
) -> list[LocalExplanation]:
    """Explain a few individual rows -- the half of SHAP a global chart cannot show.

    The rows are chosen to be *representative rather than convenient*: one
    confidently-predicted row per class for a classifier, and the extremes plus
    the middle of the predicted range for a regressor. Taking the first three
    rows of the file would explain whatever the dataset happened to be sorted by.
    """
    if n_examples <= 0 or values.shape[0] == 0:
        return []

    predictions = np.asarray(pipeline.predict(X))
    probabilities = _probabilities(pipeline, X)
    chosen = _representative_rows(
        predictions, probabilities, task_type=task_type, n_examples=n_examples
    )

    explanations: list[LocalExplanation] = []
    for position in chosen:
        class_index = _predicted_output(position, predictions, probabilities, classes, values)
        explanations.append(
            _explain_row(
                position=position,
                class_index=class_index,
                sampled=sampled,
                values=values,
                expected=expected,
                origins=origins,
                classes=classes,
                predictions=predictions,
                probabilities=probabilities,
                task_type=task_type,
            )
        )
    return explanations


def _probabilities(pipeline, X: pd.DataFrame) -> np.ndarray | None:
    if not hasattr(pipeline, "predict_proba"):
        return None
    try:
        return np.asarray(pipeline.predict_proba(X), dtype=float)
    except (AttributeError, ValueError):  # pragma: no cover - estimator-specific
        return None


def _representative_rows(
    predictions: np.ndarray,
    probabilities: np.ndarray | None,
    *,
    task_type: TaskType,
    n_examples: int,
) -> list[int]:
    """Row positions worth explaining, per the rule in ``_local_examples``."""
    n_rows = len(predictions)
    if task_type == "classification" and probabilities is not None:
        rows: list[int] = []
        # Most-confident row per class, in class order, until the budget runs
        # out. A class the model never predicts still gets its best candidate,
        # which is usually the more interesting explanation of the two.
        for class_index in range(probabilities.shape[1]):
            if len(rows) >= n_examples:
                break
            order = np.argsort(-probabilities[:, class_index])
            pick = next((int(i) for i in order if int(i) not in rows), None)
            if pick is not None:
                rows.append(pick)
        return rows

    order = np.argsort(np.asarray(predictions, dtype=float))
    candidates = [int(order[-1]), int(order[0]), int(order[len(order) // 2])]
    seen: list[int] = []
    for row in candidates[:n_examples]:
        if row not in seen and row < n_rows:
            seen.append(row)
    return seen


def _predicted_output(
    position: int,
    predictions: np.ndarray,
    probabilities: np.ndarray | None,
    classes: list[str],
    values: np.ndarray,
) -> int:
    """The output column whose decomposition explains *this row's* answer.

    The class axis is in the model's encoded order, so the predicted label is
    matched against ``classes_`` rather than assumed to be at any fixed index --
    getting this wrong would explain each row in terms of a class it was not
    predicted to be, and nothing about the output would look wrong.
    """
    n_outputs = values.shape[2]
    if n_outputs == 1:
        return 0
    if classes:
        label = str(predictions[position])
        if label in classes:
            index = classes.index(label)
            if index < n_outputs:
                return index
    if probabilities is not None and probabilities.shape[1] == n_outputs:
        return int(np.argmax(probabilities[position]))
    return n_outputs - 1


def _explain_row(
    *,
    position: int,
    class_index: int,
    sampled: pd.DataFrame,
    values: np.ndarray,
    expected: np.ndarray,
    origins: list[str],
    classes: list[str],
    predictions: np.ndarray,
    probabilities: np.ndarray | None,
    task_type: TaskType,
) -> LocalExplanation:
    """One row's contributions, aggregated to source columns and made to add up."""
    row_values = values[position, :, class_index]

    totals: dict[str, float] = {}
    for origin, contribution in zip(origins, row_values, strict=True):
        totals[origin] = totals.get(origin, 0.0) + float(contribution)

    ranked = sorted(totals.items(), key=lambda pair: abs(pair[1]), reverse=True)
    shown, rest = ranked[:_LOCAL_FEATURE_LIMIT], ranked[_LOCAL_FEATURE_LIMIT:]

    row = sampled.iloc[position]
    contributions = [
        LocalContribution(
            feature=name,
            value=_display(row[name]) if name in sampled.columns else "",
            contribution=score,
        )
        for name, score in shown
    ]

    base = float(expected[class_index])
    predicted_label = str(predictions[position])
    # The confidence in the model's *answer*, looked up by label. Reading it at
    # the SHAP output index instead would be right for a random forest and wrong
    # for every single-output model, where output 0 is the positive class's
    # margin regardless of which way the row was actually predicted.
    probability = None
    if probabilities is not None and predicted_label in classes:
        label_index = classes.index(predicted_label)
        if label_index < probabilities.shape[1]:
            probability = float(probabilities[position, label_index])

    return LocalExplanation(
        # The DataFrame's own index, not the position: it is the row number the
        # user can find in their file, which a position into a sample is not.
        row_label=str(sampled.index[position]),
        predicted=predicted_label,
        probability=probability,
        explained_class=(
            _class_label(class_index, values.shape[2], classes)
            if task_type == "classification"
            else ""
        ),
        base_value=base,
        contributions=contributions,
        other_contribution=float(sum(score for _, score in rest)),
        output_value=base + float(sum(totals.values())),
    )


def _display(value) -> str:
    """A cell as a person would read it, not as NumPy repr would print it."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "(missing)"
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.4g}"
    return str(value)
