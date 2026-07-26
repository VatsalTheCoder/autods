"""Preprocessing -- builds the **unfitted** recipe, and never runs it (spec 7.6).

This is the module the whole project's methodological claim rests on. It returns
a ``ColumnTransformer`` that has been *constructed* and not *fitted*: a recipe,
not a cooked meal. Every number a transformer needs -- the imputers' medians, the
scaler's means, the encoder's category lists, the frequency encoder's
distribution -- is learned later, inside a single cross-validation fold, from
that fold's training rows only (``modeling.py``).

Get this backwards and nothing crashes. You simply report a better score than you
earned, on every dataset, forever (spec 8).

**Where the line actually sits.** This module reads three things about a column:
its name, its dtype, and its *vocabulary* -- which distinct labels exist, and how
many. It never computes a statistic. That distinction is the whole rule, and it
is sharper than "does not touch the data":

* Counting that a column holds 5,000 distinct values, or that an ordinal ordering
  covers every label present, is a question about the dataset's schema. The
  answer does not vary with which rows land in which fold, so nothing a model
  learns from it could encode the held-out rows.
* How *often* "London" appears, what the median age is, what the mean is -- these
  all vary by fold, and every one of them is computed by a transformer that this
  module only constructs. ``FrequencyEncoder`` is the instructive case: it looks
  like a lookup table and is in fact a fitted distribution, which is why it is a
  pipeline step and not a dict built here (``encoders.py``).

**Section 7 changed who chooses, not what is guaranteed.** Section 5 hardcoded
the strategies -- median-impute and scale numbers, mode-impute and one-hot encode
labels. Now a validated ``FeatureStrategy`` says what each column gets
(``agents/feature_strategy.py``), and this module builds it. The unfitted-output
contract is untouched, and passing no strategy still gives exactly the Section 5
recipe, which is what runs whenever there is no LLM available.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    MinMaxScaler,
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
)

from app.agents.feature_strategy import default_strategy
from app.ml.contracts import (
    ColumnStrategy,
    DroppedColumn,
    FeatureStrategy,
    PreprocessingSpec,
)
from app.ml.encoders import DatetimeFeatures, FrequencyEncoder

logger = logging.getLogger(__name__)

AGENT_NAME = "preprocessing"

_NUMERIC_STRATEGY = "median imputation, then standard scaling"
_CATEGORICAL_STRATEGY = "most-frequent imputation, then one-hot encoding (unknowns ignored)"


class PreprocessingError(RuntimeError):
    """No column had a strategy, so there is no recipe to build."""


@dataclass(slots=True)
class PreprocessingResult:
    """The unfitted transformer plus the JSON description of what is in it."""

    transformer: ColumnTransformer
    spec: PreprocessingSpec


def build_preprocessor(
    frame: pd.DataFrame,
    *,
    target: str,
    strategy: FeatureStrategy | None = None,
    select_k: int | None = None,
    task_type: str = "classification",
) -> PreprocessingResult:
    """Build an unfitted recipe for ``frame``'s feature columns.

    With no ``strategy`` every column falls back to its dtype default, which is
    the Section 5 behaviour and what runs when no LLM is configured.

    ``select_k`` appends a feature selector *inside* the returned pipeline, so it
    is fitted per fold like every other step -- see ``_with_selection``.
    """
    features = [c for c in frame.columns if c != target]
    if strategy is None:
        strategy = FeatureStrategy(
            columns=[default_strategy(name, frame[name]) for name in features],
            defaulted_columns=list(features),
        )

    # Only strategies for columns still present. Cleaning may have dropped a
    # column between the strategy being chosen and the recipe being built, and a
    # ColumnTransformer naming a column that is gone fails at fit time -- inside a
    # fold, where the error is least legible.
    chosen = [c for c in strategy.columns if c.column in set(features)]

    by_role: dict[str, list[ColumnStrategy]] = defaultdict(list)
    for item in chosen:
        by_role[item.role].append(item)

    transformers: list[tuple[str, SklearnPipeline, list[str]]] = []
    transformers += _numeric_branches(by_role["numeric"])
    transformers += _categorical_branches(by_role["categorical"])
    transformers += _ordinal_branches(by_role["ordinal"])
    transformers += _datetime_branches(by_role["datetime"])
    transformers += _text_branches(by_role["text"])

    if not transformers:
        raise PreprocessingError(
            "No column could be preprocessed: every feature was dropped or had no "
            "usable strategy, so there is nothing to train on."
        )

    transformer = ColumnTransformer(
        transformers=transformers,
        # Anything without an explicit strategy is dropped rather than passed
        # through raw -- a stray text column reaching the model as-is would fail
        # the fit, and silently forwarding untransformed data is how "it worked on
        # my dataset" bugs get in.
        remainder="drop",
        verbose_feature_names_out=False,
    )

    spec = _describe(strategy, by_role, select_k)
    logger.info(
        "Preprocessing recipe (unfitted, %s): %s",
        strategy.source,
        ", ".join(f"{len(v)} {k}" for k, v in sorted(by_role.items()) if v) or "nothing",
    )
    return PreprocessingResult(
        transformer=_with_selection(transformer, select_k, task_type), spec=spec
    )


# ---- One branch per group of columns sharing a recipe ------------------------
#
# Columns with identical strategies share a transformer rather than getting one
# each: five median-imputed, standard-scaled numbers are one branch over five
# columns, not five branches. Fewer objects to clone per fold, and a
# ``preprocessing_pipeline.pkl`` a human can actually read.


def _numeric_branches(items: list[ColumnStrategy]) -> list[tuple]:
    branches = []
    for index, ((impute, scale), columns) in enumerate(_grouped(items, "impute", "scale"), start=1):
        steps = []
        if impute != "none":
            steps.append(("impute", _imputer(impute)))
        if scale != "none":
            steps.append(("scale", _scaler(scale)))
        # A column asking for neither still needs a step, or the branch is an
        # empty Pipeline scikit-learn will not build. The identity transform is
        # the honest one here: the strategy asked for the column untouched, and
        # gate 2 has already established it has no gaps to fill.
        steps = steps or [("passthrough", FunctionTransformer())]
        branches.append((f"numeric_{index}", SklearnPipeline(steps=steps), columns))
    return branches


def _categorical_branches(items: list[ColumnStrategy]) -> list[tuple]:
    branches = []
    for index, ((impute, encode, scale), columns) in enumerate(
        _grouped(items, "impute", "encode", "scale"), start=1
    ):
        steps = [("impute", _imputer(impute, fill="missing"))] if impute != "none" else []
        if encode == "frequency":
            steps.append(("encode", FrequencyEncoder()))
            if scale != "none":
                steps.append(("scale", _scaler(scale)))
        else:
            # ``handle_unknown="ignore"`` is not a nicety, it is what makes
            # per-fold fitting survive: a category that appears only in the
            # held-out fold is genuinely unknown to a correctly-fitted encoder,
            # and the alternative is a crash on every dataset with a rare
            # category. Ignoring it encodes the row as all-zeros for that
            # feature, which is the honest representation.
            steps.append(("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)))
        branches.append((f"categorical_{index}", SklearnPipeline(steps=steps), columns))
    return branches


def _ordinal_branches(items: list[ColumnStrategy]) -> list[tuple]:
    """One branch per column: the category ordering differs for each.

    ``handle_unknown="use_encoded_value"`` with -1 is the ordinal counterpart of
    the one-hot path's ``ignore``. A label absent from the training fold has no
    rank the encoder can honestly assign, and -1 places it below every known
    level rather than crashing the fold.
    """
    branches = []
    for index, item in enumerate(items, start=1):
        steps = []
        if item.impute != "none":
            steps.append(("impute", _imputer(item.impute, fill="missing")))
        steps.append(
            (
                "encode",
                OrdinalEncoder(
                    categories=[list(item.ordinal_order)],
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                ),
            )
        )
        if item.scale != "none":
            steps.append(("scale", _scaler(item.scale)))
        branches.append((f"ordinal_{index}", SklearnPipeline(steps=steps), [item.column]))
    return branches


def _datetime_branches(items: list[ColumnStrategy]) -> list[tuple]:
    """Calendar parts first, then impute and scale them as the numbers they are."""
    branches = []
    for index, ((impute, scale), columns) in enumerate(_grouped(items, "impute", "scale"), start=1):
        steps = [("parts", DatetimeFeatures())]
        # Imputation comes *after* extraction: a NaT has no year to fill, but the
        # NaN year it produces has a median like any other numeric column.
        steps.append(("impute", _imputer(impute if impute != "none" else "median")))
        if scale != "none":
            steps.append(("scale", _scaler(scale)))
        branches.append((f"datetime_{index}", SklearnPipeline(steps=steps), columns))
    return branches


def _text_branches(items: list[ColumnStrategy]) -> list[tuple]:
    """High-cardinality columns: one output column each, not hundreds."""
    branches = []
    for index, ((impute, scale), columns) in enumerate(_grouped(items, "impute", "scale"), start=1):
        steps = [("impute", _imputer(impute, fill="missing"))] if impute != "none" else []
        steps.append(("encode", FrequencyEncoder()))
        if scale != "none":
            steps.append(("scale", _scaler(scale)))
        branches.append((f"text_{index}", SklearnPipeline(steps=steps), columns))
    return branches


def _grouped(items: list[ColumnStrategy], *fields: str) -> list[tuple[tuple, list[str]]]:
    """Group columns by the strategy fields that decide their recipe.

    Insertion-ordered so the built transformer follows the dataset's column
    order, which keeps two runs over the same data byte-comparable.
    """
    groups: dict[tuple, list[str]] = {}
    for item in items:
        key = tuple(getattr(item, field) for field in fields)
        groups.setdefault(key, []).append(item.column)
    return list(groups.items())


def _imputer(strategy: str, *, fill: str | None = None) -> SimpleImputer:
    """Median rather than mean is the default because outliers do not move it.

    ``constant`` fills labels with the literal string "missing", which makes
    absence its own category -- often signal in itself, and always visible in the
    one-hot output rather than being folded into whichever level was commonest.
    """
    if strategy == "constant":
        return SimpleImputer(strategy="constant", fill_value=fill if fill is not None else 0)
    return SimpleImputer(strategy=strategy)


def _scaler(strategy: str):
    """Standard for most things; min-max for genuinely bounded quantities.

    Scaling is here for the models that need it -- the linear and distance-based
    members of the Section 7 roster -- and is harmless for the tree ensembles that
    do not. One recipe for both means the roster cannot acquire a model whose
    preprocessing was silently missing.
    """
    return MinMaxScaler() if strategy == "minmax" else StandardScaler()


# ---- Feature selection, as a step rather than a separate pass ----------------


def _with_selection(
    transformer: ColumnTransformer,
    select_k: int | None,
    task_type: str,
) -> ColumnTransformer | SklearnPipeline:
    """Append a selector, or hand back the transformer untouched.

    Feature selection is a fitted step, and a badly placed one leaks as
    thoroughly as a badly placed scaler: choosing "the 20 columns most correlated
    with the target" over the whole dataset uses the held-out targets to decide
    what the model gets to see, and every subsequent score is inflated. Ranking
    the branch as a *step in the pipeline* means it is fitted on training rows
    only, per fold, like everything else -- so the folds may legitimately select
    different columns, which is what an honest run looks like.

    The return type widens to a ``Pipeline`` when selection is on. Callers treat
    it as an opaque unfitted transformer, which both types are.
    """
    if not select_k:
        return transformer

    score_func = f_classif if task_type == "classification" else f_regression
    return SklearnPipeline(
        steps=[
            ("columns", transformer),
            # ``k`` may exceed the number of columns the recipe produces on a
            # narrow dataset; "all" is scikit-learn's own way of saying "keep
            # everything", and is what SelectKBest does rather than raising.
            ("select", SelectKBest(score_func=score_func, k=select_k)),
        ]
    )


# ---- The JSON half of the contract ------------------------------------------


def _describe(
    strategy: FeatureStrategy,
    by_role: dict[str, list[ColumnStrategy]],
    select_k: int | None,
) -> PreprocessingSpec:
    """Say what is in the recipe, in the shape the report and the UI read.

    Nothing here records whether the pipeline is fitted, because a self-reported
    flag would prove nothing -- the tests assert unfittedness against the real
    object instead.
    """
    names = {role: [c.column for c in items] for role, items in by_role.items()}
    return PreprocessingSpec(
        numeric_columns=names.get("numeric", []),
        categorical_columns=names.get("categorical", []),
        ordinal_columns=names.get("ordinal", []),
        datetime_columns=names.get("datetime", []),
        text_columns=names.get("text", []),
        unhandled_columns=[
            DroppedColumn(name=item.column, reason=item.rationale or "dropped by the strategy")
            for item in by_role.get("drop", [])
        ],
        numeric_strategy=_NUMERIC_STRATEGY if names.get("numeric") else "",
        categorical_strategy=_CATEGORICAL_STRATEGY if names.get("categorical") else "",
        column_strategies=[c for c in strategy.columns if c.role != "drop"],
        strategy_source="llm" if strategy.source == "llm" else "hardcoded",
        feature_selection=(
            f"top {select_k} features by univariate ANOVA F-score, fitted per fold"
            if select_k
            else ""
        ),
    )
