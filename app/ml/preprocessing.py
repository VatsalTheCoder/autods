"""Preprocessing -- builds the **unfitted** recipe, and never runs it (spec 7.6).

This is the module the whole project's methodological claim rests on. It returns
a ``ColumnTransformer`` that has been *constructed* and not *fitted*: a recipe,
not a cooked meal. Nothing here touches a value in the dataset. The imputers'
medians, the scaler's means, the encoder's category lists -- every one of those
is learned later, inside a single cross-validation fold, from that fold's
training rows only (``modeling.py``).

Get this backwards and nothing crashes. You simply report a better score than
you earned, on every dataset, forever (spec 8). So the contract is deliberately
narrow: this function reads the frame's *column names and dtypes* to decide which
strategy each column gets, and reads no cell values at all.

The strategies themselves are hardcoded in Section 5 -- median-impute and scale
numbers, mode-impute and one-hot encode categories. The LLM does not choose them
yet; that is Section 7, which will replace the constants below with a validated
JSON decision while leaving this module's unfitted-output contract untouched.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from app.ml.contracts import DroppedColumn, PreprocessingSpec

logger = logging.getLogger(__name__)

AGENT_NAME = "preprocessing"

# Above this many distinct values, one-hot encoding stops being reasonable -- it
# would add hundreds of near-empty columns and invite overfitting. Such columns
# are left out in Section 5 and get target/frequency encoding in Section 7.
_MAX_ONEHOT_CARDINALITY = 50

_NUMERIC_STRATEGY = "median imputation, then standard scaling"
_CATEGORICAL_STRATEGY = "most-frequent imputation, then one-hot encoding (unknowns ignored)"


class PreprocessingError(RuntimeError):
    """No column had a strategy, so there is no recipe to build."""


@dataclass(slots=True)
class PreprocessingResult:
    """The unfitted transformer plus the JSON description of what is in it."""

    transformer: ColumnTransformer
    spec: PreprocessingSpec


def build_preprocessor(frame: pd.DataFrame, *, target: str) -> PreprocessingResult:
    """Build an unfitted ``ColumnTransformer`` for ``frame``'s feature columns.

    Reads dtypes and cardinality only. The returned transformer has never seen a
    value and must be handed straight to cross-validation, which clones and fits
    it per fold.
    """
    features = [c for c in frame.columns if c != target]
    numeric: list[str] = []
    categorical: list[str] = []
    unhandled: list[DroppedColumn] = []

    for name in features:
        series = frame[name]
        if pd.api.types.is_bool_dtype(series) or pd.api.types.is_numeric_dtype(series):
            # Booleans go down the numeric path: scikit-learn reads them as 0/1,
            # and mode-imputing then one-hot encoding a two-value flag would add
            # a redundant column for no gain.
            numeric.append(name)
        elif pd.api.types.is_datetime64_any_dtype(series):
            unhandled.append(
                DroppedColumn(name=name, reason="datetime -- feature extraction is Section 7")
            )
        elif int(series.nunique(dropna=True)) > _MAX_ONEHOT_CARDINALITY:
            unhandled.append(
                DroppedColumn(
                    name=name,
                    reason=(
                        f"{series.nunique(dropna=True)} distinct values -- too many to "
                        "one-hot encode; richer encodings are Section 7"
                    ),
                )
            )
        else:
            categorical.append(name)

    if not numeric and not categorical:
        raise PreprocessingError(
            "No column could be preprocessed: every feature is a datetime or "
            "high-cardinality text column, which Section 5 does not yet handle."
        )

    transformers = []
    if numeric:
        transformers.append(("numeric", _numeric_recipe(), numeric))
    if categorical:
        transformers.append(("categorical", _categorical_recipe(), categorical))

    transformer = ColumnTransformer(
        transformers=transformers,
        # Anything without an explicit strategy is dropped rather than passed
        # through raw -- a stray text column reaching the model as-is would fail
        # the fit, and silently forwarding untransformed data is how "it worked
        # on my dataset" bugs get in.
        remainder="drop",
        verbose_feature_names_out=False,
    )

    spec = PreprocessingSpec(
        numeric_columns=numeric,
        categorical_columns=categorical,
        unhandled_columns=unhandled,
        numeric_strategy=_NUMERIC_STRATEGY if numeric else "",
        categorical_strategy=_CATEGORICAL_STRATEGY if categorical else "",
        strategy_source="hardcoded",
    )
    logger.info(
        "Preprocessing recipe (unfitted): %d numeric, %d categorical, %d unhandled",
        len(numeric),
        len(categorical),
        len(unhandled),
    )
    return PreprocessingResult(transformer=transformer, spec=spec)


def _numeric_recipe() -> SklearnPipeline:
    """Median-impute, then scale.

    Median rather than mean because it is unmoved by the outliers real datasets
    are full of. Scaling is here for the models that need it (linear, distance-
    and gradient-based); it is harmless for the tree ensembles that do not, and
    keeping one recipe means the Section 7 model roster cannot acquire a
    model whose preprocessing was silently missing.
    """
    return SklearnPipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )


def _categorical_recipe() -> SklearnPipeline:
    """Mode-impute, then one-hot encode.

    ``handle_unknown="ignore"`` is not a nicety, it is what makes per-fold
    fitting survive: a category that happens to appear only in the held-out fold
    is genuinely unknown to a correctly-fitted encoder, and the alternative is a
    crash on every dataset with a rare category. Ignoring it encodes the row as
    all-zeros for that feature, which is the honest representation.
    """
    return SklearnPipeline(
        steps=[
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
