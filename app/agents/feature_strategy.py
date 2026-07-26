"""Feature strategy -- the LLM chooses, code checks, code builds (spec 7.6).

**This is the file to read if you want to see the project's design idea in one
place.** Every other agent is a variation on it; this is the clearest instance.

The LLM is asked, column by column, "how should this be prepared?" It answers
with a small table -- role, imputation, encoding, scaling -- and nothing else. It
never sees a pipeline, never writes pandas, never touches a value. Ordinary code
then reads that table and builds the real scikit-learn recipe
(``ml/preprocessing.py``).

Why that split is the whole point: a JSON object naming a column and one of five
imputation strategies is *checkable*. You can confirm the column exists, that the
strategy is one the code implements, and that it makes sense for the column's
dtype -- before anything runs. Generated pandas code is checkable by executing
it, which is the same as trusting it. So the model's output here is constrained
to a closed vocabulary (``contracts.ColumnRole`` and friends) and then put
through three gates:

1. **Invented columns are rejected.** The spec asks for this by name. A strategy
   for a column that is not in the dataset is dropped and recorded in
   ``rejected_columns``, so a reader can see the rejection happened.
2. **Incoherent choices are overridden.** "Median-impute this text column" is a
   valid-looking JSON object and an impossible instruction. Code compares each
   choice against the column's actual dtype and cardinality, applies the nearest
   workable one, and records the substitution in ``overrides`` -- the same
   pattern as the planner's clustering method being overruled when the data is
   categorical (``ml/clustering.py``).
3. **Silence falls back to the defaults.** A column the LLM skipped gets the
   dtype-based strategy Section 5 hardcoded. The pipeline is never left with a
   column it has no plan for.

Like the planner, the LLM pass is **best-effort**: every column has a
deterministic default, so a missing API key or a rate-limited free tier produces
exactly the Section 5 recipe rather than a failed job. ``source`` says which
happened, and it is stamped by this module rather than by the model -- see
``make_strategy``.

One thing this module is careful *not* to do: compute a statistic. It reads
column names, dtypes, and the set of distinct values (to size an encoder and to
check an ordinal ordering). It never computes a mean, a median, a frequency or
anything else a model would learn -- those all belong to transformers fitted
inside a fold. See ``ml/preprocessing.py``'s docstring for where that line sits.
"""

from __future__ import annotations

import logging

import pandas as pd
from pydantic import BaseModel, Field

from app.agents.schema_models import SchemaReport
from app.core.llm.base import LLMClient, LLMError, ModelTier, UsageCallback, system, user
from app.core.llm.structured import structured_complete
from app.ml.contracts import (
    ColumnRole,
    ColumnStrategy,
    EncodeStrategy,
    FeatureStrategy,
    ImputeStrategy,
    ScaleStrategy,
    StrategyOverride,
)

logger = logging.getLogger(__name__)

AGENT_NAME = "feature_strategy"

# Above this many distinct values, one-hot encoding stops being reasonable: it
# would add hundreds of near-empty columns and invite overfitting. Such columns
# become ``text`` and get frequency encoding, which costs one column instead.
MAX_ONEHOT_CARDINALITY = 50

# How many sample values the prompt shows per column. Unlike the planner -- which
# deliberately sends no values at all -- this agent needs them: deciding that a
# column is ordinal, and in what order, is impossible without seeing that it
# contains "low", "medium" and "high". Three is enough to show the shape of a
# column without turning the prompt into a data dump (spec 10's token budget).
_SAMPLE_VALUES = 3

_SYSTEM = (
    "You decide how each column of a tabular dataset should be prepared before a "
    "model is trained on it. You are given the column list with types and example "
    "values. Reply with one entry per column and nothing else.\n"
    "\n"
    "For each column choose:\n"
    "- role: 'numeric' for quantities; 'categorical' for unordered labels; "
    "'ordinal' for labels with a genuine order (small/medium/large, "
    "poor/fair/good); 'datetime' for dates and timestamps; 'text' for free text "
    "or identifiers with very many distinct values; 'drop' for a column that "
    "cannot help a model, such as a row id.\n"
    "- impute: how to fill blanks -- 'median' or 'mean' for numbers, "
    "'most_frequent' or 'constant' for labels, 'none' if it should not be filled.\n"
    "- encode: 'onehot' for a few unordered labels, 'ordinal' for ordered ones, "
    "'frequency' when there are too many labels to one-hot, 'none' for numbers.\n"
    "- scale: 'standard' for most numbers, 'minmax' for bounded quantities like "
    "percentages, 'none' when scaling is meaningless.\n"
    "- ordinal_order: only for ordinal columns, the categories from lowest to "
    "highest, spelled exactly as they appear in the data. Leave it empty "
    "otherwise.\n"
    "\n"
    "Only use column names given to you; do not invent any. Do not include the "
    "target column. Keep each rationale to one short sentence."
)


class _StrategyReply(BaseModel):
    """What the model is asked to return.

    A thin wrapper rather than a bare list because a top-level JSON array is the
    shape LLMs most often get subtly wrong (wrapping it in prose, or emitting an
    object keyed by column name). Naming the field gives the schema hint
    something concrete to describe.
    """

    columns: list[ColumnStrategy] = Field(default_factory=list)
    rationale: str = ""


def make_strategy(
    frame: pd.DataFrame,
    *,
    target: str,
    report: SchemaReport | None = None,
    client: LLMClient | None = None,
    on_usage: UsageCallback | None = None,
) -> FeatureStrategy:
    """Decide a preparation strategy for every feature column in ``frame``.

    Never raises. With no client -- or a client that fails -- every column gets
    its dtype-based default and ``source`` stays ``"hardcoded"``.
    """
    features = [c for c in frame.columns if c != target]

    if client is None:
        logger.info("Feature strategy: no LLM configured, using dtype defaults")
        return _all_defaults(
            frame, features, "No LLM available; used default per-column strategies."
        )

    try:
        result = structured_complete(
            client,
            [system(_SYSTEM), user(_columns_prompt(frame, features, report))],
            _StrategyReply,
            tier=ModelTier.SMALL,
            on_usage=on_usage,
        )
    except LLMError as exc:
        logger.warning("Feature strategy LLM call failed, using defaults: %s", exc)
        return _all_defaults(frame, features, f"Feature strategy fell back to defaults: {exc}")
    except Exception:  # pragma: no cover - defensive; strategy must never fail a job
        logger.exception("Unexpected error choosing feature strategies; using defaults")
        return _all_defaults(
            frame, features, "Feature strategy fell back to defaults after an unexpected error."
        )

    strategy = reconcile(result.data.columns, frame, target=target)
    # ``source`` is stamped here, where whether the model actually answered is
    # known. Letting the reply set it would mean a hallucinated field could
    # record a defaulted strategy as an LLM decision.
    strategy.source = "llm"
    strategy.rationale = result.data.rationale

    logger.info(
        "Feature strategy: %d columns from the LLM, %d rejected, %d defaulted, %d overridden",
        len(strategy.columns),
        len(strategy.rejected_columns),
        len(strategy.defaulted_columns),
        len(strategy.overrides),
    )
    return strategy


def reconcile(
    proposed: list[ColumnStrategy],
    frame: pd.DataFrame,
    *,
    target: str,
) -> FeatureStrategy:
    """Put the LLM's answer through the three gates, and report what each did.

    Pure and frame-only: no LLM, no database. Split out from ``make_strategy`` so
    the validation can be tested directly against a handwritten "reply", which is
    how the rejection and override behaviour is pinned down without a model in
    the loop.
    """
    features = [c for c in frame.columns if c != target]
    known = set(features)

    rejected: list[str] = []
    overrides: list[StrategyOverride] = []
    chosen: dict[str, ColumnStrategy] = {}

    for item in proposed:
        # Gate 1. The target is not the LLM's to prepare, and a name that is not
        # in the dataset is a hallucination -- both are refused the same way.
        if item.column not in known:
            rejected.append(item.column)
            continue
        # A duplicate entry for a column is the model contradicting itself; the
        # first answer stands, so reconciliation is deterministic.
        if item.column in chosen:
            continue
        # Gate 2.
        chosen[item.column] = _coerce(item, frame[item.column], overrides)

    if rejected:
        logger.warning("Feature strategy rejected invented columns: %s", ", ".join(rejected))

    # Gate 3. Preserve the dataset's column order rather than the LLM's, so the
    # artifact reads in the same order as the data and two runs are comparable.
    defaulted = [name for name in features if name not in chosen]
    columns = [chosen.get(name) or default_strategy(name, frame[name]) for name in features]

    return FeatureStrategy(
        columns=columns,
        rejected_columns=rejected,
        defaulted_columns=defaulted,
        overrides=overrides,
    )


def default_strategy(name: str, series: pd.Series) -> ColumnStrategy:
    """The dtype-based strategy for one column -- Section 5's behaviour, per column.

    This is the floor the whole agent stands on: it is what runs with no LLM, what
    fills the gaps in a partial reply, and what an override falls back towards.
    Numbers get median imputation and standard scaling; labels get mode imputation
    and one-hot encoding; the two dtypes Section 5 could not express get the
    handling this section adds.
    """
    role = observed_role(series)
    if role == "numeric":
        return ColumnStrategy(
            column=name, role="numeric", impute="median", encode="none", scale="standard"
        )
    if role == "datetime":
        return ColumnStrategy(
            column=name, role="datetime", impute="median", encode="none", scale="standard"
        )
    if role == "text":
        return ColumnStrategy(
            column=name, role="text", impute="most_frequent", encode="frequency", scale="none"
        )
    return ColumnStrategy(
        column=name, role="categorical", impute="most_frequent", encode="onehot", scale="none"
    )


def observed_role(series: pd.Series) -> ColumnRole:
    """What the column *is*, judged from its dtype and how many values it holds.

    Never returns ``ordinal`` or ``drop``: both are judgements about meaning that
    dtypes cannot supply. Ordering is knowledge about the world (that "low"
    precedes "high"), and dropping is a decision about usefulness -- which is
    precisely the pair of things the LLM is here to contribute.
    """
    if pd.api.types.is_bool_dtype(series) or pd.api.types.is_numeric_dtype(series):
        # Booleans go down the numeric path: scikit-learn reads them as 0/1, and
        # one-hot encoding a two-value flag adds a redundant column for no gain.
        return "numeric"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    if int(series.nunique(dropna=True)) > MAX_ONEHOT_CARDINALITY:
        return "text"
    return "categorical"


# ---- Gate 2: making a choice coherent with the column it is about ------------


def _coerce(
    item: ColumnStrategy,
    series: pd.Series,
    overrides: list[StrategyOverride],
) -> ColumnStrategy:
    """Force one column's strategy into something buildable, recording each change."""
    role = _coerce_role(item, series, overrides)
    order = _coerce_ordinal_order(item, series, role, overrides)
    if order is None and role == "ordinal":
        # The stated ordering did not match the data, so the ordering claim is
        # unusable -- treat the column as the unordered labels it demonstrably is.
        role = "categorical"
        order = []

    impute = _coerce_choice(
        item, "impute", item.impute, _ALLOWED_IMPUTE[role], _DEFAULT_IMPUTE[role], overrides
    )
    encode = _coerce_choice(
        item, "encode", item.encode, _ALLOWED_ENCODE[role], _DEFAULT_ENCODE[role], overrides
    )
    scale = _coerce_choice(
        item, "scale", item.scale, _ALLOWED_SCALE[role], _DEFAULT_SCALE[role], overrides
    )

    return ColumnStrategy(
        column=item.column,
        role=role,
        impute=impute,
        encode=encode,
        scale=scale,
        ordinal_order=order or [],
        rationale=item.rationale,
    )


def _coerce_role(
    item: ColumnStrategy,
    series: pd.Series,
    overrides: list[StrategyOverride],
) -> ColumnRole:
    """Allow the roles the column can actually support; substitute the rest.

    The permissive cases are the ones where the LLM knows something the dtype does
    not. A low-cardinality integer column really can be categorical -- a postcode
    or a product code stored as a number is a label, not a quantity, and reading
    it as a quantity is a modelling error the dtype alone cannot catch. So that
    reclassification is accepted. Claiming a 5,000-value free-text column can be
    one-hot encoded is not knowledge, it is a mistake, and is overruled.
    """
    requested = item.role
    observed = observed_role(series)

    # Dropping is always allowed: it is a judgement about usefulness, and the
    # cost of wrongly honouring it is a weaker model, not a broken one.
    if requested == "drop":
        return "drop"

    if observed == "numeric":
        if requested in ("numeric", "datetime"):
            return "numeric"
        # A number used as a label, but only if there are few enough of them.
        if requested in ("categorical", "ordinal", "text"):
            if int(series.nunique(dropna=True)) <= MAX_ONEHOT_CARDINALITY:
                return requested
            return _override(
                overrides,
                item.column,
                "role",
                requested,
                "numeric",
                f"{series.nunique(dropna=True)} distinct numeric values -- too many "
                "to treat as labels",
            )
        return "numeric"

    if observed == "datetime":
        if requested == "datetime":
            return "datetime"
        return _override(
            overrides,
            item.column,
            "role",
            requested,
            "datetime",
            "the column holds timestamps, which are turned into calendar features",
        )

    if observed == "text":
        # High cardinality. Ordered or one-hot encoded labels are both off the
        # table; frequency encoding is what "text" means downstream.
        if requested in ("text", "drop"):
            return requested
        return _override(
            overrides,
            item.column,
            "role",
            requested,
            "text",
            f"{series.nunique(dropna=True)} distinct values -- too many to encode "
            "as individual categories",
        )

    # Observed categorical: low-cardinality labels. Everything except a numeric
    # reading is plausible, and a numeric reading of labels cannot be built.
    if requested in ("categorical", "ordinal", "text"):
        return requested
    return _override(
        overrides,
        item.column,
        "role",
        requested,
        "categorical",
        "the column holds non-numeric labels",
    )


def _coerce_ordinal_order(
    item: ColumnStrategy,
    series: pd.Series,
    role: ColumnRole,
    overrides: list[StrategyOverride],
) -> list[str] | None:
    """Check the stated ordering covers the data. ``None`` means it does not.

    An ordinal encoding is only as good as its ordering, and an ordering that
    omits a category present in the data would encode those rows as "unknown" --
    quietly discarding a level while the report claimed the column was ordered.
    Requiring full coverage means the ordinal path is used when it is genuinely
    right and the column falls back to one-hot when it is not.

    Reading the column's distinct values here is a *vocabulary* read, not a
    statistic: it asks which labels exist, not how often or with what outcome.
    Nothing learned from it varies with the fold, which is why it can happen
    outside one.
    """
    if role != "ordinal":
        return []

    present = {str(v) for v in series.dropna().unique()}
    stated = [str(v) for v in item.ordinal_order]

    if not stated:
        _override(
            overrides,
            item.column,
            "role",
            "ordinal",
            "categorical",
            "no ordering was given, so the categories cannot be ranked",
        )
        return None

    missing = present - set(stated)
    if missing:
        _override(
            overrides,
            item.column,
            "role",
            "ordinal",
            "categorical",
            f"the stated ordering leaves out {len(missing)} value(s) present in the "
            f"data ({', '.join(sorted(missing)[:3])})",
        )
        return None

    # Categories the LLM listed but the data does not contain are harmless -- the
    # encoder simply never sees them -- so they are kept rather than pruned,
    # which also keeps the recorded ordering the one the model actually stated.
    return stated


def _coerce_choice[T: str](
    item: ColumnStrategy,
    field: str,
    requested: T,
    allowed: frozenset[str],
    fallback: T,
    overrides: list[StrategyOverride],
) -> T:
    """One field's worth of gate 2: keep it if buildable, else record and replace."""
    if requested in allowed:
        return requested
    return _override(
        overrides,
        item.column,
        field,
        requested,
        fallback,
        f"'{requested}' cannot be applied to a {item.role} column",
    )


def _override[T: str](
    overrides: list[StrategyOverride],
    column: str,
    field: str,
    requested: str,
    applied: T,
    reason: str,
) -> T:
    """Record a substitution and return what will actually be used."""
    overrides.append(
        StrategyOverride(
            column=column, field=field, requested=requested, applied=applied, reason=reason
        )
    )
    logger.info("Feature strategy override on %s.%s: %s -> %s", column, field, requested, applied)
    return applied


# What each role can be built with. Keyed by the *final* role, so these tables and
# ``preprocessing.py``'s recipe builders are two views of one contract: anything
# allowed here has an implementation there.
_ALLOWED_IMPUTE: dict[ColumnRole, frozenset[str]] = {
    "numeric": frozenset({"median", "mean", "constant", "none"}),
    # Calendar features are numbers once extracted, so they impute like numbers.
    "datetime": frozenset({"median", "mean", "constant", "none"}),
    "categorical": frozenset({"most_frequent", "constant", "none"}),
    "ordinal": frozenset({"most_frequent", "constant", "none"}),
    "text": frozenset({"most_frequent", "constant", "none"}),
    "drop": frozenset({"none"}),
}
_DEFAULT_IMPUTE: dict[ColumnRole, ImputeStrategy] = {
    "numeric": "median",
    "datetime": "median",
    "categorical": "most_frequent",
    "ordinal": "most_frequent",
    "text": "most_frequent",
    "drop": "none",
}

_ALLOWED_ENCODE: dict[ColumnRole, frozenset[str]] = {
    "numeric": frozenset({"none"}),
    "datetime": frozenset({"none"}),
    "categorical": frozenset({"onehot", "frequency"}),
    # The role *is* the encoding here; allowing anything else would make the
    # ordering the LLM supplied pointless.
    "ordinal": frozenset({"ordinal"}),
    "text": frozenset({"frequency"}),
    "drop": frozenset({"none"}),
}
_DEFAULT_ENCODE: dict[ColumnRole, EncodeStrategy] = {
    "numeric": "none",
    "datetime": "none",
    "categorical": "onehot",
    "ordinal": "ordinal",
    "text": "frequency",
    "drop": "none",
}

_ALLOWED_SCALE: dict[ColumnRole, frozenset[str]] = {
    "numeric": frozenset({"standard", "minmax", "none"}),
    "datetime": frozenset({"standard", "minmax", "none"}),
    # One-hot columns are already 0/1; scaling them destroys that reading for no
    # benefit. Ordinal codes and frequency counts are genuine numbers, so scaling
    # them is allowed and useful for the linear models in the roster.
    "categorical": frozenset({"none"}),
    "ordinal": frozenset({"standard", "minmax", "none"}),
    "text": frozenset({"standard", "minmax", "none"}),
    "drop": frozenset({"none"}),
}
_DEFAULT_SCALE: dict[ColumnRole, ScaleStrategy] = {
    "numeric": "standard",
    "datetime": "standard",
    "categorical": "none",
    "ordinal": "none",
    "text": "none",
    "drop": "none",
}


# ---- The prompt -------------------------------------------------------------


def _all_defaults(frame: pd.DataFrame, features: list[str], rationale: str) -> FeatureStrategy:
    """Every column on its dtype default -- the no-LLM path, and Section 5's recipe."""
    return FeatureStrategy(
        columns=[default_strategy(name, frame[name]) for name in features],
        defaulted_columns=list(features),
        source="hardcoded",
        rationale=rationale,
    )


def _columns_prompt(
    frame: pd.DataFrame,
    features: list[str],
    report: SchemaReport | None,
) -> str:
    """The column list, with the few example values the ordinal decision needs.

    The schema report is used for the semantic type and meaning when it is
    available -- the LLM already reasoned about those at the checkpoint, and
    repeating that work here would pay twice for the same conclusion.
    """
    lines = [f"Dataset: {frame.shape[0]} rows.", "", "Columns:"]
    for name in features:
        series = frame[name]
        profile = report.column(name) if report is not None else None
        semantic = profile.semantic_type if profile else observed_role(series)
        samples = ", ".join(repr(v) for v in series.dropna().unique()[:_SAMPLE_VALUES])
        line = (
            f"- {name} (type={semantic}, distinct={int(series.nunique(dropna=True))}, "
            f"missing={float(series.isna().mean()):.0%}) e.g. {samples or 'no values'}"
        )
        if profile is not None and profile.meaning:
            line += f" -- {profile.meaning}"
        lines.append(line)
    return "\n".join(lines)
