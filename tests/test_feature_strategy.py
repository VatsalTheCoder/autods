"""Tests for the feature strategy agent -- mostly tests of what it *refuses*.

The agent's value is not that it relays the LLM's answer; it is that a wrong
answer cannot reach the pipeline. So the bulk of this file hands ``reconcile`` a
handwritten "reply" containing a specific mistake and asserts on what came out
the other side. That needs no model at all, which is why the nasty cases can be
enumerated cheaply.

The three gates, in the order the module applies them:

* invented column names are rejected and named (spec 7.6's stated requirement),
* choices that cannot be built are overridden, with the substitution recorded,
* columns the model ignored fall back to the dtype default.

Plus the property that outranks all of them: a strategy is always produced. No
API key, a rate limit, or ten malformed replies must give the Section 5 recipe
rather than a failed job.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.agents.feature_strategy import (
    default_strategy,
    is_near_unique,
    make_strategy,
    observed_role,
    reconcile,
)
from app.core.llm.base import LLMConfigError, RateLimitError
from app.core.llm.fake import FakeLLM
from app.ml.contracts import ColumnStrategy

VALID_REPLY = (
    '{"columns": [{"column": "age", "role": "numeric", "impute": "median", '
    '"encode": "none", "scale": "standard", "rationale": "A quantity."}], '
    '"rationale": "Standard treatment."}'
)


@pytest.fixture
def frame() -> pd.DataFrame:
    """A frame with one column of each kind the agent has to route."""
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "age": rng.integers(18, 80, 60),
            "city": rng.choice(["London", "Leeds", "Bristol"], 60),
            "size": rng.choice(["small", "medium", "large"], 60),
            "signed_up": pd.to_datetime(rng.integers(1_600_000_000, 1_700_000_000, 60), unit="s"),
            # High cardinality but with repeats, so frequency encoding has
            # something to encode. A column with one value per row is a
            # different case -- see ``TestNearUniqueTextIsDropped``.
            "user_ref": [f"ref-{i % 52:04d}" for i in range(60)],
            "listing_name": [f"a nice place {i}" for i in range(60)],
            "churn": ["yes", "no"] * 30,
        }
    )


def _strategy(frame, *items: ColumnStrategy):
    return reconcile(list(items), frame, target="churn")


class TestObservedRole:
    """The dtype-only reading, which is the floor everything else falls back to."""

    def test_numbers_and_booleans_are_numeric(self, frame):
        assert observed_role(frame["age"]) == "numeric"
        assert observed_role(pd.Series([True, False, True])) == "numeric"

    def test_timestamps_are_datetime(self, frame):
        assert observed_role(frame["signed_up"]) == "datetime"

    def test_few_labels_are_categorical(self, frame):
        assert observed_role(frame["city"]) == "categorical"

    def test_many_labels_are_text(self, frame):
        """60 distinct references is past the one-hot threshold."""
        assert observed_role(frame["user_ref"]) == "text"

    def test_it_never_guesses_meaning(self, frame):
        """Ordinality and uselessness are judgements dtypes cannot make."""
        roles = {observed_role(frame[c]) for c in frame.columns}
        assert "ordinal" not in roles
        assert "drop" not in roles


class TestGateOneInventedColumns:
    """The requirement the spec names explicitly."""

    def test_a_column_that_does_not_exist_is_rejected(self, frame):
        result = _strategy(frame, ColumnStrategy(column="salary", role="numeric"))
        assert result.rejected_columns == ["salary"]
        assert result.for_column("salary") is None

    def test_a_rejected_column_never_reaches_the_recipe(self, frame):
        result = _strategy(frame, ColumnStrategy(column="salary", role="numeric"))
        assert "salary" not in [c.column for c in result.columns]

    def test_the_target_is_not_the_agents_to_prepare(self, frame):
        """The target is a real column, but preparing it is not this agent's job."""
        result = _strategy(frame, ColumnStrategy(column="churn", role="categorical"))
        assert "churn" == result.rejected_columns[0]
        assert "churn" not in [c.column for c in result.columns]

    def test_rejection_does_not_discard_the_good_entries(self, frame):
        result = _strategy(
            frame,
            ColumnStrategy(column="salary", role="numeric"),
            ColumnStrategy(column="age", role="numeric", impute="mean", scale="minmax"),
        )
        age = result.for_column("age")
        assert age is not None and age.impute == "mean" and age.scale == "minmax"

    def test_a_contradictory_duplicate_does_not_win(self, frame):
        """The first answer stands, so reconciliation is deterministic."""
        result = _strategy(
            frame,
            ColumnStrategy(column="age", role="numeric", scale="standard"),
            ColumnStrategy(column="age", role="numeric", scale="minmax"),
        )
        assert result.for_column("age").scale == "standard"


class TestGateTwoIncoherentChoices:
    """Valid-looking JSON that describes an impossible transformation."""

    def test_a_text_column_cannot_be_median_imputed(self, frame):
        result = _strategy(frame, ColumnStrategy(column="city", role="numeric", impute="median"))
        assert result.for_column("city").role == "categorical"
        assert result.for_column("city").impute == "most_frequent"

    def test_the_substitution_is_recorded_not_silent(self, frame):
        result = _strategy(frame, ColumnStrategy(column="city", role="numeric", impute="median"))
        override = next(o for o in result.overrides if o.column == "city" and o.field == "role")
        assert override.requested == "numeric"
        assert override.applied == "categorical"
        assert override.reason

    def test_a_high_cardinality_column_cannot_be_one_hot_encoded(self, frame):
        result = _strategy(
            frame, ColumnStrategy(column="user_ref", role="categorical", encode="onehot")
        )
        assert result.for_column("user_ref").role == "text"
        assert result.for_column("user_ref").encode == "frequency"

    def test_one_hot_columns_are_not_scaled(self, frame):
        """Scaling 0/1 dummies destroys their reading for no benefit."""
        result = _strategy(
            frame, ColumnStrategy(column="city", role="categorical", scale="standard")
        )
        assert result.for_column("city").scale == "none"

    def test_a_timestamp_cannot_be_one_hot_encoded(self, frame):
        result = _strategy(
            frame, ColumnStrategy(column="signed_up", role="categorical", encode="onehot")
        )
        assert result.for_column("signed_up").role == "datetime"

    def test_a_number_may_be_reread_as_a_label(self, frame):
        """The permissive case: a code stored as an integer really is a label."""
        codes = pd.DataFrame({"zone": [1, 2, 3] * 20, "churn": ["yes", "no"] * 30})
        result = reconcile(
            [ColumnStrategy(column="zone", role="categorical", encode="onehot")],
            codes,
            target="churn",
        )
        assert result.for_column("zone").role == "categorical"
        assert not result.overrides

    def test_but_not_when_there_are_too_many_of_them(self, frame):
        """Past the threshold the label reading is a mistake, not knowledge."""
        wide = pd.DataFrame({"reading": np.arange(200.0), "churn": ["yes", "no"] * 100})
        result = reconcile(
            [ColumnStrategy(column="reading", role="categorical")], wide, target="churn"
        )
        assert result.for_column("reading").role == "numeric"
        assert "too many" in result.overrides[0].reason


class TestOrdinalHandling:
    """Spec 7.6 asks for ordinal handling with the choice recorded."""

    def test_a_stated_ordering_is_kept(self, frame):
        result = _strategy(
            frame,
            ColumnStrategy(
                column="size",
                role="ordinal",
                encode="ordinal",
                ordinal_order=["small", "medium", "large"],
            ),
        )
        size = result.for_column("size")
        assert size.role == "ordinal"
        assert size.ordinal_order == ["small", "medium", "large"]

    def test_an_ordering_that_misses_a_value_is_refused(self, frame):
        """Encoding the missing level as 'unknown' would quietly discard rows."""
        result = _strategy(
            frame,
            ColumnStrategy(
                column="size", role="ordinal", encode="ordinal", ordinal_order=["small", "medium"]
            ),
        )
        assert result.for_column("size").role == "categorical"
        assert result.for_column("size").encode == "onehot"

    def test_the_refusal_says_which_value_was_missing(self, frame):
        result = _strategy(
            frame,
            ColumnStrategy(
                column="size", role="ordinal", encode="ordinal", ordinal_order=["small", "medium"]
            ),
        )
        override = next(o for o in result.overrides if o.column == "size")
        assert "large" in override.reason

    def test_ordinal_without_an_ordering_is_not_ordinal(self, frame):
        result = _strategy(frame, ColumnStrategy(column="size", role="ordinal", encode="ordinal"))
        assert result.for_column("size").role == "categorical"

    def test_extra_categories_the_data_lacks_are_harmless(self, frame):
        """The encoder simply never sees them; the stated order is kept as given."""
        result = _strategy(
            frame,
            ColumnStrategy(
                column="size",
                role="ordinal",
                encode="ordinal",
                ordinal_order=["tiny", "small", "medium", "large"],
            ),
        )
        assert result.for_column("size").ordinal_order[0] == "tiny"
        assert result.for_column("size").role == "ordinal"


class TestGateThreeSilence:
    def test_an_unmentioned_column_gets_its_default(self, frame):
        result = _strategy(frame, ColumnStrategy(column="age", role="numeric"))
        assert "city" in result.defaulted_columns
        assert result.for_column("city").encode == "onehot"

    def test_every_feature_column_ends_up_with_a_strategy(self, frame):
        result = _strategy(frame, ColumnStrategy(column="age", role="numeric"))
        assert [c.column for c in result.columns] == [c for c in frame.columns if c != "churn"]

    def test_the_column_order_follows_the_data_not_the_reply(self, frame):
        """Two runs stay comparable regardless of what order the model answered in."""
        result = _strategy(
            frame,
            ColumnStrategy(column="user_ref", role="text"),
            ColumnStrategy(column="age", role="numeric"),
        )
        assert [c.column for c in result.columns].index("age") < [
            c.column for c in result.columns
        ].index("user_ref")


class TestDefaults:
    """The no-LLM recipe, which must stay exactly Section 5's behaviour."""

    def test_numbers_are_median_imputed_and_scaled(self, frame):
        strategy = default_strategy("age", frame["age"])
        assert (strategy.impute, strategy.scale) == ("median", "standard")

    def test_labels_are_mode_imputed_and_one_hot_encoded(self, frame):
        strategy = default_strategy("city", frame["city"])
        assert (strategy.impute, strategy.encode) == ("most_frequent", "onehot")

    def test_the_two_dtypes_section_five_could_not_express_now_have_one(self, frame):
        assert default_strategy("signed_up", frame["signed_up"]).role == "datetime"
        assert default_strategy("user_ref", frame["user_ref"]).encode == "frequency"


class TestWithAWorkingModel:
    def test_the_models_choices_are_obeyed(self, frame):
        strategy = make_strategy(frame, target="churn", client=FakeLLM([VALID_REPLY]))
        assert strategy.for_column("age").scale == "standard"

    def test_it_is_marked_as_coming_from_the_llm(self, frame):
        strategy = make_strategy(frame, target="churn", client=FakeLLM([VALID_REPLY]))
        assert strategy.source == "llm"

    def test_the_prompt_shows_values_because_ordinality_needs_them(self, frame):
        """Unlike the planner, this agent cannot decide without seeing the labels."""
        client = FakeLLM([VALID_REPLY])
        make_strategy(frame, target="churn", client=client)
        assert "small" in client.last_prompt or "medium" in client.last_prompt

    def test_the_prompt_leaves_the_target_out(self, frame):
        client = FakeLLM([VALID_REPLY])
        make_strategy(frame, target="churn", client=client)
        assert "- churn (" not in client.last_prompt

    def test_usage_is_recorded_through_the_callback(self, frame):
        recorded = []
        make_strategy(
            frame, target="churn", client=FakeLLM([VALID_REPLY]), on_usage=recorded.append
        )
        assert len(recorded) == 1
        assert recorded[0].total_tokens > 0


class TestDegradingToDefaults:
    """A strategy is always produced; the pipeline never fails for want of one."""

    def test_no_client_gives_the_hardcoded_recipe(self, frame):
        strategy = make_strategy(frame, target="churn", client=None)
        assert strategy.source == "hardcoded"
        assert len(strategy.columns) == len(frame.columns) - 1

    def test_a_missing_api_key_does_not_fail_the_job(self, frame):
        strategy = make_strategy(frame, target="churn", client=FakeLLM([LLMConfigError("no key")]))
        assert strategy.source == "hardcoded"

    def test_a_rate_limit_does_not_fail_the_job(self, frame):
        strategy = make_strategy(frame, target="churn", client=FakeLLM([RateLimitError("429")]))
        assert strategy.source == "hardcoded"

    def test_persistently_malformed_output_falls_back(self, frame):
        client = FakeLLM(["nonsense"] * 10, default="still nonsense")
        assert make_strategy(frame, target="churn", client=client).source == "hardcoded"

    def test_the_fallback_says_why(self, frame):
        strategy = make_strategy(frame, target="churn", client=FakeLLM([RateLimitError("429")]))
        assert strategy.rationale

    def test_an_empty_reply_still_covers_every_column(self, frame):
        """A model that answers with nothing is the same as a model that is absent."""
        strategy = make_strategy(frame, target="churn", client=FakeLLM(['{"columns": []}']))
        assert len(strategy.columns) == len(frame.columns) - 1
        assert len(strategy.defaulted_columns) == len(frame.columns) - 1


class TestNearUniqueTextIsDropped:
    """A column with one value per row has nothing for frequency encoding to say.

    NYC Airbnb's ``name`` is the case: 47,905 distinct listing titles across
    48,895 rows. Encoded by frequency it becomes 1/n for every training row -- a
    constant no model can split on -- and 0.0 for every held-out row, since none
    of those titles were seen during the fold's fit. It cost a column and an
    entry in every SHAP chart, and carried no signal at all.
    """

    def test_the_dtype_default_drops_it(self, frame):
        strategy = default_strategy("listing_name", frame["listing_name"])
        assert strategy.role == "drop"
        assert strategy.encode == "none"
        assert "nearly one per row" in strategy.rationale

    def test_a_column_with_repeats_is_still_frequency_encoded(self, frame):
        """The rule must not swallow the case frequency encoding exists for."""
        strategy = default_strategy("user_ref", frame["user_ref"])
        assert strategy.role == "text"
        assert strategy.encode == "frequency"

    def test_the_model_asking_for_frequency_encoding_is_overruled(self, frame):
        result = _strategy(frame, ColumnStrategy(column="listing_name", role="text"))

        assert result.for_column("listing_name").role == "drop"
        override = next(
            o for o in result.overrides if o.column == "listing_name" and o.field == "role"
        )
        assert override.requested == "text"
        assert override.applied == "drop"
        assert "nearly one per row" in override.reason

    def test_the_model_asking_to_drop_it_is_not_recorded_as_an_override(self, frame):
        """Agreeing with the code is not a correction."""
        result = _strategy(frame, ColumnStrategy(column="listing_name", role="drop"))

        assert result.for_column("listing_name").role == "drop"
        assert not [o for o in result.overrides if o.column == "listing_name"]

    def test_is_near_unique_reads_the_ratio_not_the_count(self):
        many_repeats = pd.Series([f"v{i % 100}" for i in range(10_000)])
        assert not is_near_unique(many_repeats), "100 distinct in 10,000 rows has plenty of signal"
        all_distinct = pd.Series([f"v{i}" for i in range(200)])
        assert is_near_unique(all_distinct)

    def test_an_empty_column_is_not_near_unique(self):
        assert not is_near_unique(pd.Series([], dtype=object))
