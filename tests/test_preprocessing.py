"""Tests for the preprocessing recipe builder.

That the returned transformer is *unfitted* is tested in ``test_leakage.py``,
alongside the other leakage proofs, since that is what it is. This file covers
what gets built: which column ends up in which branch, and that the strategy the
agent chose is the one the recipe reflects.

Section 5's version of this file asserted that datetime and high-cardinality text
columns were *named as unhandled* -- a stated gap. Section 7 closes it, so those
assertions are now the opposite: both dtypes get a real branch, and the only
columns left unhandled are the ones something deliberately dropped.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.ml.contracts import ColumnStrategy, FeatureStrategy
from app.ml.preprocessing import PreprocessingError, build_preprocessor


@pytest.fixture
def frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "age": [34, 28, 45, 51],
            "income": [52000.0, 41000.0, 68000.0, 71000.0],
            "city": ["London", "Leeds", "Bristol", "London"],
            "subscribed": [True, False, True, False],
            "churn": ["yes", "no", "no", "yes"],
        }
    )


def _branch_names(result) -> list[str]:
    return [name for name, _, _ in result.transformer.transformers]


class TestColumnRouting:
    """The dtype defaults, which are what runs when no LLM is configured."""

    def test_numeric_columns_go_to_the_numeric_recipe(self, frame):
        spec = build_preprocessor(frame, target="churn").spec
        assert set(spec.numeric_columns) == {"age", "income", "subscribed"}

    def test_booleans_are_treated_as_numeric(self, frame):
        """0/1 flags need no encoding; one-hot would add a redundant column."""
        spec = build_preprocessor(frame, target="churn").spec
        assert "subscribed" in spec.numeric_columns
        assert "subscribed" not in spec.categorical_columns

    def test_categorical_columns_go_to_the_categorical_recipe(self, frame):
        spec = build_preprocessor(frame, target="churn").spec
        assert spec.categorical_columns == ["city"]

    def test_the_target_is_never_a_feature(self, frame):
        spec = build_preprocessor(frame, target="churn").spec
        assert "churn" not in spec.numeric_columns + spec.categorical_columns


class TestTheGapsSectionSevenCloses:
    """The two dtypes Section 5 could only name and drop."""

    def test_datetime_columns_now_get_a_branch(self, frame):
        frame["signed_up"] = pd.to_datetime(
            ["2024-01-01", "2024-02-01", "2024-03-01", "2024-04-01"]
        )
        result = build_preprocessor(frame, target="churn")
        assert result.spec.datetime_columns == ["signed_up"]
        assert not result.spec.unhandled_columns
        assert any(name.startswith("datetime") for name in _branch_names(result))

    def test_high_cardinality_text_now_gets_a_branch(self):
        frame = pd.DataFrame(
            {
                "note": [f"free text {i % 52}" for i in range(60)],
                "age": np.arange(60),
                "churn": ["yes", "no"] * 30,
            }
        )
        result = build_preprocessor(frame, target="churn")
        assert result.spec.text_columns == ["note"]
        assert not result.spec.unhandled_columns

    def test_a_datetime_column_becomes_several_calendar_features(self, frame):
        """One timestamp in, five numbers out -- and they survive a fit."""
        frame["signed_up"] = pd.to_datetime(
            ["2024-01-01", "2024-02-01", "2024-03-01", "2024-04-01"]
        )
        transformer = build_preprocessor(frame, target="churn").transformer
        out = transformer.fit_transform(frame.drop(columns=["churn"]))
        names = list(transformer.get_feature_names_out())
        assert "signed_up_month" in names
        assert np.isfinite(out).all()

    def test_text_encoding_adds_one_column_not_hundreds(self):
        frame = pd.DataFrame(
            {"note": [f"n{i % 52}" for i in range(60)], "churn": ["yes", "no"] * 30}
        )
        transformer = build_preprocessor(frame, target="churn").transformer
        out = transformer.fit_transform(frame.drop(columns=["churn"]))
        assert out.shape[1] == 1


class TestStrategyDrivenRouting:
    """When the agent has chosen, the recipe follows it rather than the dtypes."""

    def test_an_ordinal_column_gets_its_own_branch(self):
        frame = pd.DataFrame(
            {
                "size": ["small", "medium", "large", "small"],
                "churn": ["yes", "no", "no", "yes"],
            }
        )
        strategy = FeatureStrategy(
            columns=[
                ColumnStrategy(
                    column="size",
                    role="ordinal",
                    impute="most_frequent",
                    encode="ordinal",
                    ordinal_order=["small", "medium", "large"],
                )
            ],
            source="llm",
        )
        result = build_preprocessor(frame, target="churn", strategy=strategy)
        assert result.spec.ordinal_columns == ["size"]

    def test_the_ordering_is_the_one_that_was_chosen(self):
        """small < medium < large, not the alphabetical order sklearn would infer."""
        frame = pd.DataFrame(
            {
                "size": ["small", "medium", "large", "small"],
                "churn": ["yes", "no", "no", "yes"],
            }
        )
        strategy = FeatureStrategy(
            columns=[
                ColumnStrategy(
                    column="size",
                    role="ordinal",
                    impute="most_frequent",
                    encode="ordinal",
                    ordinal_order=["small", "medium", "large"],
                )
            ],
            source="llm",
        )
        transformer = build_preprocessor(frame, target="churn", strategy=strategy).transformer
        out = transformer.fit_transform(frame.drop(columns=["churn"]))
        assert list(out.ravel()) == [0.0, 1.0, 2.0, 0.0]

    def test_a_dropped_column_is_named_and_left_out(self, frame):
        strategy = FeatureStrategy(
            columns=[
                ColumnStrategy(column="age", role="numeric", impute="median", scale="standard"),
                ColumnStrategy(column="city", role="drop", rationale="A free-text note."),
            ],
            source="llm",
        )
        spec = build_preprocessor(frame, target="churn", strategy=strategy).spec
        assert [c.name for c in spec.unhandled_columns] == ["city"]
        assert "city" not in spec.categorical_columns

    def test_columns_sharing_a_recipe_share_a_branch(self, frame):
        """Five identically-treated numbers are one branch, not five."""
        result = build_preprocessor(frame, target="churn")
        numeric = [name for name in _branch_names(result) if name.startswith("numeric")]
        assert numeric == ["numeric_1"]

    def test_a_column_cleaning_removed_does_not_break_the_recipe(self, frame):
        """The strategy is chosen before cleaning may drop a column."""
        strategy = FeatureStrategy(
            columns=[
                ColumnStrategy(column="age", role="numeric", impute="median", scale="standard"),
                ColumnStrategy(column="gone", role="numeric", impute="median"),
            ],
            source="llm",
        )
        spec = build_preprocessor(frame, target="churn", strategy=strategy).spec
        assert "gone" not in spec.numeric_columns

    def test_the_artifact_says_the_llm_chose(self, frame):
        strategy = FeatureStrategy(
            columns=[ColumnStrategy(column="age", role="numeric", impute="median")],
            source="llm",
        )
        spec = build_preprocessor(frame, target="churn", strategy=strategy).spec
        assert spec.strategy_source == "llm"

    def test_the_per_column_detail_is_recorded(self, frame):
        spec = build_preprocessor(frame, target="churn").spec
        assert {c.column for c in spec.column_strategies} == set(frame.columns) - {"churn"}


class TestTransformerShape:
    def test_unhandled_columns_are_dropped_not_passed_through(self, frame):
        """``remainder="drop"``: nothing reaches the model without a strategy."""
        transformer = build_preprocessor(frame, target="churn").transformer
        assert transformer.remainder == "drop"

    def test_only_the_needed_branches_are_built(self):
        """An all-numeric frame gets no categorical branch at all."""
        frame = pd.DataFrame({"age": [1, 2, 3, 4], "churn": ["y", "n", "y", "n"]})
        result = build_preprocessor(frame, target="churn")
        assert _branch_names(result) == ["numeric_1"]
        assert result.spec.categorical_strategy == ""

    def test_both_branches_describe_their_strategy(self, frame):
        spec = build_preprocessor(frame, target="churn").spec
        assert "median" in spec.numeric_strategy
        assert "one-hot" in spec.categorical_strategy

    def test_the_default_recipe_does_not_claim_the_llm_chose_it(self, frame):
        """With no strategy the artifact must say hardcoded, not overclaim."""
        assert build_preprocessor(frame, target="churn").spec.strategy_source == "hardcoded"


class TestFeatureSelection:
    """Selection is a pipeline step, so it is fitted per fold like everything else."""

    def test_no_selector_unless_asked_for(self, frame):
        result = build_preprocessor(frame, target="churn")
        assert not hasattr(result.transformer, "named_steps")
        assert result.spec.feature_selection == ""

    def test_the_selector_is_a_step_inside_the_recipe(self, frame):
        result = build_preprocessor(frame, target="churn", select_k=2)
        assert "select" in result.transformer.named_steps
        assert result.spec.feature_selection

    def test_selection_reduces_the_columns_the_model_sees(self, frame):
        result = build_preprocessor(frame, target="churn", select_k=2)
        out = result.transformer.fit_transform(frame.drop(columns=["churn"]), frame["churn"])
        assert out.shape[1] == 2


class TestNoUsableColumns:
    def test_a_frame_with_every_column_dropped_is_an_error(self, frame):
        strategy = FeatureStrategy(
            columns=[ColumnStrategy(column=c, role="drop") for c in frame.columns if c != "churn"],
            source="llm",
        )
        with pytest.raises(PreprocessingError, match="No column could be preprocessed"):
            build_preprocessor(frame, target="churn", strategy=strategy)
