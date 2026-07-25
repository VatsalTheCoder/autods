"""Tests for the preprocessing recipe builder.

That the returned transformer is *unfitted* is tested in ``test_leakage.py``,
alongside the other leakage proofs, since that is what it is. This file covers the
routing decisions: which column gets which strategy, and what happens to the ones
Section 5 has no strategy for.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

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


class TestColumnRouting:
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

    def test_datetime_columns_are_named_as_unhandled(self, frame):
        frame["signed_up"] = pd.to_datetime(
            ["2024-01-01", "2024-02-01", "2024-03-01", "2024-04-01"]
        )
        spec = build_preprocessor(frame, target="churn").spec
        assert [c.name for c in spec.unhandled_columns] == ["signed_up"]
        assert "Section 7" in spec.unhandled_columns[0].reason

    def test_high_cardinality_text_is_named_as_unhandled(self):
        frame = pd.DataFrame(
            {
                "note": [f"free text {i}" for i in range(60)],
                "age": np.arange(60),
                "churn": ["yes", "no"] * 30,
            }
        )
        spec = build_preprocessor(frame, target="churn").spec
        assert [c.name for c in spec.unhandled_columns] == ["note"]
        assert "distinct values" in spec.unhandled_columns[0].reason

    def test_unhandled_columns_are_named_not_silently_ignored(self, frame):
        """A gap in the recipe should be visible in the artifact."""
        frame["signed_up"] = pd.to_datetime(["2024-01-01"] * 4)
        spec = build_preprocessor(frame, target="churn").spec
        assert spec.unhandled_columns[0].reason


class TestTransformerShape:
    def test_unhandled_columns_are_dropped_not_passed_through(self, frame):
        """``remainder="drop"``: a text column must not reach the model raw."""
        frame["note"] = [f"free text {i}" for i in range(4)]
        transformer = build_preprocessor(frame, target="churn").transformer
        assert transformer.remainder == "drop"

    def test_only_the_needed_branches_are_built(self):
        """An all-numeric frame gets no categorical branch at all."""
        frame = pd.DataFrame({"age": [1, 2, 3, 4], "churn": ["y", "n", "y", "n"]})
        result = build_preprocessor(frame, target="churn")
        assert [name for name, _, _ in result.transformer.transformers] == ["numeric"]
        assert result.spec.categorical_strategy == ""

    def test_both_branches_describe_their_strategy(self, frame):
        spec = build_preprocessor(frame, target="churn").spec
        assert "median" in spec.numeric_strategy
        assert "one-hot" in spec.categorical_strategy

    def test_section_5_does_not_claim_the_llm_chose_the_strategy(self, frame):
        """The LLM picks strategies in Section 7; the artifact must not overclaim."""
        assert build_preprocessor(frame, target="churn").spec.strategy_source == "hardcoded"


class TestNoUsableColumns:
    def test_a_frame_of_only_unhandled_columns_is_an_error(self):
        frame = pd.DataFrame(
            {
                "note": [f"free text {i}" for i in range(60)],
                "churn": ["yes", "no"] * 30,
            }
        )
        with pytest.raises(PreprocessingError, match="No column could be preprocessed"):
            build_preprocessor(frame, target="churn")
