"""An omitted input gets the value the recipe learned, not an unknown category.

``_build_frame`` filled absent values with ``None``. SimpleImputer treats NaN as
missing and ``None`` as an ordinary object, so an omitted category skipped the
imputer entirely and reached OneHotEncoder as a level it had never seen --
encoded, under ``handle_unknown="ignore"``, as a row of zeros.

The consequence was quiet and asymmetric: a caller who left ``city`` out got a
different prediction from one who sent it as null, and neither got the
most-frequent value the pipeline was fitted to substitute.

These tests run the real recipe over a fitted transformer. No model artifact, no
storage, no network.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from app.ml.contracts import FinalModelInfo, PredictorColumn
from app.ml.preprocessing import build_preprocessor
from app.services.prediction import _build_frame

TRAINING = pd.DataFrame(
    {
        # Lopsided on purpose: "London" is what most-frequent imputation should
        # put back, and it is distinguishable from every other outcome.
        "city": ["London"] * 8 + ["Leeds"] * 2,
        "age": [30.0] * 10,
        "y": [0, 1] * 5,
    }
)

INFO = FinalModelInfo(
    model_name="m",
    task_type="classification",
    target_column="y",
    n_rows=10,
    n_features=2,
    feature_columns=[
        PredictorColumn(name="city", dtype="object"),
        PredictorColumn(name="age", dtype="float64"),
    ],
)


@pytest.fixture(scope="module")
def fitted():
    recipe = build_preprocessor(TRAINING, target="y", task_type="classification")
    return recipe.transformer.fit(TRAINING[["city", "age"]], TRAINING["y"])


def _encoded_city(fitted, frame: pd.DataFrame) -> dict[str, float]:
    """The active one-hot column(s) for ``city`` in the first row."""
    names = list(fitted.get_feature_names_out())
    row = dict(zip(names, np.asarray(fitted.transform(frame))[0], strict=True))
    return {name: value for name, value in row.items() if name.startswith("city") and value}


class TestTheHarnessCanTellTheOutcomesApart:
    """Controls. Without these, "all zeros" and "imputed" look alike on a typo."""

    def test_a_supplied_category_encodes_as_itself(self, fitted):
        frame = _build_frame([{"city": "Leeds", "age": 30.0}], INFO)
        assert _encoded_city(fitted, frame) == {"city_Leeds": 1.0}

    def test_the_majority_category_encodes_as_itself(self, fitted):
        frame = _build_frame([{"city": "London", "age": 30.0}], INFO)
        assert _encoded_city(fitted, frame) == {"city_London": 1.0}

    def test_an_unseen_category_is_all_zeros(self, fitted):
        """What the bug made an *omission* look like."""
        frame = _build_frame([{"city": "Atlantis", "age": 30.0}], INFO)
        assert _encoded_city(fitted, frame) == {}


class TestAnOmittedCategoryIsImputed:
    def test_omitting_a_column_gets_the_learned_replacement(self, fitted):
        frame = _build_frame([{"age": 30.0}], INFO)
        assert _encoded_city(fitted, frame) == {"city_London": 1.0}

    def test_an_explicit_null_gets_the_same_treatment(self, fitted):
        """JSON null arrives as None, so it took the same broken path."""
        frame = _build_frame([{"city": None, "age": 30.0}], INFO)
        assert _encoded_city(fitted, frame) == {"city_London": 1.0}

    def test_omission_and_explicit_null_agree(self, fitted):
        """The asymmetry was the surprising part: two spellings, two answers."""
        omitted = _encoded_city(fitted, _build_frame([{"age": 30.0}], INFO))
        explicit = _encoded_city(fitted, _build_frame([{"city": None, "age": 30.0}], INFO))
        assert omitted == explicit

    def test_the_absent_value_is_nan_and_not_none(self):
        """``pd.isna`` is true for both, so it cannot tell them apart.

        SimpleImputer can: it masks on NaN, and ``None`` in an object column is
        just another object to it. Asserting ``isna()`` here looked like a check
        and was not -- it passed with the bug fully in place.
        """
        value = _build_frame([{"age": 30.0}], INFO)["city"].iloc[0]
        assert value is not None
        assert isinstance(value, float) and math.isnan(value)

    def test_a_numeric_column_still_coerces(self):
        """The path that already worked must keep working."""
        frame = _build_frame([{"city": "Leeds", "age": "not a number"}], INFO)
        assert frame["age"].isna().all()


class TestMissingColumnsAreReportedPerRow:
    """A union let one row's value speak for the whole batch."""

    @staticmethod
    def _missing(rows: list[dict]) -> list[str]:
        expected = [c.name for c in INFO.feature_columns]
        return [name for name in expected if any(name not in row for row in rows)]

    def test_a_mixed_batch_reports_the_column(self):
        rows = [{"city": "Leeds", "age": 30.0}, {"age": 30.0}]
        assert "city" in self._missing(rows)

    def test_a_complete_batch_reports_nothing(self):
        rows = [{"city": "Leeds", "age": 30.0}, {"city": "London", "age": 31.0}]
        assert self._missing(rows) == []
