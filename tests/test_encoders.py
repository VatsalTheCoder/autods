"""Tests for the two transformers scikit-learn does not ship.

``DatetimeFeatures`` is the easy one: it learns nothing, so the tests are about
whether it produces the right numbers and survives the awkward inputs real
datasets contain.

``FrequencyEncoder`` is the one that matters. It learns a distribution, which
makes it the component in this section most able to leak -- and the leak would be
invisible, because a frequency map built over the whole dataset produces
perfectly reasonable-looking features. So the tests below pin down that what it
learned came from the rows it was fitted on and nothing else, and that a value it
never saw encodes as "never seen" rather than crashing or being back-filled from
somewhere it should not have looked.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted

from app.ml.encoders import DatetimeFeatures, FrequencyEncoder


class TestDatetimeFeatures:
    @pytest.fixture
    def frame(self) -> pd.DataFrame:
        return pd.DataFrame({"signed_up": pd.to_datetime(["2024-01-15 09:00", "2023-06-30 17:30"])})

    def test_it_pulls_out_the_calendar_parts(self, frame):
        out = DatetimeFeatures().fit_transform(frame)
        assert out.loc[0, "signed_up_year"] == 2024
        assert out.loc[0, "signed_up_month"] == 1
        assert out.loc[0, "signed_up_day"] == 15
        assert out.loc[0, "signed_up_hour"] == 9

    def test_the_output_names_are_prefixed_by_the_source_column(self, frame):
        """Two datetime columns must not both claim to produce "month"."""
        encoder = DatetimeFeatures().fit(frame)
        assert "signed_up_month" in list(encoder.get_feature_names_out())

    def test_two_datetime_columns_do_not_collide(self):
        frame = pd.DataFrame(
            {
                "signed_up": pd.to_datetime(["2024-01-15"]),
                "cancelled": pd.to_datetime(["2024-03-20"]),
            }
        )
        names = list(DatetimeFeatures().fit(frame).get_feature_names_out())
        assert len(names) == len(set(names))
        assert {"signed_up_month", "cancelled_month"} <= set(names)

    def test_a_missing_timestamp_becomes_nan_for_the_imputer(self):
        """NaT has no year to extract; the imputer after this step fills it."""
        frame = pd.DataFrame({"signed_up": pd.to_datetime(["2024-01-15", None])})
        out = DatetimeFeatures().fit_transform(frame)
        assert np.isnan(out.loc[1, "signed_up_year"])

    def test_an_unparseable_value_costs_that_row_not_the_run(self):
        frame = pd.DataFrame({"signed_up": ["2024-01-15", "not a date"]})
        out = DatetimeFeatures().fit_transform(frame)
        assert out.loc[0, "signed_up_year"] == 2024
        assert np.isnan(out.loc[1, "signed_up_year"])

    def test_it_learns_nothing_from_the_rows_it_saw(self):
        """Stateless: the same row transforms the same regardless of the fit set."""
        early = pd.DataFrame({"d": pd.to_datetime(["2020-01-01", "2020-02-01"])})
        late = pd.DataFrame({"d": pd.to_datetime(["2099-11-01", "2099-12-01"])})
        row = pd.DataFrame({"d": pd.to_datetime(["2024-05-09"])})

        assert (
            DatetimeFeatures()
            .fit(early)
            .transform(row)
            .equals(DatetimeFeatures().fit(late).transform(row))
        )


class TestFrequencyEncoder:
    @pytest.fixture
    def frame(self) -> pd.DataFrame:
        # London 3, Leeds 2, Bristol 1.
        return pd.DataFrame({"city": ["London", "London", "London", "Leeds", "Leeds", "Bristol"]})

    def test_a_label_becomes_how_often_it_occurred(self, frame):
        out = FrequencyEncoder().fit_transform(frame)
        assert out.loc[0, "city_frequency"] == pytest.approx(3 / 6)
        assert out.loc[3, "city_frequency"] == pytest.approx(2 / 6)
        assert out.loc[5, "city_frequency"] == pytest.approx(1 / 6)

    def test_it_adds_one_column_not_one_per_label(self, frame):
        assert FrequencyEncoder().fit_transform(frame).shape[1] == 1

    def test_proportions_not_counts_so_folds_stay_comparable(self):
        """Folds differ in size; a raw count would mean different things in each."""
        small = pd.DataFrame({"city": ["London", "Leeds"]})
        large = pd.DataFrame({"city": ["London", "Leeds"] * 50})
        row = pd.DataFrame({"city": ["London"]})

        assert FrequencyEncoder().fit(small).transform(row).iloc[0, 0] == pytest.approx(
            FrequencyEncoder().fit(large).transform(row).iloc[0, 0]
        )

    def test_an_unseen_label_encodes_as_never_seen(self, frame):
        """0.0 is the honest answer, and it is what stops a fold from crashing."""
        out = FrequencyEncoder().fit(frame).transform(pd.DataFrame({"city": ["Nowhere"]}))
        assert out.iloc[0, 0] == 0.0

    def test_what_it_learned_came_only_from_the_rows_it_was_fitted_on(self, frame):
        """The leakage property, stated directly.

        Fitted on the first three rows -- all London -- "London" is the whole
        distribution and everything else is unseen. If the encoder had consulted
        the rows it was not given, Leeds would come back non-zero.
        """
        encoder = FrequencyEncoder().fit(frame.iloc[:3])

        out = encoder.transform(pd.DataFrame({"city": ["London", "Leeds", "Bristol"]}))
        assert out.iloc[0, 0] == pytest.approx(1.0)
        assert out.iloc[1, 0] == 0.0
        assert out.iloc[2, 0] == 0.0

    def test_numbers_and_their_string_forms_are_one_category(self):
        frame = pd.DataFrame({"code": [1, "1", 2]})
        out = FrequencyEncoder().fit_transform(frame)
        assert out.iloc[0, 0] == pytest.approx(2 / 3)

    def test_missing_values_do_not_crash_the_encoder(self):
        """In the pipeline an imputer runs first, but the step must be robust."""
        frame = pd.DataFrame({"city": ["London", None, "London"]})
        assert FrequencyEncoder().fit_transform(frame).notna().all().all()


class TestBothAreProperPipelineSteps:
    """Being real estimators is what puts them inside the fold (test_leakage.py)."""

    @pytest.mark.parametrize("encoder", [DatetimeFeatures(), FrequencyEncoder()])
    def test_unfitted_until_fitted(self, encoder):
        with pytest.raises(NotFittedError):
            check_is_fitted(encoder)

    @pytest.mark.parametrize("encoder", [DatetimeFeatures(), FrequencyEncoder()])
    def test_transform_before_fit_is_refused(self, encoder):
        with pytest.raises(NotFittedError):
            encoder.transform(pd.DataFrame({"a": ["x"]}))

    @pytest.mark.parametrize("encoder", [DatetimeFeatures(), FrequencyEncoder()])
    def test_they_accept_the_arrays_an_imputer_hands_on(self, encoder):
        """Imputers return bare arrays, so the column names have to be re-attached."""
        frame = pd.DataFrame({"a": ["2024-01-01", "2024-02-01"]})
        encoder.fit(frame)
        assert len(encoder.transform(frame.to_numpy())) == 2
