"""The two transformers scikit-learn does not ship, written to be fitted in-fold.

Section 5's recipe named datetime and high-cardinality text columns as unhandled
and dropped them. Section 7 gives them real handling, and both need a transformer
that does not exist in scikit-learn: one to turn a timestamp into calendar
numbers, one to encode a column with hundreds of distinct values without adding
hundreds of columns.

Writing them as proper ``BaseEstimator`` / ``TransformerMixin`` classes is not
ceremony -- it is the thing that keeps the leakage guarantee true. A helper
function that added ``signed_up_month`` to the DataFrame would run once, over
every row, outside any fold. As pipeline steps these are cloned and fitted per
fold with everything else, so they cannot be the one component that saw the
held-out rows. ``tests/test_leakage.py`` asserts that against the real objects.

The distinction that decides which of the two designs a step needs:

* ``DatetimeFeatures`` learns **nothing**. The month of a timestamp is a fact
  about that row, not a statistic over the dataset, so its ``fit`` only records
  column names. It would be safe outside a fold -- it is a step here for
  uniformity, and because a stateless step costs nothing.
* ``FrequencyEncoder`` learns **a distribution**, which is exactly the kind of
  thing that leaks. How often "London" appears is a property of the rows it was
  fitted on. Computed over the whole dataset it would encode, into every training
  row, a summary of the test fold. So it is fitted per fold and unseen values map
  to zero -- the same honest answer ``handle_unknown="ignore"`` gives the one-hot
  path (``preprocessing.py``).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

# The calendar parts pulled out of a timestamp. Deliberately plain: these are the
# components a tree can split on and a linear model can weight. Cyclical
# encodings (sin/cos of month) are a refinement that would need justifying per
# dataset, and derived spans ("days since signup") need a reference date the
# pipeline does not have.
DATETIME_PARTS: tuple[str, ...] = ("year", "month", "day", "dayofweek", "hour")


class _FrameTransformer(BaseEstimator, TransformerMixin):
    """Shared plumbing: remember the input columns, and accept arrays or frames.

    A ``ColumnTransformer`` branch hands on whatever the previous step returned,
    and scikit-learn's imputers return a bare NumPy array unless asked otherwise.
    Recording the column names at ``fit`` and re-attaching them at ``transform``
    means these two work the same either way, rather than depending on a
    ``set_output`` call somewhere else staying in place.
    """

    def _fit_columns(self, X) -> pd.DataFrame:  # noqa: N803 - sklearn's parameter name
        frame = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        self.columns_ = [str(c) for c in frame.columns]
        self.n_features_in_ = frame.shape[1]
        # Return the frame under the same names that were just recorded. An array
        # arriving from an imputer carries integer labels, and fitting against
        # those while ``columns_`` holds their string forms is a mismatch waiting
        # to happen -- normalising once, here, is what keeps the two in step.
        return self._as_frame(frame)

    def _as_frame(self, X) -> pd.DataFrame:  # noqa: N803 - sklearn's parameter name
        frame = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        # Compared against the labels themselves, not their string forms: an
        # array's integer label 0 stringifies to the recorded "0" and would look
        # like a match while ``frame["0"]`` still raised.
        if list(frame.columns) == self.columns_[: frame.shape[1]]:
            return frame
        # Positional renaming is correct here and only here: a ColumnTransformer
        # branch always hands its steps the same columns in the same order, so
        # position identifies a column as reliably as its name would.
        frame = frame.copy()
        frame.columns = self.columns_[: frame.shape[1]]
        return frame


class DatetimeFeatures(_FrameTransformer):
    """Split each timestamp column into its calendar parts.

    Stateless by construction: ``fit`` records column names and nothing else, so
    there is no statistic here that could differ between folds. Unparseable
    values and ``NaT`` become ``NaN``, which the imputer after this step fills --
    the same division of labour as every other column, where this module decides
    the shape and a fitted step supplies the numbers.

    Output names are prefixed with the source column (``signed_up_month``), which
    is what lets two datetime columns coexist under the ``ColumnTransformer``'s
    ``verbose_feature_names_out=False``.
    """

    def fit(self, X, y=None):  # noqa: N803 - sklearn's parameter name
        self._fit_columns(X)
        return self

    def transform(self, X):  # noqa: N803 - sklearn's parameter name
        check_is_fitted(self)
        frame = self._as_frame(X)
        out: dict[str, pd.Series] = {}
        for column in self.columns_:
            # ``errors="coerce"`` rather than raising: a column that cleaning
            # believed was a datetime but holds one unparseable string should cost
            # that row a value, not the whole run.
            values = pd.to_datetime(frame[column], errors="coerce")
            for part in DATETIME_PARTS:
                out[f"{column}_{part}"] = getattr(values.dt, part)
        return pd.DataFrame(out, index=frame.index).astype(float)

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        check_is_fitted(self)
        return np.asarray(
            [f"{column}_{part}" for column in self.columns_ for part in DATETIME_PARTS],
            dtype=object,
        )


class FrequencyEncoder(_FrameTransformer):
    """Replace each label with how often it occurred in the training rows.

    The answer to a column with 5,000 distinct values, where one-hot encoding
    would add 5,000 mostly-empty columns and invite the model to memorise. This
    adds exactly one column and keeps the signal that usually matters about such
    a column -- whether a value is common or rare.

    **Proportions, not raw counts.** Folds differ in size, so a count learned on
    one fold's 80 rows is not comparable to one learned on another's 79. A
    proportion is, which keeps the feature meaning the same thing in every fold.

    A value not seen during ``fit`` encodes as 0.0, meaning "this never occurred
    in training" -- which is true, and is the honest reading. The alternative,
    fitting over the whole dataset so no value is ever unseen, is precisely the
    leak this module exists to avoid.
    """

    def fit(self, X, y=None):  # noqa: N803 - sklearn's parameter name
        frame = self._fit_columns(X)
        # Cast to str so that 1 and "1" cannot be counted as two categories, and
        # so a mixed-dtype object column has one consistent key space.
        self.frequencies_ = {
            column: frame[column].astype(str).value_counts(normalize=True).to_dict()
            for column in self.columns_
        }
        return self

    def transform(self, X):  # noqa: N803 - sklearn's parameter name
        check_is_fitted(self)
        frame = self._as_frame(X)
        # Named to match ``get_feature_names_out`` exactly: scikit-learn treats
        # the two as one promise, and a mismatch surfaces as a confusing column
        # error much further down the pipeline.
        out = {
            f"{column}_frequency": frame[column]
            .astype(str)
            .map(self.frequencies_[column])
            .fillna(0.0)
            for column in self.columns_
        }
        return pd.DataFrame(out, index=frame.index).astype(float)

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        check_is_fitted(self)
        return np.asarray([f"{column}_frequency" for column in self.columns_], dtype=object)
