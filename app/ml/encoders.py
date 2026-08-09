"""The transformers scikit-learn does not ship, written to be fitted in-fold.

Section 5's recipe named datetime and high-cardinality text columns as unhandled
and dropped them. Section 7 gives them real handling, and both need a transformer
that does not exist in scikit-learn: one to turn a timestamp into calendar
numbers, one to encode a column with hundreds of distinct values without adding
hundreds of columns.

A third arrived later, for the columns the second one could not help. Frequency
encoding asks how often a value repeats, which is a good question about a city
name and a useless one about a paragraph -- so prose gets ``TfidfFeatures``
instead, and the choice between them is made in ``feature_strategy.py``.

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
* ``TfidfFeatures`` learns **a vocabulary and its document frequencies**, and is
  the most leakage-prone of the three: an IDF weight is a statistic over every
  document it was fitted on. Per fold like the others, and words that appear only
  in held-out rows are simply out of vocabulary there.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.utils.validation import check_is_fitted

logger = logging.getLogger(__name__)

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

    def _names_in(self, input_features=None) -> list[str]:
        """The source column names to build output names from.

        ``columns_`` is only as good as what ``fit`` was handed, and a step that
        sits *behind* an imputer is handed a bare NumPy array -- so ``columns_``
        holds ``["0"]`` and naming from it produces ``0_frequency`` instead of
        ``email_frequency``. That is what ``input_features`` is for: scikit-learn
        threads the real names down a ``Pipeline`` and offers them to each step's
        ``get_feature_names_out``, precisely so a step that lost them can still
        name its output correctly. Preferring the argument over ``columns_`` is
        the fix, and it is why ``DatetimeFeatures`` never had the bug -- it is
        first in its branch and so receives a real DataFrame.

        ``columns_`` remains the fallback for a direct call with no argument,
        where it is the only thing available and is usually right.
        """
        if input_features is None:
            return list(self.columns_)
        names = [str(c) for c in input_features]
        if len(names) != self.n_features_in_:
            raise ValueError(
                f"{type(self).__name__} was fitted on {self.n_features_in_} columns "
                f"but input_features names {len(names)}."
            )
        return names


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
            [
                f"{column}_{part}"
                for column in self._names_in(input_features)
                for part in DATETIME_PARTS
            ],
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
        # Named from ``columns_``, which matches ``get_feature_names_out`` for a
        # direct call. Behind an imputer the two legitimately differ -- these
        # labels are then the positional stand-ins the branch works in, while
        # ``get_feature_names_out(input_features)`` supplies the real names to
        # whoever is naming the output. The *order* is what has to agree, and it
        # does: one output per fitted column, in fitted order.
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
        return np.asarray(
            [f"{column}_frequency" for column in self._names_in(input_features)], dtype=object
        )


class TfidfFeatures(_FrameTransformer):
    """Reduce a free-text column to a handful of numeric topic scores.

    The third answer to a high-cardinality string column, and the one for prose.
    ``FrequencyEncoder`` asks "how often does this exact value occur", which is a
    good question about a city name and a useless one about a paragraph -- every
    paragraph occurs once. This asks what the paragraph *says*.

    A New York real-estate dataset is the case it was built for. Its
    ``listPrice`` had no location feature of any kind: no zip, no borough, no
    coordinates. Location lived entirely inside a 168-word listing description,
    which frequency encoding turned into the constant 1.0 for 99.1% of rows.
    Routing that column here instead cut median absolute error by 38%.

    **Two stages, for two different reasons.** TF-IDF turns documents into a
    sparse matrix with one dimension per term, which is the right representation
    and the wrong shape -- tens of thousands of mostly-empty columns handed to a
    gradient-boosted tree is a slow way to overfit. ``TruncatedSVD`` projects that
    down to ``n_components`` dense columns, each a weighted blend of terms that
    co-occur. It is the standard latent-semantic pairing, and it is what keeps
    one text column costing the model a bounded number of features rather than a
    vocabulary's worth.

    **Cost is bounded on purpose**, because this runs once per model per fold.
    ``max_features`` caps the vocabulary, ``min_df`` discards terms too rare to
    generalise, and ``n_components`` fixes the output width no matter how much
    text arrives.
    """

    def __init__(
        self,
        n_components: int = 120,
        max_features: int = 30_000,
        min_df: int = 3,
        ngram_range: tuple[int, int] = (1, 2),
        random_state: int = 42,
    ):
        self.n_components = n_components
        self.max_features = max_features
        self.min_df = min_df
        self.ngram_range = ngram_range
        self.random_state = random_state

    def fit(self, X, y=None):  # noqa: N803 - sklearn's parameter name
        frame = self._fit_columns(X)
        # One vectorizer per column, never one shared across them: "studio" in a
        # property description and "studio" in a neighbourhood note are different
        # words, and a shared vocabulary would collapse them into one dimension.
        self.vectorizers_: dict[str, TfidfVectorizer | None] = {}
        self.decomposers_: dict[str, TruncatedSVD | None] = {}
        self.widths_: list[int] = []
        self.kinds_: list[str] = []

        for column in self.columns_:
            documents = self._documents(frame[column])
            vectorizer, matrix = self._fit_vectorizer(column, documents)
            self.vectorizers_[column] = vectorizer
            if vectorizer is None:
                self.decomposers_[column] = None
                self.widths_.append(0)
                self.kinds_.append("topic")
                continue

            # SVD only earns its place when there is something to reduce. A
            # vocabulary already no wider than the target would be *narrowed* by
            # projecting it -- three terms into two components discards a third of
            # the information to save nothing -- so the sparse matrix is densified
            # as-is and the outputs are named for what they are: terms, not
            # topics. The document count bounds the width too, since a projection
            # cannot have more components than there are rows to fit it on.
            n_terms = int(matrix.shape[1])
            width = min(self.n_components, int(matrix.shape[0]) - 1)
            if width < 1 or n_terms <= width:
                self.decomposers_[column] = None
                self.widths_.append(n_terms)
                self.kinds_.append("term")
                continue

            decomposer = TruncatedSVD(n_components=width, random_state=self.random_state)
            decomposer.fit(matrix)
            self.decomposers_[column] = decomposer
            self.widths_.append(width)
            self.kinds_.append("topic")

        return self

    def transform(self, X):  # noqa: N803 - sklearn's parameter name
        check_is_fitted(self)
        frame = self._as_frame(X)
        blocks: list[np.ndarray] = []
        for column in self.columns_:
            vectorizer = self.vectorizers_[column]
            if vectorizer is None:
                continue
            matrix = vectorizer.transform(self._documents(frame[column]))
            decomposer = self.decomposers_[column]
            blocks.append(
                decomposer.transform(matrix) if decomposer is not None else matrix.toarray()
            )

        if not blocks:
            # Every column had an unusable vocabulary. An empty block is the
            # honest output, and ``ColumnTransformer`` drops a branch that
            # produces no columns rather than failing the fit.
            return pd.DataFrame(index=frame.index)
        return pd.DataFrame(
            np.hstack(blocks), index=frame.index, columns=self._names(self.columns_)
        ).astype(float)

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        check_is_fitted(self)
        return np.asarray(self._names(self._names_in(input_features)), dtype=object)

    # ---- internals ----------------------------------------------------------

    def _documents(self, series: pd.Series) -> list[str]:
        """The column as plain strings, with missing values as empty documents.

        An absent description is not the string ``"nan"``, and letting the default
        cast produce one would put that token in the vocabulary and let the model
        split on it. Empty is what "no text" means, and TF-IDF handles it as a row
        of zeros.
        """
        return series.astype(str).replace({"nan": "", "None": "", "<NA>": ""}).fillna("").tolist()

    def _fit_vectorizer(self, column: str, documents: list[str]):
        """Fit TF-IDF, relaxing ``min_df`` once before giving up on the column.

        ``min_df`` is a guard against terms too rare to generalise, but on a small
        fold it can prune the entire vocabulary and raise. Falling back to
        ``min_df=1`` keeps a small dataset working at the cost of noisier terms,
        which is the better trade -- the alternative is a run that dies inside a
        fold on data the user was told was fine.
        """
        for min_df in (self.min_df, 1):
            try:
                vectorizer = TfidfVectorizer(
                    max_features=self.max_features,
                    min_df=min_df,
                    ngram_range=self.ngram_range,
                    sublinear_tf=True,
                    stop_words="english",
                    strip_accents="unicode",
                    lowercase=True,
                )
                matrix = vectorizer.fit_transform(documents)
            except ValueError:
                continue
            if matrix.shape[1]:
                return vectorizer, matrix

        # Nothing but stop words, punctuation or empty strings. Reported once, at
        # fit, rather than silently emitting a block of zero columns.
        logger.info(
            "Text column %r yielded no usable vocabulary; it contributes no features.", column
        )
        return None, None

    def _names(self, columns: list[str]) -> list[str]:
        return [
            f"{column}_{kind}_{index}"
            for column, width, kind in zip(columns, self.widths_, self.kinds_, strict=True)
            for index in range(1, width + 1)
        ]
