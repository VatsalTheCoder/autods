"""Tests for reading a free-text column instead of counting how often it repeats.

Three things have to hold together for a description column to reach the model
as something useful, and they fail in different places:

* ``TfidfFeatures`` has to turn prose into a bounded number of numeric columns
  without learning anything from rows it was not fitted on. It is the most
  leakage-prone transformer in the project -- an IDF weight is a statistic over
  every document it saw -- so the fold tests matter more here than anywhere else.
* The strategy agent has to *route* prose there, while leaving the columns that
  were already handled correctly exactly where they were. A change that improved
  descriptions by reclassifying every ticket subject would not be an improvement.
* ``build_preprocessor`` has to wire the two text encodings side by side, since a
  dataset may hold one of each.

The dataset behind these tests is a New York real-estate file whose ``listPrice``
had no location column of any kind -- location lived only in a 168-word
description that frequency encoding turned into the constant 1.0 for 99.1% of
rows.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted

from app.agents.feature_strategy import (
    MIN_WORDS_FOR_PROSE,
    default_strategy,
    is_long_form,
    reconcile,
)
from app.ml.contracts import ColumnStrategy
from app.ml.encoders import FrequencyEncoder, TfidfFeatures
from app.ml.preprocessing import build_preprocessor

# Long enough to read as prose, and written so that two topics genuinely differ:
# the vocabulary has to carry signal or SVD has nothing to find.
_BROOKLYN = (
    "Discover an exceptional investment opportunity in the heart of East New York "
    "with this all brick six unit residential building featuring spacious two "
    "bedroom apartments near multiple train lines and local shops with a shared "
    "driveway garage and a large finished basement throughout the property"
)
_MANHATTAN = (
    "A bright and quiet prewar cooperative apartment on a tree lined Upper West "
    "Side street offering high beamed ceilings hardwood floors generous closet "
    "space a renovated windowed kitchen and a full time doorman moments from "
    "Central Park museums and express subway service downtown"
)


def prose_frame(n: int = 60) -> pd.DataFrame:
    """A frame whose text column predicts its target, for end-to-end checks."""
    descriptions = [_BROOKLYN if i % 2 else _MANHATTAN for i in range(n)]
    return pd.DataFrame(
        {
            "description": descriptions,
            "sqft": [900 + (i % 5) * 50 for i in range(n)],
            "price": [400_000 if i % 2 else 1_200_000 for i in range(n)],
        }
    )


class TestTfidfFeaturesProducesUsableColumns:
    def test_prose_becomes_a_bounded_block_of_numbers(self):
        """The whole point: a vocabulary's worth of terms, in a fixed width."""
        frame = prose_frame()[["description"]]
        out = TfidfFeatures(n_components=8).fit_transform(frame)

        assert out.shape == (len(frame), 8)
        assert out.notna().all().all()
        # A projection that returned the same number for every row would be
        # bounded and useless; the two descriptions have to land apart.
        assert out.iloc[0].to_numpy().tolist() != out.iloc[1].to_numpy().tolist()

    def test_output_names_are_prefixed_by_the_source_column(self):
        frame = prose_frame()[["description"]]
        encoder = TfidfFeatures(n_components=4).fit(frame)
        assert list(encoder.get_feature_names_out()) == [
            f"description_topic_{i}" for i in range(1, 5)
        ]

    def test_two_text_columns_do_not_collide(self):
        """One shared vocabulary would merge them; one per column keeps them apart."""
        frame = pd.DataFrame(
            {
                "description": [_BROOKLYN, _MANHATTAN] * 10,
                "agent_notes": [_MANHATTAN, _BROOKLYN] * 10,
            }
        )
        encoder = TfidfFeatures(n_components=3).fit(frame)
        names = list(encoder.get_feature_names_out())

        assert len(names) == len(set(names)) == 6
        assert {"description_topic_1", "agent_notes_topic_1"} <= set(names)
        assert encoder.transform(frame).shape == (20, 6)

    def test_the_number_of_columns_matches_the_names(self):
        """``ColumnTransformer`` names the output from one and sizes it from the
        other, so a mismatch surfaces as a mislabelled report rather than a crash."""
        frame = prose_frame()[["description"]]
        encoder = TfidfFeatures(n_components=5).fit(frame)
        assert encoder.transform(frame).shape[1] == len(encoder.get_feature_names_out())

    def test_it_refuses_to_transform_before_being_fitted(self):
        with pytest.raises(NotFittedError):
            TfidfFeatures().transform(prose_frame()[["description"]])


class TestTfidfFeaturesSurvivesAwkwardColumns:
    """The inputs a real CSV supplies that a clean fixture does not."""

    def test_a_missing_description_is_an_empty_document(self):
        """Not the string "nan", which would become a token the model can split on."""
        frame = pd.DataFrame({"description": [_BROOKLYN, _MANHATTAN, None] * 5})
        encoder = TfidfFeatures(n_components=3).fit(frame)

        vocabulary = encoder.vectorizers_["description"].vocabulary_
        assert "nan" not in vocabulary
        assert not encoder.transform(frame).isna().any().any()

    def test_a_vocabulary_smaller_than_the_requested_width_skips_the_projection(self):
        """Reducing three terms to 120 dimensions is a rotation, not a reduction.

        The outputs are then named for what they actually are -- terms rather
        than topics -- so a report cannot describe a passthrough as a projection.
        """
        frame = pd.DataFrame({"description": ["alpha beta gamma"] * 10})
        encoder = TfidfFeatures(n_components=120).fit(frame)

        names = list(encoder.get_feature_names_out())
        assert names and all("_term_" in name for name in names)
        assert encoder.decomposers_["description"] is None
        assert encoder.transform(frame).shape[1] == len(names)

    def test_a_column_with_no_usable_words_contributes_nothing(self):
        """Stop words and punctuation only. Zero columns beats raising in a fold."""
        frame = pd.DataFrame({"description": ["the and of", "a an the", "!!! ..."] * 5})
        encoder = TfidfFeatures(n_components=4).fit(frame)

        assert list(encoder.get_feature_names_out()) == []
        assert encoder.transform(frame).shape == (15, 0)

    def test_a_tiny_fold_relaxes_min_df_rather_than_failing(self):
        """``min_df=3`` can prune every term on a small fold; the run must survive."""
        frame = pd.DataFrame({"description": [_BROOKLYN, _MANHATTAN]})
        out = TfidfFeatures(n_components=4, min_df=3).fit_transform(frame)
        assert out.shape[0] == 2
        assert out.shape[1] >= 1

    def test_it_works_behind_an_imputer_that_strips_column_names(self):
        """A ``SimpleImputer`` hands on a bare array, so names arrive separately.

        This is the bug ``_names_in`` exists for, checked against the arrangement
        ``_text_branches`` actually builds.
        """
        frame = pd.DataFrame({"description": [_BROOKLYN, _MANHATTAN, None] * 5})
        pipeline = Pipeline(
            steps=[
                ("impute", SimpleImputer(strategy="constant", fill_value="missing")),
                ("encode", TfidfFeatures(n_components=3)),
            ]
        )
        pipeline.fit(frame)

        names = list(pipeline.get_feature_names_out())
        assert names == [f"description_topic_{i}" for i in range(1, 4)]


class TestVocabularyComesFromTheFittedRowsOnly:
    """The leakage guarantee, at the level the transformer can be held to.

    An IDF weight is a statistic over every document the vectorizer saw. Fitted
    over a whole dataset it would carry, into every training row, a summary of the
    held-out fold -- the same failure ``FrequencyEncoder`` is guarded against, and
    harder to spot because the resulting features look perfectly reasonable.
    """

    def test_a_word_only_in_held_out_rows_is_out_of_vocabulary(self):
        train = pd.DataFrame({"description": [_BROOKLYN] * 10})
        held_out = pd.DataFrame({"description": [_BROOKLYN + " penthouse solarium"] * 3})

        encoder = TfidfFeatures(n_components=3, min_df=1).fit(train)
        assert "penthouse" not in encoder.vectorizers_["description"].vocabulary_

        # And it transforms rather than raising: unseen words are simply ignored.
        assert encoder.transform(held_out).shape == (3, 3)

    def test_fitting_on_different_rows_learns_a_different_vocabulary(self):
        brooklyn = TfidfFeatures(n_components=2, min_df=1).fit(
            pd.DataFrame({"description": [_BROOKLYN] * 8})
        )
        manhattan = TfidfFeatures(n_components=2, min_df=1).fit(
            pd.DataFrame({"description": [_MANHATTAN] * 8})
        )

        assert (
            brooklyn.vectorizers_["description"].vocabulary_.keys()
            != manhattan.vectorizers_["description"].vocabulary_.keys()
        )

    def test_the_recipe_is_handed_back_unfitted(self):
        """``build_preprocessor`` constructs; the folds fit. Nothing learned yet."""
        transformer = build_preprocessor(prose_frame(), target="price").transformer
        with pytest.raises(NotFittedError):
            check_is_fitted(transformer)


class TestProseIsRoutedToTfidf:
    def test_a_description_column_is_read_rather_than_dropped(self):
        """Near-unique *and* the most informative column in its dataset."""
        descriptions = pd.Series([f"{_BROOKLYN} unit {i}" for i in range(50)])
        strategy = default_strategy("description", descriptions)

        assert strategy.role == "text"
        assert strategy.encode == "tfidf"
        assert "168" not in strategy.rationale  # the count is measured, not hardcoded
        assert "words" in strategy.rationale

    def test_a_description_is_imputed_with_a_constant_not_the_commonest_one(self):
        """Filling an absent description with the most frequent one invents text."""
        descriptions = pd.Series([f"{_BROOKLYN} unit {i}" for i in range(50)])
        assert default_strategy("description", descriptions).impute == "constant"

    def test_near_unique_short_strings_are_still_dropped(self):
        """The Airbnb ``name`` case, which this change must not reclassify."""
        titles = pd.Series([f"Cozy {i}BR loft in Bushwick" for i in range(200)])

        assert not is_long_form(titles)
        assert default_strategy("name", titles).role == "drop"

    def test_low_cardinality_labels_are_untouched(self):
        cities = pd.Series(["London", "Leeds", "York"] * 40)
        assert default_strategy("city", cities).encode == "onehot"

    def test_repeated_labels_too_many_to_onehot_still_get_frequency(self):
        """High cardinality without prose is what frequency encoding is for."""
        skus = pd.Series([f"SKU-{i % 120}" for i in range(600)])
        strategy = default_strategy("sku", skus)
        assert strategy.role == "text"
        assert strategy.encode == "frequency"

    def test_the_threshold_is_word_count_not_string_length(self):
        """A long URL is not prose; a short paragraph is."""
        urls = pd.Series(
            [f"https://example.com/listings/{i}/photos/gallery/full" for i in range(80)]
        )
        assert not is_long_form(urls)

        sentences = pd.Series([" ".join(f"word{j}" for j in range(MIN_WORDS_FOR_PROSE))] * 30)
        assert is_long_form(sentences)


class TestTheStrategyAgentIsHeldToTheSameRule:
    """``reconcile`` is the path a real run takes, with the LLM's answer in hand."""

    @pytest.fixture
    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "description": [f"{_BROOKLYN} unit {i}" for i in range(40)],
                "price": list(range(40)),
            }
        )

    def test_frequency_on_near_unique_prose_is_overridden(self, frame):
        """The exact failure this pair of predicates exists to catch."""
        result = reconcile(
            [ColumnStrategy(column="description", role="text", encode="frequency")],
            frame,
            target="price",
        )

        assert result.columns[0].encode == "tfidf"
        override = next(o for o in result.overrides if o.field == "encode")
        assert override.requested == "frequency"
        assert override.applied == "tfidf"
        assert "constant" in override.reason

    def test_an_omitted_encoding_defaults_to_tfidf_for_prose(self, frame):
        result = reconcile(
            [ColumnStrategy(column="description", role="text")], frame, target="price"
        )
        assert result.columns[0].encode == "tfidf"

    def test_a_deliberate_drop_is_still_honoured(self, frame):
        """Usefulness is the judgement the LLM is here to make; this is not a veto."""
        result = reconcile(
            [ColumnStrategy(column="description", role="drop")], frame, target="price"
        )
        assert result.columns[0].role == "drop"

    def test_prose_is_no_longer_forced_to_drop_by_near_uniqueness(self, frame):
        """``_coerce_role`` drops near-unique text; prose has to be exempt."""
        result = reconcile(
            [ColumnStrategy(column="description", role="text", encode="tfidf")],
            frame,
            target="price",
        )
        assert result.columns[0].role == "text"
        assert not [o for o in result.overrides if o.field == "role"]


class TestThePreprocessorWiresBothTextEncodings:
    def test_a_tfidf_strategy_builds_a_tfidf_step(self):
        transformer = build_preprocessor(prose_frame(), target="price").transformer

        branch = next(b for name, b, _ in transformer.transformers if name.startswith("text_"))
        assert isinstance(branch.named_steps["encode"], TfidfFeatures)
        assert not isinstance(branch.named_steps["encode"], FrequencyEncoder)

    def test_the_two_encodings_coexist_in_one_dataset(self):
        """Grouping by encode as well as impute/scale is what makes this work."""
        # The SKU repeats often enough to stay frequency-encoded: a column with
        # one distinct value per row is dropped, not encoded, so a fixture that
        # never repeats would test nothing.
        frame = pd.DataFrame(
            {
                "description": [f"{_BROOKLYN} unit {i}" for i in range(300)],
                "sku": [f"SKU-{i % 120}" for i in range(300)],
                "price": list(range(300)),
            }
        )
        transformer = build_preprocessor(frame, target="price").transformer

        encoders = [
            type(branch.named_steps["encode"]).__name__
            for name, branch, _ in transformer.transformers
            if name.startswith("text_")
        ]
        assert sorted(encoders) == ["FrequencyEncoder", "TfidfFeatures"]

    def test_the_fitted_pipeline_names_every_column_it_produces(self):
        """What the SHAP charts and the report read; a gap here mislabels them."""
        frame = prose_frame()
        transformer = build_preprocessor(frame, target="price").transformer
        features = frame.drop(columns=["price"])

        transformed = transformer.fit_transform(features, frame["price"])
        names = list(transformer.get_feature_names_out())

        assert transformed.shape[1] == len(names)
        assert any(name.startswith("description_topic_") for name in names)
        assert "sqft" in names


class TestItActuallyHelps:
    """The claim the change is made on, held to a number.

    Not a benchmark -- the fixture is synthetic and the margin is enormous by
    construction. It is here so that wiring the encoder up wrongly, in a way that
    produces well-shaped columns carrying nothing, fails a test rather than
    quietly shipping.
    """

    def test_reading_the_description_beats_ignoring_it(self):
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.model_selection import cross_val_score

        frame = prose_frame(80)
        # The text is the only thing that separates the two prices: sqft cycles
        # independently of them, so a model without the description cannot do
        # better than predicting the mean.
        target = frame["price"]

        with_text = build_preprocessor(frame, target="price").transformer
        without_text = build_preprocessor(
            frame.drop(columns=["description"]), target="price"
        ).transformer

        def scored(transformer, features):
            pipeline = Pipeline(
                steps=[("pre", transformer), ("model", RandomForestRegressor(random_state=0))]
            )
            return float(np.mean(cross_val_score(pipeline, features, target, cv=3, scoring="r2")))

        assert scored(with_text, frame.drop(columns=["price"])) > 0.9
        assert scored(without_text, frame.drop(columns=["price", "description"])) < 0.1
