"""Tests for SHAP explanations -- Section 8's substance (spec 7.10).

Four things here are worth more than the rest, and each was a real failure mode
found while building this rather than a hypothetical:

* **The name mapping.** ``city_London`` has to lead back to ``city`` through every
  branch type the recipe can build, including the ones that lose their column
  names on the way (an imputer hands on a bare array). A mislabelled feature is
  worse than a missing one, because it gets believed.
* **Additivity.** Contributions that do not reconstruct the model's output are
  not a decomposition of anything. Checked against each family's own raw output,
  because ``TreeExplainer`` works in probability space for a forest and in log-odds
  for a booster -- comparing against the wrong one would fail a good explanation.
* **The class axis.** Multiclass SHAP values are indexed by the model's *encoded*
  class order. Explaining a row against the wrong column of that axis produces
  output that looks entirely reasonable and is about a different class.
* **Dispatch.** A linear model given ``TreeExplainer`` raises; a tree model given
  ``LinearExplainer`` is meaningless. The roster has both.

No LLM anywhere: strategies are constructed directly, which is also how the
tests pin the exact branch shapes they want to exercise.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.naive_bayes import GaussianNB

from app.ml.contracts import ColumnStrategy, FeatureStrategy
from app.ml.explain import (
    ExplainabilityError,
    explain_model,
    source_columns,
)
from app.ml.final_training import train_final_model
from app.ml.modeling import build_pipeline
from app.ml.preprocessing import build_preprocessor

N_ROWS = 120

TREE_MODELS = ["RandomForest", "XGBoost", "LightGBM"]
ALL_MODELS = [*TREE_MODELS, "LogisticRegression"]


def churn_frame(n: int = N_ROWS, *, classes: list[str] | None = None) -> pd.DataFrame:
    """A small dataset with real signal, so importances are not noise.

    ``support_calls`` and ``tenure_months`` decide the label. That is deliberate:
    a test asserting that the top feature is the one the data was built around is
    only meaningful if the data was built around one.
    """
    rng = np.random.default_rng(0)
    labels = classes or ["yes", "no"]
    support = rng.integers(0, 10, n)
    tenure = rng.integers(1, 60, n)
    # Banded rather than thresholded, so a three-label request really produces
    # three classes -- a "multiclass" test on a two-class dataset tests nothing.
    width = 10 / len(labels)
    target = [labels[min(int(s // width), len(labels) - 1)] for s in support]
    return pd.DataFrame(
        {
            "age": rng.integers(18, 80, n),
            "city": rng.choice(["London", "Leeds", "Bristol"], n),
            "support_calls": support,
            "tenure_months": tenure,
            "churned": target,
        }
    )


def fitted(
    frame: pd.DataFrame,
    *,
    model_name: str,
    target: str = "churned",
    task_type: str = "classification",
    strategy: FeatureStrategy | None = None,
    select_k: int | None = None,
):
    """The fitted final pipeline for one candidate, as the node would build it."""
    recipe = build_preprocessor(
        frame, target=target, strategy=strategy, select_k=select_k, task_type=task_type
    ).transformer
    return train_final_model(
        frame,
        target=target,
        task_type=task_type,
        preprocessor=recipe,
        model_name=model_name,
    ).pipeline


def explained(frame: pd.DataFrame, *, model_name: str, **kw):
    pipeline = fitted(frame, model_name=model_name, **kw)
    return explain_model(
        pipeline,
        frame,
        target=kw.get("target", "churned"),
        task_type=kw.get("task_type", "classification"),
        model_name=model_name,
    )


class TestFeatureNameMapping:
    """The unglamorous half of Section 8, and the half most likely to be wrong."""

    def test_every_branch_type_leads_back_to_its_source_column(self):
        """One column of each role, through the whole recipe, named correctly.

        The text branch is the one that used to fail: ``FrequencyEncoder`` sits
        behind an imputer, which hands on a bare array, so the encoder had no
        column names to work from and produced ``0_frequency``.
        """
        frame = pd.DataFrame(
            {
                "email": [f"user{i}@example.com" for i in range(40)],
                "signed_up": pd.date_range("2021-01-01", periods=40, freq="D").astype(str),
                "city": ["London", "Leeds"] * 20,
                "size": ["small", "large"] * 20,
                "age": range(40),
                "churned": ["yes", "no"] * 20,
            }
        )
        strategy = FeatureStrategy(
            columns=[
                ColumnStrategy(column="email", role="text", impute="constant", encode="frequency"),
                ColumnStrategy(
                    column="signed_up", role="datetime", impute="median", scale="standard"
                ),
                ColumnStrategy(
                    column="city", role="categorical", impute="most_frequent", encode="onehot"
                ),
                ColumnStrategy(
                    column="size",
                    role="ordinal",
                    impute="most_frequent",
                    encode="ordinal",
                    ordinal_order=["small", "large"],
                ),
                ColumnStrategy(column="age", role="numeric", impute="median", scale="standard"),
            ],
            source="llm",
        )
        recipe = build_preprocessor(frame, target="churned", strategy=strategy).transformer
        recipe.fit(frame.drop(columns=["churned"]))

        mapping = source_columns(recipe)
        assert mapping["age"] == "age"
        assert mapping["size"] == "size"
        assert mapping["city_London"] == "city"
        assert mapping["signed_up_month"] == "signed_up"
        assert mapping["email_frequency"] == "email"
        # Nothing may be left pointing at a positional stand-in.
        assert not [name for name in mapping if name.startswith("0_")]

    def test_the_longest_matching_column_wins(self):
        """``city_tier_London`` belongs to ``city_tier``, not to ``city``.

        Both are prefixes. Taking the first match would attribute a column's
        influence to a different column, silently and plausibly.
        """
        frame = pd.DataFrame(
            {
                "city": ["London", "Leeds"] * 20,
                "city_tier": ["London", "Leeds"] * 20,
                "churned": ["yes", "no"] * 20,
            }
        )
        recipe = build_preprocessor(frame, target="churned").transformer
        recipe.fit(frame.drop(columns=["churned"]))

        mapping = source_columns(recipe)
        assert mapping["city_tier_London"] == "city_tier"
        assert mapping["city_London"] == "city"

    def test_feature_selection_does_not_break_the_mapping(self):
        """With selection on, the recipe is a Pipeline wrapping the transformer.

        The encoded names then come from the outer object (the selector drops
        some) while the mapping comes from the inner one -- so the two have to be
        read from different places, and this is the test that says so.
        """
        frame = churn_frame()
        result = explained(frame, model_name="RandomForest", select_k=3)

        assert result.report.n_encoded_features == 3
        # Every surviving feature still names the column it came from, and the
        # selector's own output names are not among them.
        assert set(result.report.feature_name_mapping.values()) <= set(frame.columns)
        assert result.report.global_importance

    def test_the_mapping_is_published_in_the_report(self):
        """A reader has to be able to check the translation, not just trust it."""
        report = explained(churn_frame(), model_name="RandomForest").report
        assert report.feature_name_mapping
        assert all(origin for origin in report.feature_name_mapping.values())


class TestExplainerDispatch:
    @pytest.mark.parametrize("model_name", TREE_MODELS)
    def test_tree_families_get_the_exact_tree_explainer(self, model_name):
        assert explained(churn_frame(), model_name=model_name).report.explainer == "TreeExplainer"

    def test_a_linear_model_gets_the_linear_explainer(self):
        """``TreeExplainer`` refuses a logistic regression outright."""
        result = explained(churn_frame(), model_name="LogisticRegression")
        assert result.report.explainer == "LinearExplainer"

    def test_the_xgboost_wrapper_is_unwrapped_rather_than_worked_around(self):
        """SHAP rejects ``LabelEncodedClassifier``; it accepts what is inside it.

        This is why the wrapper needed no SHAP-specific code -- the finding that
        settled the design of this module.
        """
        result = explained(churn_frame(), model_name="XGBoost")
        assert result.report.explainer == "TreeExplainer"
        assert result.report.global_importance

    def test_an_unsupported_family_is_refused_with_a_reason(self):
        """Not silently skipped: a model nobody decided how to explain is a bug."""
        frame = churn_frame()
        recipe = build_preprocessor(frame, target="churned").transformer
        pipeline = build_pipeline(recipe, "classification", random_seed=0, estimator=GaussianNB())
        pipeline.fit(frame.drop(columns=["churned"]), frame["churned"])

        with pytest.raises(ExplainabilityError, match="GaussianNB"):
            explain_model(
                pipeline,
                frame,
                target="churned",
                task_type="classification",
                model_name="GaussianNB",
            )


class TestAdditivity:
    """If the contributions do not add up, nothing else in the report means anything."""

    @pytest.mark.parametrize("model_name", ALL_MODELS)
    def test_contributions_reconstruct_the_models_output(self, model_name):
        report = explained(churn_frame(), model_name=model_name).report
        assert report.additivity_max_error < 1e-3
        assert not [w for w in report.warnings if "reconstruct" in w]

    @pytest.mark.parametrize("model_name", ALL_MODELS)
    def test_each_local_explanation_adds_up_to_its_stated_output(self, model_name):
        """The waterfall's arithmetic, checked as a reader would check it."""
        report = explained(churn_frame(), model_name=model_name).report
        for example in report.examples:
            total = (
                example.base_value
                + sum(c.contribution for c in example.contributions)
                + example.other_contribution
            )
            assert total == pytest.approx(example.output_value, abs=1e-6)


class TestGlobalImportance:
    def test_the_column_the_data_was_built_around_comes_top(self):
        report = explained(churn_frame(), model_name="RandomForest").report
        assert report.global_importance[0].feature == "support_calls"

    def test_a_one_hot_column_is_summed_back_into_one_row(self):
        """``city`` appears once, carrying the influence of all three levels.

        Summed rather than averaged: a wide column acts through every one of its
        outputs at once, and averaging would penalise it for being expressive.
        """
        report = explained(churn_frame(), model_name="RandomForest").report
        city = next(item for item in report.global_importance if item.feature == "city")
        assert sorted(city.encoded_features) == ["city_Bristol", "city_Leeds", "city_London"]
        assert [item.feature for item in report.global_importance].count("city") == 1

    def test_shares_are_of_the_whole_and_never_exceed_it(self):
        report = explained(churn_frame(), model_name="RandomForest").report
        assert sum(item.share for item in report.global_importance) <= 1.0 + 1e-9

    def test_direction_is_stated_only_where_it_is_defined(self):
        """A one-hot column has no "higher value", so it gets no direction."""
        report = explained(churn_frame(), model_name="RandomForest").report
        by_name = {item.feature: item for item in report.global_importance}
        assert by_name["city"].direction == ""
        assert "push" in by_name["support_calls"].direction


class TestClassLabelling:
    """The class axis is in the model's encoded order -- index k means classes_[k].

    And on a binary target the axis may have *one* column rather than two: a
    gradient-booster and a logistic regression both emit a single margin, towards
    the positive class, regardless of which way a given row came out. Labelling
    that column ``classes_[0]`` reverses the meaning of every chart drawn from it
    and nothing about the numbers looks wrong -- which is why it is tested rather
    than reasoned about.
    """

    @pytest.mark.parametrize("model_name", ALL_MODELS)
    def test_the_probability_is_confidence_in_the_answer_given(self, model_name):
        """Read by label, not by SHAP output index -- the two differ on binary.

        Off-by-one here would report the probability of a class the row was not
        predicted to be, alongside the label it was.
        """
        frame = churn_frame(classes=["gold", "silver", "bronze"])
        report = explained(frame, model_name=model_name).report

        assert len(report.classes) == len(set(frame["churned"]))
        assert report.examples
        for example in report.examples:
            assert example.probability is not None
            # The model's own answer is, by definition, its most likely one.
            assert example.probability >= 1.0 / len(report.classes)

    @pytest.mark.parametrize("model_name", ALL_MODELS)
    def test_a_binary_row_reports_confidence_in_its_own_label(self, model_name):
        """The case a three-class test cannot reach: one output, two classes."""
        report = explained(churn_frame(), model_name=model_name).report
        assert {e.predicted for e in report.examples} == {"yes", "no"}
        for example in report.examples:
            assert example.probability is not None
            assert example.probability > 0.5

    @pytest.mark.parametrize("model_name", ALL_MODELS)
    def test_the_explained_direction_is_named_and_is_a_real_class(self, model_name):
        report = explained(churn_frame(), model_name=model_name).report
        for example in report.examples:
            assert example.explained_class in report.classes

    def test_a_single_output_model_explains_the_positive_class(self):
        """LightGBM's binary SHAP output is one column: the margin towards ``yes``.

        ``no`` is ``classes_[0]``, so an implementation that labelled the column
        by its index would say every contribution pushes towards ``no`` -- exactly
        backwards.
        """
        report = explained(churn_frame(), model_name="LightGBM").report
        assert {e.explained_class for e in report.examples} == {"yes"}

    def test_a_per_class_model_explains_the_class_it_predicted(self):
        """A random forest emits one column per class, so the index is the label."""
        report = explained(churn_frame(), model_name="RandomForest").report
        for example in report.examples:
            assert example.explained_class == example.predicted

    def test_importance_is_aggregated_across_classes_and_says_so(self):
        report = explained(
            churn_frame(classes=["gold", "silver", "bronze"]), model_name="RandomForest"
        ).report
        assert "across the 3 classes" in report.aggregation


class TestLocalExplanations:
    def test_one_row_is_explained_per_class(self):
        report = explained(churn_frame(), model_name="RandomForest").report
        assert len({e.row_label for e in report.examples}) == len(report.examples)
        assert {e.predicted for e in report.examples} == {"yes", "no"}

    def test_contributions_are_in_source_columns_not_encoded_ones(self):
        report = explained(churn_frame(), model_name="RandomForest").report
        named = {c.feature for e in report.examples for c in e.contributions}
        assert "city" in named
        assert not [name for name in named if name.startswith("city_")]

    def test_the_row_label_is_the_users_own_index(self):
        """A position into a sample is not a row number the user can look up."""
        frame = churn_frame()
        report = explained(frame, model_name="RandomForest").report
        for example in report.examples:
            assert int(example.row_label) in frame.index

    def test_the_shown_value_is_the_rows_real_value(self):
        frame = churn_frame()
        report = explained(frame, model_name="RandomForest").report
        example = report.examples[0]
        row = frame.loc[int(example.row_label)]
        for contribution in example.contributions:
            if contribution.feature == "city":
                assert contribution.value == row["city"]


class TestRegression:
    def test_a_regressor_is_explained_in_the_targets_own_units(self):
        """No classes, no probabilities -- and the output is the prediction itself."""
        frame = churn_frame().drop(columns=["churned"])
        pipeline = fitted(
            frame, model_name="RandomForest", target="tenure_months", task_type="regression"
        )
        result = explain_model(
            pipeline,
            frame,
            target="tenure_months",
            task_type="regression",
            model_name="RandomForest",
        )
        report = result.report
        assert report.classes == []
        assert report.additivity_max_error < 1e-3
        for example in report.examples:
            assert example.probability is None
            assert float(example.predicted) == pytest.approx(example.output_value, abs=1e-6)


class TestCharts:
    def test_the_shap_charts_are_real_pngs(self):
        result = explained(churn_frame(), model_name="RandomForest")
        assert {"shap_importance.png", "shap_summary.png"} <= {c.name for c in result.charts}
        for chart in result.charts:
            assert chart.png.startswith(b"\x89PNG"), chart.name

    def test_the_report_indexes_the_charts_it_produced(self):
        result = explained(churn_frame(), model_name="RandomForest")
        assert result.report.plots == [chart.name for chart in result.charts]


class TestSampling:
    def test_a_large_dataset_is_sampled_and_the_report_says_so(self):
        frame = churn_frame(400)
        pipeline = fitted(frame, model_name="RandomForest")
        report = explain_model(
            pipeline,
            frame,
            target="churned",
            task_type="classification",
            model_name="RandomForest",
            max_rows=100,
        ).report

        assert report.n_rows_explained == 100
        assert "sample of 100 rows" in report.sampling_note

    def test_a_small_dataset_is_explained_whole_and_says_nothing(self):
        report = explained(churn_frame(60), model_name="RandomForest").report
        assert report.n_rows_explained == 60
        assert report.sampling_note == ""
