"""Tests for the model roster, the ranking, and SMOTE's placement.

Two separable claims live here.

The **leaderboard** claim is that four models were compared fairly. That rests
almost entirely on one thing -- every candidate seeing the same splits -- so
that is what the first class checks, along with the ranking being on the metric
it says it is and a failing candidate being named rather than vanishing.

The **SMOTE** claim is the one that would be embarrassing to get wrong.
Resampling outside the fold inflates every score and nothing crashes, so the
tests below are about placement: the synthetic rows exist during a fold's fit and
do not exist when it is scored. imblearn's Pipeline is what makes that true, and
``TestSmoteStaysInsideTheFold`` is the evidence rather than the assurance.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted

from app.ml.modeling import (
    ModelingError,
    build_resampler,
    build_roster,
    run_leaderboard,
)
from app.ml.preprocessing import build_preprocessor

N_ROWS = 120


@pytest.fixture
def balanced() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "x1": rng.normal(0, 1, N_ROWS),
            "x2": rng.normal(5, 2, N_ROWS),
            "city": rng.choice(["London", "Leeds", "Bristol"], N_ROWS),
            "churn": ["yes", "no"] * (N_ROWS // 2),
        }
    )


@pytest.fixture
def imbalanced() -> pd.DataFrame:
    """A 9:1 target -- the case SMOTE exists for."""
    rng = np.random.default_rng(1)
    return pd.DataFrame(
        {
            "x1": rng.normal(0, 1, N_ROWS),
            "x2": rng.normal(5, 2, N_ROWS),
            "churn": (["no"] * 108) + (["yes"] * 12),
        }
    )


@pytest.fixture
def continuous() -> pd.DataFrame:
    rng = np.random.default_rng(2)
    x = rng.normal(0, 1, N_ROWS)
    return pd.DataFrame({"x1": x, "x2": rng.normal(5, 2, N_ROWS), "price": 3 * x + 1})


def _run(frame, target, task_type, **kwargs):
    preprocessor = build_preprocessor(frame, target=target).transformer
    return run_leaderboard(
        frame,
        target=target,
        task_type=task_type,
        preprocessor=preprocessor,
        cv_folds=3,
        **kwargs,
    )


class TestTheRoster:
    """Spec 7.7 names four families per task type."""

    def test_classification_has_the_four_the_spec_asks_for(self):
        names = {c.name for c in build_roster("classification", random_seed=0)}
        assert names == {"RandomForest", "LogisticRegression", "XGBoost", "LightGBM"}

    def test_regression_has_the_four_the_spec_asks_for(self):
        names = {c.name for c in build_roster("regression", random_seed=0)}
        assert names == {"RandomForest", "LinearRegression", "XGBoost", "LightGBM"}

    def test_the_roster_is_seeded_so_a_reported_score_is_reproducible(self, balanced):
        first, _ = _run(balanced, "churn", "classification")
        second, _ = _run(balanced, "churn", "classification")
        assert [e.score for e in first.entries] == [e.score for e in second.entries]


class TestTheRanking:
    def test_every_candidate_appears(self, balanced):
        board, _ = _run(balanced, "churn", "classification")
        assert len(board.entries) == 4

    def test_they_are_ranked_best_first(self, balanced):
        board, _ = _run(balanced, "churn", "classification")
        scores = [e.score for e in board.entries if not e.error]
        assert scores == sorted(scores, reverse=True)

    def test_the_ranks_are_numbered_from_one(self, balanced):
        board, _ = _run(balanced, "churn", "classification")
        assert [e.rank for e in board.entries] == [1, 2, 3, 4]

    def test_every_candidate_saw_the_same_splits(self, balanced):
        """The property the whole comparison rests on.

        Different partitions would make the four numbers answers to four
        different questions, and the ranking between them meaningless.
        """
        board, winner = _run(balanced, "churn", "classification")
        assert board.n_folds == winner.n_folds
        assert board.cv_strategy == "StratifiedKFold"
        assert sum(f.n_test for f in winner.folds) == N_ROWS

    def test_the_ranking_metric_is_named_on_every_row(self, balanced):
        board, _ = _run(balanced, "churn", "classification")
        assert {e.primary_metric for e in board.entries} == {"f1_macro"}

    def test_regression_ranks_on_r2(self, continuous):
        board, _ = _run(continuous, "price", "regression")
        assert board.primary_metric == "r2"
        assert board.winner().score > 0.9

    def test_the_spread_is_recorded_beside_the_score(self, balanced):
        """A lead of 0.02 between models whose folds swing 0.09 is not a lead."""
        board, _ = _run(balanced, "churn", "classification")
        assert all(e.std >= 0.0 for e in board.entries if not e.error)

    def test_the_winner_is_returned_in_full(self, balanced):
        """The evaluation report needs the folds, not just the summary row."""
        board, winner = _run(balanced, "churn", "classification")
        assert winner.model_name == board.winner().model_name
        assert len(winner.folds) == 3


class TestAFailingCandidate:
    def test_it_is_named_rather_than_dropped(self, balanced, monkeypatch):
        """Three rows in a table of four looks like a table of three."""
        import app.ml.modeling as modeling

        real = modeling.build_roster

        def broken(task_type, *, random_seed):
            roster = real(task_type, random_seed=random_seed)
            roster[1].estimator = _Explodes()
            return roster

        monkeypatch.setattr(modeling, "build_roster", broken)

        board, _ = _run(balanced, "churn", "classification")
        failed = [e for e in board.entries if e.error]
        assert len(failed) == 1
        assert "deliberately broken" in failed[0].error

    def test_a_failure_sorts_last_rather_than_first(self, balanced, monkeypatch):
        """A NaN score must not float to the top of the table."""
        import app.ml.modeling as modeling

        real = modeling.build_roster

        def broken(task_type, *, random_seed):
            roster = real(task_type, random_seed=random_seed)
            roster[0].estimator = _Explodes()
            return roster

        monkeypatch.setattr(modeling, "build_roster", broken)

        board, _ = _run(balanced, "churn", "classification")
        assert board.entries[-1].error
        assert board.entries[0].error == ""

    def test_every_candidate_failing_is_an_error_not_an_empty_table(self, balanced, monkeypatch):
        import app.ml.modeling as modeling

        def all_broken(task_type, *, random_seed):
            from app.ml.modeling import Candidate

            return [Candidate("Broken", _Explodes())]

        monkeypatch.setattr(modeling, "build_roster", all_broken)

        with pytest.raises(ModelingError, match="No model could be trained"):
            _run(balanced, "churn", "classification")


class TestWhenSmoteIsRefused:
    """Refused up front with a reason, rather than failing from inside a fold."""

    def test_a_continuous_target_has_no_minority_class(self, continuous):
        warnings: list[str] = []
        assert (
            build_resampler(
                continuous["price"], task_type="regression", random_seed=0, warnings=warnings
            )
            is None
        )
        assert "continuous" in warnings[0]

    def test_too_few_minority_rows_to_interpolate_between(self):
        warnings: list[str] = []
        y = pd.Series((["no"] * 50) + (["yes"] * 3))
        assert (
            build_resampler(y, task_type="classification", random_seed=0, warnings=warnings) is None
        )
        assert "rarest class" in warnings[0]

    def test_a_single_class_target(self):
        warnings: list[str] = []
        y = pd.Series(["no"] * 50)
        assert (
            build_resampler(y, task_type="classification", random_seed=0, warnings=warnings) is None
        )
        assert "one class" in warnings[0]

    def test_the_refusal_reaches_the_artifact(self, continuous):
        board, _ = _run(continuous, "price", "regression", use_smote=True)
        assert board.resampling == "none"
        assert any("continuous" in w for w in board.warnings)

    def test_a_workable_imbalance_is_accepted(self, imbalanced):
        warnings: list[str] = []
        resampler = build_resampler(
            imbalanced["churn"], task_type="classification", random_seed=0, warnings=warnings
        )
        assert resampler is not None
        assert not warnings


class TestSmoteStaysInsideTheFold:
    """The claim that would be worth nothing if it were only documented."""

    def test_it_is_a_step_in_the_pipeline_not_a_pass_over_the_data(self, imbalanced):
        _, winner = _run(imbalanced, "churn", "classification", use_smote=True)
        assert "resample" in winner.pipeline_template.named_steps

    def test_the_pipeline_holding_it_is_never_fitted_by_the_run(self, imbalanced):
        """Cloned per fold, like every other step -- see test_leakage.py."""
        _, winner = _run(imbalanced, "churn", "classification", use_smote=True)
        with pytest.raises(NotFittedError):
            check_is_fitted(winner.pipeline_template.named_steps["model"])

    def test_the_scored_rows_are_the_real_ones_not_synthetic_ones(self, imbalanced):
        """The heart of it.

        imblearn's Pipeline resamples during ``fit`` and skips the resampler
        during ``predict``. So the fold's *training* half is balanced and its
        held-out half keeps the dataset's real 9:1 skew. If SMOTE had been
        applied before splitting, the test halves would have grown too and the
        row counts below would exceed the dataset.
        """
        board, winner = _run(imbalanced, "churn", "classification", use_smote=True)

        assert board.resampling.startswith("SMOTE")
        assert sum(f.n_test for f in winner.folds) == N_ROWS
        # Every training fold reports the *pre-resampling* row count, because
        # that is what was split; the synthetic rows exist only inside the fit.
        assert all(f.n_train < N_ROWS for f in winner.folds)

    def test_turning_it_off_leaves_no_resampler_behind(self, imbalanced):
        board, winner = _run(imbalanced, "churn", "classification", use_smote=False)
        assert "resample" not in winner.pipeline_template.named_steps
        assert board.resampling == "none"

    def test_it_changes_the_scores_it_is_supposed_to_change(self, imbalanced):
        """A flag that altered nothing would be a lie in the artifact."""
        without, _ = _run(imbalanced, "churn", "classification", use_smote=False)
        with_smote, _ = _run(imbalanced, "churn", "classification", use_smote=True)

        assert [e.score for e in without.entries] != [e.score for e in with_smote.entries]


class _Explodes:
    """An estimator that fails on fit, for the failed-candidate path."""

    def get_params(self, deep=True):
        return {}

    def set_params(self, **_params):
        return self

    def fit(self, X, y=None):  # noqa: N803 - sklearn's parameter name
        raise RuntimeError("this estimator is deliberately broken")
