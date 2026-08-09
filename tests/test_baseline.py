"""The featureless baseline: the row that says what the other scores are worth.

R² 0.07 on listing prices reads as a total failure. Macro F1 0.62 reads as a
respectable result. Neither impression survives contact with what a model that
ignores every feature scores on the same folds -- the first is a real
improvement over 0.00, and the second can be *worse* than always answering with
the commonest class.

So the baseline is scored over the same splits as everything else and ranked
alongside it. Two things must hold for that to be safe: it is never the model
that gets served (``TestItIsNeverTheServedModel``), and when it wins, the run
says so in words rather than leaving a reader to subtract two numbers.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.ml.modeling import build_roster, run_leaderboard
from app.ml.preprocessing import build_preprocessor

N_ROWS = 200


def _run(frame: pd.DataFrame, target: str, task_type: str):
    return run_leaderboard(
        frame,
        target=target,
        task_type=task_type,
        preprocessor=build_preprocessor(frame, target=target, task_type=task_type).transformer,
        cv_folds=3,
    )


@pytest.fixture
def learnable() -> pd.DataFrame:
    """A target the features genuinely explain."""
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, N_ROWS)
    return pd.DataFrame({"x": x, "y": np.where(x + rng.normal(0, 0.3, N_ROWS) > 0, "yes", "no")})


@pytest.fixture
def noise() -> pd.DataFrame:
    """Features with nothing whatsoever to say about the target.

    Deliberately larger than the other fixtures. On a couple of hundred rows a
    linear model fits two pure-noise features well enough to post a positive
    held-out R² by luck, which is a real small-sample effect and not what these
    tests are about.
    """
    rng = np.random.default_rng(1)
    n = 400
    return pd.DataFrame(
        {
            "x1": rng.normal(0, 1, n),
            "x2": rng.normal(0, 1, n),
            "y": rng.normal(100, 20, n),
        }
    )


class TestItAppearsAndIsLabelled:
    def test_the_board_carries_a_baseline_row(self, learnable):
        board, _ = _run(learnable, "y", "classification")
        baselines = [e for e in board.entries if e.is_baseline]
        assert len(baselines) == 1
        assert "Baseline" in baselines[0].model_name

    def test_it_is_scored_on_the_same_metric_as_everything_else(self, learnable):
        board, _ = _run(learnable, "y", "classification")
        assert {e.primary_metric for e in board.entries} == {"f1_macro"}

    def test_the_regression_baseline_predicts_the_median(self, noise):
        """Not the mean: R² is defined against the mean, so a mean-predictor
        scores 0.000 by construction and tells the reader nothing."""
        names = [c.name for c in build_roster("regression", random_seed=0) if c.is_baseline]
        assert names == ["Baseline (always the median)"]
        board, _ = _run(noise, "y", "regression")
        baseline = next(e for e in board.entries if e.is_baseline)
        assert baseline.score < 0.01


class TestItIsNeverTheServedModel:
    def test_the_winner_is_a_real_model_even_when_the_baseline_ranks_first(self, noise):
        """On pure noise the baseline usually tops the board. Serving it would
        hand SHAP an estimator that never looked at a feature."""
        board, winner = _run(noise, "y", "regression")

        assert board.entries[0].is_baseline, "expected the baseline to win on noise"
        assert "Baseline" not in winner.model_name

    def test_the_winner_is_still_the_best_real_model(self, learnable):
        board, winner = _run(learnable, "y", "classification")
        best_real = next(e for e in board.entries if not e.is_baseline)
        assert winner.model_name == best_real.model_name


class TestTheVerdict:
    def test_a_baseline_nobody_beats_is_called_out(self, noise):
        board, _ = _run(noise, "y", "regression")
        assert any("No model beat the featureless baseline" in w for w in board.warnings), (
            board.warnings
        )

    def test_a_comfortably_beaten_baseline_says_nothing(self, learnable):
        board, _ = _run(learnable, "y", "classification")
        assert not any("baseline" in w.lower() for w in board.warnings), board.warnings

    @staticmethod
    def _verdict(best_score: float, best_std: float, baseline_score: float) -> list[str]:
        """Drive the rule directly, rather than hunting for a seed that lands on it.

        The band this tests is narrow by definition -- a lead smaller than the
        winner's own fold spread -- and a fixture that happens to fall inside it
        for one scikit-learn version will fall outside it for the next.
        """
        from app.ml.contracts import LeaderboardEntry
        from app.ml.modeling import _baseline_verdict

        rows = [
            (
                LeaderboardEntry(
                    rank=1,
                    model_name="LightGBM",
                    primary_metric="f1_macro",
                    score=best_score,
                    std=best_std,
                ),
                None,
                False,
            ),
            (
                LeaderboardEntry(
                    rank=2,
                    model_name="Baseline (most frequent class)",
                    primary_metric="f1_macro",
                    score=baseline_score,
                    std=0.01,
                    is_baseline=True,
                ),
                None,
                True,
            ),
        ]
        return _baseline_verdict(rows, primary="f1_macro")

    def test_a_lead_inside_the_fold_spread_is_called_unproven(self):
        """A win smaller than the winner's own fold-to-fold noise is not a win."""
        warnings = self._verdict(best_score=0.62, best_std=0.05, baseline_score=0.60)
        assert any("unproven" in w for w in warnings), warnings

    def test_a_lead_larger_than_the_spread_is_left_to_speak_for_itself(self):
        warnings = self._verdict(best_score=0.90, best_std=0.02, baseline_score=0.60)
        assert warnings == []

    def test_a_tie_counts_as_not_beaten(self):
        warnings = self._verdict(best_score=0.60, best_std=0.02, baseline_score=0.60)
        assert any("No model beat" in w for w in warnings), warnings
