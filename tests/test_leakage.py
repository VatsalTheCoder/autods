"""The leakage tests -- the ones that matter more than any others here.

The build plan is blunt about it: the split-into-folds step is where the whole
project's credibility sits, and the failure mode is silent. Nothing crashes when
a pipeline is fitted over the entire dataset before splitting; you simply report a
score that is a lie. So these tests do not check that cross-validation *ran* --
they check the specific thing that could be wrong and would otherwise be
invisible.

Three independent proofs, deliberately overlapping:

1. **A spy counts the rows it was fitted on.** With 100 rows and 5 folds, a
   correctly-fitted pipeline fits five times on 80 rows each. A leaky one fits
   once, or fits on 100. The assertion is on the exact numbers, and the absence of
   100 from the log is asserted separately, because that is the number that would
   appear if anyone ever "optimised" the fold loop by fitting up front.
2. **The pipeline handed in is still unfitted afterwards.** Cross-validation
   clones; it must not fit the caller's object, or the second dataset run through
   the same recipe would inherit the first one's statistics.
3. **What preprocessing hands over has never seen data.** Asserted on the object
   ``build_preprocessor`` returns, which is spec 8's actual requirement, and on the
   pickle the pipeline stores -- so the claim is checkable from the artifact and
   not only from the code.

Everything here runs on synthetic frames with no database and no worker, so these
run in the ordinary test suite on every commit, which is the point.

**These assertions were checked by breaking the thing they guard** (2026-07-30).
A test defending a silent failure mode is worth exactly as much as its ability to
go red, and a passing test proves nothing about that on its own. Three leaks were
introduced into ``cross_validate_model`` in turn and reverted:

===============================  ==============  ===================================
Mutation                         Tests failing   Caught by
===============================  ==============  ===================================
``template.fit(X, y)`` before     5 of 15        proofs 1, 2 and 3
the fold loop -- the classic
leak the project exists to
rule out

``pipeline = template`` instead   2 of 15        proofs 2 and 3 **only**
of ``clone(template)``

``pipeline.fit(X, y)`` inside     3 of 15        proof 1 only
the loop rather than
``fit(X_train, y_train)``
===============================  ==============  ===================================

The middle row is the interesting one, and the reason the overlap in these proofs
is deliberate rather than wasteful. Dropping the ``clone`` is **not** caught by
the row-counting spy at all: each fold still refits on 80 training rows, so the
log looks perfect. It is caught only because the caller's template comes back
fitted and because the encoder's per-fold refit is checked separately. Any one of
these three proofs alone would leave a real leak undetected.
"""

from __future__ import annotations

import io

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted

from app.ml.contracts import ColumnStrategy, FeatureStrategy
from app.ml.modeling import cross_validate_model
from app.ml.preprocessing import build_preprocessor

N_ROWS = 100
N_FOLDS = 5
# 100 rows, 5 stratified folds, balanced classes: every fold trains on exactly 80.
EXPECTED_TRAIN_ROWS = 80


@pytest.fixture
def frame() -> pd.DataFrame:
    """A balanced 100-row classification frame, so fold sizes are exact."""
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "x1": rng.normal(0, 1, N_ROWS),
            "x2": rng.normal(5, 2, N_ROWS),
            "city": rng.choice(["London", "Leeds", "Bristol"], N_ROWS),
            # Exactly 50/50, alternating, so stratification splits cleanly.
            "churn": ["yes", "no"] * (N_ROWS // 2),
        }
    )


class FitLog:
    """A record of every ``fit`` call, shared across all of a spy's clones.

    ``sklearn.clone`` deep-copies an estimator's non-estimator constructor params.
    That is the right default -- it is part of why cloning per fold cannot leak
    state -- but it means a plain list handed to the spy would be duplicated for
    each fold and every clone would count into its own copy.

    Defining ``__deepcopy__`` to return ``self`` opts this one object out, so all
    five per-fold copies of the spy append to the same log and the run becomes
    observable from outside. Nothing in ``app/`` does this; it exists purely so a
    test can watch what fitting actually happened.
    """

    def __init__(self) -> None:
        self.rows: list[int] = []

    def __deepcopy__(self, memo) -> FitLog:
        return self

    def __len__(self) -> int:
        return len(self.rows)


class RowCountSpy(BaseEstimator, TransformerMixin):
    """A transformer that records how many rows each ``fit`` call saw."""

    def __init__(self, log: FitLog):
        self.log = log

    def fit(self, X, y=None):  # noqa: N803 - sklearn's parameter name
        self.log.rows.append(len(X))
        return self

    def transform(self, X):  # noqa: N803 - sklearn's parameter name
        return np.asarray(X, dtype=float)


class TestFittedInsideTheFold:
    """Proof 1: the pipeline is fitted per fold, on training rows only."""

    def test_fit_sees_only_training_rows(self, frame):
        log = FitLog()
        preprocessor = ColumnTransformer([("spy", RowCountSpy(log), ["x1", "x2"])])

        cross_validate_model(
            frame,
            target="churn",
            task_type="classification",
            preprocessor=preprocessor,
            cv_folds=N_FOLDS,
        )

        assert len(log) == N_FOLDS, f"expected one fit per fold, got {len(log)}"
        assert log.rows == [EXPECTED_TRAIN_ROWS] * N_FOLDS

    def test_fit_never_sees_the_whole_dataset(self, frame):
        """The single number whose presence would mean the scores are inflated."""
        log = FitLog()
        preprocessor = ColumnTransformer([("spy", RowCountSpy(log), ["x1", "x2"])])

        cross_validate_model(
            frame,
            target="churn",
            task_type="classification",
            preprocessor=preprocessor,
            cv_folds=N_FOLDS,
        )

        assert N_ROWS not in log.rows, (
            "the preprocessor was fitted on all 100 rows -- test-fold data leaked "
            "into training and every reported score is inflated"
        )

    def test_fold_row_counts_are_recorded_honestly(self, frame):
        """The artifact's own numbers must match what actually happened."""
        log = FitLog()
        preprocessor = ColumnTransformer([("spy", RowCountSpy(log), ["x1", "x2"])])

        result = cross_validate_model(
            frame,
            target="churn",
            task_type="classification",
            preprocessor=preprocessor,
            cv_folds=N_FOLDS,
        )

        assert [fold.n_train for fold in result.folds] == log.rows
        # Every row is held out exactly once across the folds.
        assert sum(fold.n_test for fold in result.folds) == N_ROWS


class TestTemplateStaysUnfitted:
    """Proof 2: cross-validation clones, and never fits what it was handed."""

    def test_the_pipeline_passed_in_is_not_fitted(self, frame):
        preprocessor = build_preprocessor(frame, target="churn").transformer

        result = cross_validate_model(
            frame,
            target="churn",
            task_type="classification",
            preprocessor=preprocessor,
            cv_folds=N_FOLDS,
        )

        # The caller's object.
        assert not hasattr(preprocessor, "transformers_")
        with pytest.raises(NotFittedError):
            check_is_fitted(preprocessor)

        # And the template the run assembled around it.
        with pytest.raises(NotFittedError):
            check_is_fitted(result.pipeline_template.named_steps["model"])

    def test_a_second_run_is_unaffected_by_the_first(self, frame):
        """A recipe reused on different data must not carry statistics across.

        This is the practical consequence of not fitting the template: the same
        unfitted preprocessor can be handed to two runs and the second one's
        scores owe nothing to the first one's data.
        """
        preprocessor = build_preprocessor(frame, target="churn").transformer

        first = cross_validate_model(
            frame,
            target="churn",
            task_type="classification",
            preprocessor=preprocessor,
            cv_folds=N_FOLDS,
        )
        second = cross_validate_model(
            frame,
            target="churn",
            task_type="classification",
            preprocessor=preprocessor,
            cv_folds=N_FOLDS,
        )

        assert [f.metrics for f in first.folds] == [f.metrics for f in second.folds]


class TestRecipeLeavesPreprocessingUnfitted:
    """Proof 3: spec 8's literal requirement -- an unfitted pipeline is handed over."""

    def test_build_preprocessor_returns_something_never_fitted(self, frame):
        result = build_preprocessor(frame, target="churn")

        with pytest.raises(NotFittedError):
            check_is_fitted(result.transformer)
        assert not hasattr(result.transformer, "transformers_")

    def test_inner_steps_are_unfitted_too(self, frame):
        """An unfitted outer object wrapping fitted inner ones would still leak."""
        transformer = build_preprocessor(frame, target="churn").transformer

        for _, step, _ in transformer.transformers:
            for _, inner in step.steps:
                with pytest.raises(NotFittedError):
                    check_is_fitted(inner)

    def test_the_section_7_steps_are_unfitted_too(self):
        """The new transformers are where an unfitted recipe could quietly regress.

        ``FrequencyEncoder`` learns a distribution and ``SelectKBest`` learns
        which columns beat the rest -- both are fitted state, and both are new in
        this section. Naming them here means a future change that computed either
        one up front fails a test rather than silently inflating every score.
        """
        frame = pd.DataFrame(
            {
                "note": [f"n{i}" for i in range(N_ROWS)],
                "signed_up": pd.to_datetime(["2024-01-01"] * N_ROWS),
                "x1": np.arange(float(N_ROWS)),
                "churn": ["yes", "no"] * (N_ROWS // 2),
            }
        )
        pipeline = build_preprocessor(frame, target="churn", select_k=2).transformer

        with pytest.raises(NotFittedError):
            check_is_fitted(pipeline.named_steps["select"])
        for _, step, _ in pipeline.named_steps["columns"].transformers:
            for _, inner in step.steps:
                with pytest.raises(NotFittedError):
                    check_is_fitted(inner)


class TestFrequenciesComeFromTheFoldOnly:
    """The Section 7 step most able to leak, checked end to end.

    A frequency map is the kind of feature that looks like good engineering and
    leaks completely: build it over the whole dataset and every training row
    carries a summary of the test fold. The encoder is a pipeline step precisely
    so this cannot happen, and this is the test that says so.
    """

    def test_the_encoder_is_refitted_for_every_fold(self):
        frame = pd.DataFrame(
            {
                "city": (["London"] * 60) + (["Leeds"] * 40),
                "x1": np.arange(float(N_ROWS)),
                "churn": ["yes", "no"] * (N_ROWS // 2),
            }
        )
        strategy = FeatureStrategy(
            columns=[
                ColumnStrategy(
                    column="city", role="text", impute="most_frequent", encode="frequency"
                ),
                ColumnStrategy(column="x1", role="numeric", impute="median", scale="standard"),
            ],
            source="llm",
        )
        preprocessor = build_preprocessor(frame, target="churn", strategy=strategy).transformer

        result = cross_validate_model(
            frame,
            target="churn",
            task_type="classification",
            preprocessor=preprocessor,
            cv_folds=N_FOLDS,
        )

        # The template is what proves it: had the frequencies been computed once,
        # up front, they would be sitting on this object now.
        assert len(result.folds) == N_FOLDS
        with pytest.raises(NotFittedError):
            check_is_fitted(preprocessor)

    def test_survives_a_pickle_round_trip_still_unfitted(self, frame):
        """The stored artifact is checkable, not just the in-memory object.

        ``preprocessing_pipeline.pkl`` is what the pipeline writes to S3. Loading
        it back and finding it unfitted is how the methodology claim can be
        verified from outside the codebase.
        """
        transformer = build_preprocessor(frame, target="churn").transformer

        buffer = io.BytesIO()
        joblib.dump(transformer, buffer)
        buffer.seek(0)
        restored = joblib.load(buffer)

        with pytest.raises(NotFittedError):
            check_is_fitted(restored)


class TestClusterLabelsNeverBecomeFeatures:
    """Section 6's guardrail, and it is the same failure as the rest of this file.

    Cluster labels are computed over *every* row, including the rows that later
    land in a test fold. Adding them to the dataset as a feature would hand the
    model a summary of the held-out data -- leakage by a different route, and one
    that looks like a feature-engineering win rather than a mistake (spec 9).

    So clustering hands its labels back as a bare array rather than a column, and
    these tests pin down that using them as a feature stays a deliberate act
    instead of an accident waiting to happen.
    """

    def test_clustering_does_not_touch_the_dataframe(self, frame):
        from app.ml.clustering import run_clustering

        before = frame.copy()
        run_clustering(frame, target="churn")

        pd.testing.assert_frame_equal(frame, before)

    def test_labels_come_back_outside_the_data(self, frame):
        """An array, not a column -- the shape of the API is the safeguard."""
        from app.ml.clustering import run_clustering

        result = run_clustering(frame, target="churn")

        assert result.labels is None or isinstance(result.labels, np.ndarray)
        assert "cluster" not in frame.columns

    def test_no_cluster_column_reaches_the_preprocessor(self, frame):
        """The end of the chain: whatever EDA did, the model's features are clean."""
        from app.ml.clustering import run_clustering

        run_clustering(frame, target="churn")
        spec = build_preprocessor(frame, target="churn").spec

        routed = spec.numeric_columns + spec.categorical_columns
        assert not [name for name in routed if "cluster" in name.lower()]

    def test_the_eda_node_rejects_a_frame_it_modified(self, monkeypatch):
        """The check is enforced in code, not left to convention.

        If a future change ever appended cluster labels to the frame, the EDA node
        raises rather than quietly passing a contaminated dataset to the model.
        This drives that path by making clustering misbehave on purpose.
        """
        import app.worker.graph as graph
        from app.ml.clustering import ClusteringResult
        from app.ml.contracts import ClusteringReport

        state_frame = pd.DataFrame({"x1": np.arange(20.0), "churn": ["yes", "no"] * 10})

        def sabotage(frame, **_kwargs):
            # Exactly the mistake the guardrail exists to catch.
            frame["cluster"] = 0
            return ClusteringResult(
                report=ClusteringReport(method="kmeans", k=2, silhouette=0.5), labels=None
            )

        monkeypatch.setattr(graph, "run_clustering", sabotage)
        monkeypatch.setattr(graph, "compute_statistics", lambda *a, **k: _stub_eda())
        monkeypatch.setattr(graph, "render_charts", lambda *a, **k: [])
        monkeypatch.setattr(graph, "describe_clusters", lambda profiles, **k: profiles)
        monkeypatch.setattr(graph, "register_json_artifact", lambda *a, **k: None)
        monkeypatch.setattr(graph, "register_bytes_artifact", lambda *a, **k: None)
        monkeypatch.setattr(graph, "make_usage_recorder", lambda *a, **k: lambda _r: None)
        monkeypatch.setattr(graph, "get_optional_llm", lambda: None)
        monkeypatch.setattr(graph, "SessionLocal", _NullSession)

        with pytest.raises(RuntimeError, match="never become model features"):
            graph.eda_node(
                {
                    "job_id": 1,
                    "cleaned": state_frame,
                    "target": "churn",
                    "task_type": "classification",
                    "plan": None,
                }
            )


def _stub_eda():
    from app.ml.contracts import EdaReport

    return EdaReport(n_rows=20, n_columns=2, target_column="churn")


class _NullSession:
    """A stand-in for SessionLocal so the node's persistence is a no-op here."""

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def commit(self):
        pass


class TestFoldSafety:
    """Per-fold fitting must survive the data it will actually meet."""

    def test_a_category_only_in_the_test_fold_does_not_crash(self):
        """A rare category is unknown to a correctly-fitted encoder -- by design.

        This is the failure that tempts people back into encoding the whole
        dataset up front. ``handle_unknown="ignore"`` is the correct fix; fitting
        the encoder on everything is not.
        """
        rng = np.random.default_rng(1)
        frame = pd.DataFrame(
            {
                "x1": rng.normal(0, 1, N_ROWS),
                "city": ["London"] * 50 + ["Leeds"] * 49 + ["Nowhere"],
                "churn": ["yes", "no"] * (N_ROWS // 2),
            }
        )
        preprocessor = build_preprocessor(frame, target="churn").transformer

        result = cross_validate_model(
            frame,
            target="churn",
            task_type="classification",
            preprocessor=preprocessor,
            cv_folds=N_FOLDS,
        )

        assert len(result.folds) == N_FOLDS
        assert all(fold.metrics for fold in result.folds)
