"""What a run gets chunked into (spec 7.13).

The claim being defended is that these are *semantic units*, not fixed-size
windows: one passage per thing a person might ask about. The build plan's own
example question is "why was transaction amount important?", and these tests
check that such a question has a passage to land on -- one that names the feature
and is self-contained enough to be read on its own.
"""

from __future__ import annotations

from app.ml.chunking import build_chunks
from app.ml.contracts import (
    CleaningReport,
    ClusteringReport,
    ClusterProfile,
    CriticFinding,
    CriticReport,
    EvaluationReport,
    ExplainabilityReport,
    FeatureImportance,
    MetricSummary,
    NarrativeReport,
)


def explainability_of(*features: tuple[str, float, str]) -> ExplainabilityReport:
    return ExplainabilityReport(
        model_name="RandomForest",
        task_type="classification",
        target_column="churn",
        explainer="TreeExplainer",
        n_rows_explained=500,
        n_encoded_features=11,
        aggregation="Mean absolute SHAP value per feature",
        global_importance=[
            FeatureImportance(feature=name, importance=share, share=share, direction=direction)
            for name, share, direction in features
        ],
    )


class TestOnePassagePerSubject:
    def test_each_feature_gets_its_own_passage(self):
        """So "why was X important?" has something to retrieve."""
        chunks = build_chunks(
            explainability=explainability_of(
                ("transaction_amount", 0.4, "higher values push the prediction up"),
                ("age", 0.3, ""),
            )
        )
        headings = [chunk.heading for chunk in chunks]
        assert "Why transaction_amount matters to the model" in headings
        assert "Why age matters to the model" in headings

    def test_a_feature_passage_names_its_own_subject(self):
        """A passage that does not say what it is about cannot be retrieved by it."""
        chunks = build_chunks(explainability=explainability_of(("transaction_amount", 0.4, "")))
        passage = next(c for c in chunks if "transaction_amount" in c.heading)
        assert "transaction_amount" in passage.content
        # The embedded text is heading plus body, so the subject is in both.
        assert "transaction_amount" in passage.embedding_text()

    def test_the_direction_is_used_verbatim_when_there_is_one(self):
        """One wording for this claim across the report, the page and the chat."""
        chunks = build_chunks(
            explainability=explainability_of(
                ("support_calls", 0.4, "higher values push the prediction up")
            )
        )
        passage = next(c for c in chunks if "support_calls" in c.heading)
        assert "higher values push the prediction up" in passage.content

    def test_no_direction_is_claimed_when_the_artifact_gives_none(self):
        """Empty means the effect is not one-directional -- inventing one would lie."""
        chunks = build_chunks(explainability=explainability_of(("city", 0.2, "")))
        passage = next(c for c in chunks if "city" in c.heading)
        assert "push" not in passage.content

    def test_each_cluster_gets_its_own_passage(self):
        chunks = build_chunks(
            clustering=ClusteringReport(
                method="kmeans",
                k=2,
                silhouette=0.51,
                profiles=[
                    ClusterProfile(cluster=0, size=100, share=0.5, description="High spenders."),
                    ClusterProfile(cluster=1, size=100, share=0.5, description="Low spenders."),
                ],
            )
        )
        assert sum("Group" in chunk.heading for chunk in chunks) == 2

    def test_each_finding_gets_its_own_passage(self):
        chunks = build_chunks(
            critic=CriticReport(
                verdict="Mixed.",
                findings=[
                    CriticFinding(
                        area="modelling", severity="blocker", finding="A", recommendation="X"
                    ),
                    CriticFinding(
                        area="features", severity="note", finding="B", recommendation="Y"
                    ),
                ],
            )
        )
        assert sum("the review raised" in chunk.heading for chunk in chunks) == 2


class TestWhatIsDeliberatelyNotChunked:
    def test_per_column_statistics_are_not_indexed(self):
        """Arithmetic is the pandas tool's job -- see the module docstring.

        Indexing hundreds of passages of means and medians would flood retrieval
        with numbers that the other tool answers exactly.
        """
        chunks = build_chunks(
            cleaning=CleaningReport(
                n_rows_before=520, n_rows_after=500, n_columns_before=9, n_columns_after=8
            )
        )
        # One passage about cleaning as a whole, not one per column.
        assert len(chunks) == 1
        assert "cleaning" in chunks[0].heading.lower()


class TestTolerance:
    def test_a_run_with_nothing_produces_no_chunks(self):
        assert build_chunks() == []

    def test_a_partial_run_still_chunks_what_it_has(self):
        """A run whose EDA failed still has a model to ask about."""
        chunks = build_chunks(explainability=explainability_of(("age", 0.5, "")))
        assert chunks
        assert all(c.source == "explainability_report.json" for c in chunks)

    def test_an_empty_narrative_contributes_nothing(self):
        """No key means no prose -- and an empty passage answers nothing."""
        assert build_chunks(narrative=NarrativeReport()) == []

    def test_blank_content_is_never_stored(self):
        chunks = build_chunks(
            narrative=NarrativeReport(
                executive_summary="Something.", data_story="", model_story="   ", source="llm"
            )
        )
        assert len(chunks) == 1
        assert all(chunk.content.strip() for chunk in chunks)

    def test_ordinals_are_contiguous_and_stable(self):
        """Position is used to show passages in written order, not by score."""
        chunks = build_chunks(
            evaluation=EvaluationReport(
                task_type="classification",
                target_column="churn",
                model_name="RandomForest",
                n_folds=5,
                cv_strategy="StratifiedKFold",
                n_rows=500,
                n_features=8,
                metrics={"f1_macro": MetricSummary(mean=0.81, std=0.01)},
                primary_metric="f1_macro",
            )
        )
        assert [chunk.ordinal for chunk in chunks] == list(range(len(chunks)))


class TestTheNumbersComeFromTheArtifacts:
    def test_a_figure_in_a_passage_matches_the_report_it_came_from(self):
        """Code may state a figure from an artifact; a model may not invent one."""
        chunks = build_chunks(explainability=explainability_of(("support_calls", 0.375, "")))
        passage = next(c for c in chunks if "support_calls" in c.heading)
        assert "37.5%" in passage.content
