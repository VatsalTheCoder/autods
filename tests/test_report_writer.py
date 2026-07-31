"""Tests for the report agent and the PDF (spec 7.12).

The design being defended: **the model writes sentences and code writes numbers.**
Section 5 built the report deterministically on the grounds that "a report that
hallucinates a number is worse than no report", and Section 9 adds a model
without giving that up -- by never asking it for a figure and never letting its
output carry one into the document unchecked.

So the tests here are mostly about what the narrative is *not* allowed to do to
the report: it cannot remove a metric, cannot replace a table, and its absence
must leave the Section 5 document intact rather than a document with holes in it.
"""

from __future__ import annotations

import json

from app.agents.report_writer import write_narrative
from app.agents.summaries import summarise_run
from app.core.llm.base import ModelTier, RateLimitError, TransientLLMError
from app.core.llm.fake import FakeLLM
from app.ml.contracts import (
    CleaningReport,
    CriticFinding,
    CriticReport,
    EvaluationReport,
    MetricSummary,
    NarrativeReport,
    PlannerPlan,
    PreprocessingSpec,
)
from app.ml.pdf import PdfError, render_pdf
from app.ml.report import build_markdown_report

# One scripted failure per tier. Derived from the enum rather than written as a
# literal so that adding a tier does not quietly turn these degradation tests
# into success tests: with too few failures scripted, FakeLLM falls through to
# its default reply and the agent succeeds, which is what happened when
# ModelTier.FALLBACK was added.
TIERS = len(ModelTier)


def evaluation_of() -> EvaluationReport:
    return EvaluationReport(
        task_type="classification",
        target_column="churn",
        model_name="RandomForest",
        n_folds=5,
        cv_strategy="StratifiedKFold",
        n_rows=500,
        n_features=8,
        metrics={"f1_macro": MetricSummary(mean=0.8123, std=0.0141)},
        primary_metric="f1_macro",
    )


def narrative_json(**overrides) -> str:
    payload = {
        "executive_summary": "The model predicts churn reasonably well.",
        "data_story": "Most customers stayed, and support calls vary widely.",
        "model_story": "A forest led a close field; the lead is not decisive.",
        "recommendations": ["Check the satisfaction scale."],
    }
    payload.update(overrides)
    return json.dumps(payload)


def report_with(narrative=None, critic=None) -> str:
    return build_markdown_report(
        filename="customers.csv",
        plan=PlannerPlan(source="llm"),
        cleaning=CleaningReport(
            n_rows_before=520, n_rows_after=500, n_columns_before=9, n_columns_after=8
        ),
        preprocessing=PreprocessingSpec(numeric_columns=["age"], strategy_source="llm"),
        evaluation=evaluation_of(),
        narrative=narrative,
        critic=critic,
    )


class TestTheAgent:
    def test_prose_is_used_when_a_model_answers(self):
        narrative = write_narrative(
            summarise_run(budget_tokens=6000), client=FakeLLM([narrative_json()])
        )
        assert narrative.source == "llm"
        assert narrative.executive_summary
        assert narrative.recommendations

    def test_no_client_gives_an_empty_narrative_not_a_generated_one(self):
        """A stand-in would be this module restating its own tables in worse English."""
        narrative = write_narrative(summarise_run(budget_tokens=6000), client=None)
        assert narrative.source == "default"
        assert narrative.is_empty()

    def test_a_provider_failure_degrades_to_the_plain_report(self):
        """A malformed reply or an outage is not something a smaller model fixes."""
        narrative = write_narrative(
            summarise_run(budget_tokens=6000),
            client=FakeLLM([TransientLLMError("503 after retries")] * TIERS),
        )
        assert narrative.source == "default"
        assert narrative.is_empty()

    def test_a_rate_limit_steps_down_a_tier_rather_than_giving_up(self):
        """These two agents are why the large model's quota runs out first.

        Losing the report's prose for the rest of the day because one model's
        minute-quota was exhausted -- while another sits idle -- is the wrong
        trade, so a rate limit tries the smaller model before degrading.
        """
        client = FakeLLM([RateLimitError("429"), narrative_json()])
        narrative = write_narrative(summarise_run(budget_tokens=6000), client=client)
        assert narrative.source == "llm"
        assert narrative.executive_summary

    def test_a_rate_limit_on_every_tier_still_degrades(self):
        client = FakeLLM([RateLimitError("429")] * TIERS)
        narrative = write_narrative(summarise_run(budget_tokens=6000), client=client)
        assert narrative.source == "default"
        assert narrative.is_empty()

    def test_a_model_cannot_declare_its_own_source(self):
        narrative = write_narrative(
            summarise_run(budget_tokens=6000),
            client=FakeLLM([narrative_json(source="default")]),
        )
        assert narrative.source == "llm"

    def test_the_critics_findings_are_put_in_front_of_the_writer(self):
        """The summary must not read as though the concerns do not exist."""
        client = FakeLLM([narrative_json()])
        critic = CriticReport(
            verdict="One thing needs checking.",
            findings=[
                CriticFinding(
                    area="explainability",
                    severity="blocker",
                    finding="Satisfaction pushes the wrong way.",
                    recommendation="Audit the encoding.",
                )
            ],
        )
        write_narrative(summarise_run(budget_tokens=6000), critic=critic, client=client)

        # Across the whole conversation, not the last turn: structured output
        # appends its own schema hint after the prompt.
        sent = "\n".join(m.content for m in client.calls[-1].messages)
        assert "Satisfaction pushes the wrong way." in sent
        assert "One thing needs checking." in sent


class TestTheNarrativeCannotDamageTheReport:
    """The guarantee that lets an LLM near a report at all."""

    def test_the_numbers_are_unchanged_by_the_prose(self):
        plain = report_with()
        written = report_with(narrative=NarrativeReport(**json.loads(narrative_json())))

        for figure in ("0.8123", "0.0141", "5 folds", "StratifiedKFold"):
            assert figure in plain, figure
            assert figure in written, figure

    def test_every_deterministic_section_survives(self):
        written = report_with(narrative=NarrativeReport(**json.loads(narrative_json())))
        for heading in ("## Result", "## How this was validated", "## Data quality"):
            assert heading in written or heading.replace("## ", "") in written

    def test_an_empty_narrative_leaves_the_section_5_report(self):
        assert report_with(narrative=NarrativeReport()) == report_with()

    def test_the_prose_appears_where_a_reader_starts(self):
        written = report_with(narrative=NarrativeReport(**json.loads(narrative_json())))
        assert "## In short" in written
        # Above the metrics, which is the whole point of an executive summary.
        assert written.index("## In short") < written.index("0.8123")

    def test_recommendations_are_rendered(self):
        written = report_with(narrative=NarrativeReport(**json.loads(narrative_json())))
        assert "## What to do next" in written
        assert "Check the satisfaction scale." in written


class TestTheReviewInTheReport:
    def test_the_findings_are_rendered_worst_first(self):
        critic = CriticReport(
            verdict="Mixed.",
            findings=[
                CriticFinding(
                    area="features", severity="note", finding="Minor.", recommendation="Nothing."
                ),
                CriticFinding(
                    area="explainability",
                    severity="blocker",
                    finding="Serious.",
                    recommendation="Fix it.",
                ),
            ],
        )
        written = report_with(critic=critic)
        assert "## What the review found" in written
        assert written.index("Serious.") < written.index("Minor.")

    def test_measured_findings_are_marked_as_such(self):
        """A threshold result and an opinion are not worth the same to a reader."""
        critic = CriticReport(
            findings=[
                CriticFinding(
                    area="modelling",
                    severity="concern",
                    finding="Measured thing.",
                    recommendation="Act.",
                    measured=True,
                ),
                CriticFinding(
                    area="modelling",
                    severity="concern",
                    finding="Written thing.",
                    recommendation="Act.",
                ),
            ]
        )
        written = report_with(critic=critic)
        assert "Measured thing. *(measured)*" in written
        assert "Written thing. *(measured)*" not in written

    def test_what_the_review_could_not_see_is_printed(self):
        critic = CriticReport(
            verdict="Partial review.",
            omissions=["Showing 20 of 121 per-column feature decisions."],
        )
        written = report_with(critic=critic)
        assert "The review did not see everything" in written
        assert "20 of 121" in written

    def test_a_checklist_only_review_says_so(self):
        critic = CriticReport(
            verdict="Checks ran.",
            findings=[
                CriticFinding(
                    area="modelling",
                    severity="note",
                    finding="X",
                    recommendation="Y",
                    measured=True,
                )
            ],
            source="default",
        )
        written = report_with(critic=critic)
        assert "not a written review" in written

    def test_no_critic_leaves_no_empty_section(self):
        assert "## What the review found" not in report_with()


class TestPdf:
    def test_a_report_renders_to_a_real_pdf(self):
        data = render_pdf(report_with(), title="Test report")
        assert data.startswith(b"%PDF-")
        assert len(data) > 1000

    def test_tables_survive_the_render(self):
        """Markdown tables need the `tables` extension; without it they vanish."""
        markdown = "# T\n\n| Model | Score |\n| --- | --- |\n| Forest | 0.81 |\n"
        assert render_pdf(markdown).startswith(b"%PDF-")

    def test_a_title_with_markup_in_it_does_not_break_the_document(self):
        """The filename reaches raw HTML, and filenames are user input."""
        data = render_pdf("# Hi\n\ntext\n", title="<script>alert(1)</script>.csv")
        assert data.startswith(b"%PDF-")

    def test_an_empty_report_still_renders(self):
        assert render_pdf("").startswith(b"%PDF-")

    def test_the_error_type_is_specific_enough_to_catch(self):
        """The graph catches PdfError to keep a render failure off the job."""
        assert issubclass(PdfError, RuntimeError)
