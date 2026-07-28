"""The Results page, rendered headlessly (spec 7.11, 7.12).

Several of Section 9's exit criteria are claims about a *screen*: that the review
appears, that its worst finding is on top, that a PDF can be downloaded. Asserting
those against the Markdown report would prove the wrong thing, and clicking
through the browser proves it once rather than on every commit.

``AppTest`` runs the page in-process with no browser and no server, which is what
makes these assertions cheap enough to keep. The API is stubbed at
``requests.get`` -- the page is the subject here, and the endpoints it calls have
their own tests.

The stub returns 404 for anything a test does not explicitly provide, which is
deliberate: it means each test states only the artifacts it is about, and the
page has to stay tolerant of missing ones -- exactly the behaviour a partly
finished run depends on.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from app.ml.contracts import (
    CriticFinding,
    CriticReport,
    EvaluationReport,
    FoldScore,
    MetricSummary,
)

PAGE = str(Path(__file__).resolve().parents[1] / "ui" / "pages" / "3_Results.py")

# Built from the contracts the API actually serves rather than hand-written
# dicts, so the page is tested against the real payload shape. A field added to
# a contract turns up here without anyone remembering to copy it across -- and
# if the page cannot cope with it, these tests are where that surfaces.
EVALUATION = EvaluationReport(
    task_type="classification",
    target_column="churned",
    model_name="LogisticRegression",
    n_folds=5,
    cv_strategy="StratifiedKFold",
    n_rows=500,
    n_features=8,
    folds=[
        FoldScore(fold=i, n_train=400, n_test=100, metrics={"f1_macro": 0.77 + i / 1000})
        for i in range(1, 6)
    ],
    metrics={"f1_macro": MetricSummary(mean=0.7762, std=0.0167)},
    primary_metric="f1_macro",
).model_dump(mode="json")

CRITIC = CriticReport(
    verdict="The model is usable with one thing to check first.",
    findings=[
        CriticFinding(
            area="features",
            severity="note",
            finding="A column was dropped as constant.",
            recommendation="Confirm it is genuinely constant upstream.",
        ),
        CriticFinding(
            area="explainability",
            severity="blocker",
            finding="Satisfaction pushes towards churn, which is backwards.",
            recommendation="Audit the encoding of the satisfaction scale.",
            measured=True,
        ),
        CriticFinding(
            area="modelling",
            severity="concern",
            finding="The top two models are within one standard deviation.",
            recommendation="Treat the ranking as provisional.",
            measured=True,
        ),
    ],
    strengths=["The validation strategy is sound."],
    recommended_next_steps=["Collect more rows before deploying."],
    source="llm",
).model_dump(mode="json")

REPORT = {"markdown": "# Analysis\n\nThe report body.\n"}


class FakeResponse:
    def __init__(self, payload, status_code=200, content=b""):
        self._payload = payload
        self.status_code = status_code
        self.content = content

    def json(self):
        if self._payload is None:
            return {"detail": "Not found."}
        return self._payload


def run_page(monkeypatch, *, routes: dict, pdf: FakeResponse | None = None) -> AppTest:
    """Render the page for job 42 with the given endpoints available."""

    def fake_get(url, **_kwargs):
        if url.endswith("/report/pdf"):
            return pdf if pdf is not None else FakeResponse(None, status_code=404)
        path = url.rsplit("/jobs/42/", 1)[-1]
        if path in routes:
            return FakeResponse(routes[path])
        return FakeResponse(None, status_code=404)

    monkeypatch.setattr("requests.get", fake_get)

    app = AppTest.from_file(PAGE, default_timeout=30).run()
    # The page stops until a job is entered, which is the first thing to prove
    # about it -- then re-runs with one, which is the state under test.
    app.text_input[0].set_value("42").run()
    return app


def page_text(app: AppTest) -> str:
    """Everything the page rendered, as one searchable string.

    Walks the element tree rather than using the typed accessors, because half of
    this page's Section 9 content sits inside expanders and columns -- and a test
    that only searched the top level would pass while the review was invisible.

    Dataframes contribute their serialised data: the findings are a table, not
    prose, so the text under test is inside one.
    """
    found: list[str] = []

    def visit(node) -> None:
        for attribute in ("value", "label", "body"):
            value = getattr(node, attribute, None)
            if isinstance(value, str):
                found.append(value)
        if type(node).__name__ == "Dataframe":
            found.append(_table_text(node))
        # ``children`` is a dict keyed by position, so iterate its values.
        children = getattr(node, "children", None) or {}
        for child in children.values() if isinstance(children, dict) else children:
            visit(child)

    visit(app.main)
    return "\n".join(found)


def _table_text(element) -> str:
    """A rendered table's full contents.

    ``str()`` on a DataFrame elides both rows and long cells, which is precisely
    where the findings live -- so the frame is serialised properly rather than
    printed.
    """
    value = element.value
    if isinstance(value, pd.DataFrame):
        return value.to_json(orient="records")
    return json.dumps(value, default=str)


def download_labels(app: AppTest) -> list[str]:
    """``st.download_button`` is its own element type, not a Button."""
    return [element.label for element in app.get("download_button")]


def _findings_table(app: AppTest) -> str:
    """The rendered review table, found by content rather than by position."""
    for element in app.dataframe:
        serialised = _table_text(element)
        if "Severity" in serialised:
            return serialised
    raise AssertionError("the review's findings table was not rendered")


class TestThePageStillWorks:
    def test_no_job_id_asks_for_one_rather_than_erroring(self, monkeypatch):
        monkeypatch.setattr("requests.get", lambda url, **kw: FakeResponse(None, status_code=404))
        app = AppTest.from_file(PAGE, default_timeout=30).run()
        assert not app.exception
        assert any("Enter a job ID" in info.value for info in app.info)

    def test_a_job_with_results_renders_the_headline(self, monkeypatch):
        app = run_page(monkeypatch, routes={"evaluation": EVALUATION})
        assert not app.exception
        assert any("0.7762" in metric.value for metric in app.metric)

    def test_a_run_with_nothing_but_an_evaluation_does_not_crash(self, monkeypatch):
        """Every other artifact 404s here -- a partly finished run must still render."""
        app = run_page(monkeypatch, routes={"evaluation": EVALUATION})
        assert not app.exception


class TestTheReviewOnScreen:
    def test_the_review_is_shown(self, monkeypatch):
        app = run_page(monkeypatch, routes={"evaluation": EVALUATION, "critic": CRITIC})
        assert not app.exception

        text = page_text(app)
        assert "What the review found" in text
        assert "The model is usable with one thing to check first." in text
        assert "Satisfaction pushes towards churn, which is backwards." in text

    def test_the_worst_finding_is_at_the_top(self, monkeypatch):
        """A review in the order the checks happened to run buries the blocker."""
        app = run_page(monkeypatch, routes={"evaluation": EVALUATION, "critic": CRITIC})
        rendered = _findings_table(app)

        blocker = rendered.index("Satisfaction pushes")
        concern = rendered.index("The top two models")
        note = rendered.index("A column was dropped")
        assert blocker < concern < note

    def test_a_measured_finding_is_distinguished_from_an_opinion(self, monkeypatch):
        """Both are worth reading; they are not worth the same."""
        app = run_page(monkeypatch, routes={"evaluation": EVALUATION, "critic": CRITIC})
        rendered = _findings_table(app)
        assert "measured" in rendered
        assert "review" in rendered

    def test_the_strengths_and_next_steps_are_shown(self, monkeypatch):
        app = run_page(monkeypatch, routes={"evaluation": EVALUATION, "critic": CRITIC})
        text = page_text(app)
        assert "The validation strategy is sound." in text
        assert "Collect more rows before deploying." in text

    def test_a_partial_review_says_what_it_did_not_see(self, monkeypatch):
        """The wide-dataset case: a critique of part of a run must admit it."""
        partial = {
            **CRITIC,
            "detail_level": "compact",
            "omissions": ["Showing 40 of 121 column statistics."],
        }
        app = run_page(monkeypatch, routes={"evaluation": EVALUATION, "critic": partial})
        text = page_text(app)
        assert "What the review did not see" in text
        assert "Showing 40 of 121 column statistics." in text

    def test_a_checklist_only_review_is_labelled(self, monkeypatch):
        """Without a key the findings are threshold checks, not a written review."""
        checks_only = {**CRITIC, "source": "default"}
        app = run_page(monkeypatch, routes={"evaluation": EVALUATION, "critic": checks_only})
        assert "no language model was available" in page_text(app)

    def test_no_review_leaves_no_empty_heading(self, monkeypatch):
        app = run_page(monkeypatch, routes={"evaluation": EVALUATION})
        assert "What the review found" not in page_text(app)


class TestDownloadingTheReport:
    def test_the_pdf_button_appears_when_there_is_a_pdf(self, monkeypatch):
        """Section 9's "Done when": the PDF downloads from the browser."""
        app = run_page(
            monkeypatch,
            routes={"evaluation": EVALUATION, "report": REPORT},
            pdf=FakeResponse(None, status_code=200, content=b"%PDF-1.7 body"),
        )
        assert not app.exception

        labels = download_labels(app)
        assert any("PDF" in label for label in labels), labels
        assert any("Markdown" in label for label in labels), labels

    def test_a_missing_pdf_degrades_to_a_caption_not_a_dead_button(self, monkeypatch):
        """Rendering is best-effort, so its absence is a normal state to render."""
        app = run_page(monkeypatch, routes={"evaluation": EVALUATION, "report": REPORT})
        assert not app.exception

        assert not any("PDF" in label for label in download_labels(app))
        assert "authoritative version" in page_text(app)

    @pytest.mark.parametrize("failure", [ConnectionError, TimeoutError])
    def test_the_page_survives_the_pdf_request_failing(self, monkeypatch, failure):
        """A network error fetching an optional extra must not take the page down."""
        import requests

        def fake_get(url, **_kwargs):
            if url.endswith("/report/pdf"):
                raise requests.exceptions.RequestException(str(failure))
            path = url.rsplit("/jobs/42/", 1)[-1]
            payload = {"evaluation": EVALUATION, "report": REPORT}.get(path)
            return FakeResponse(payload, status_code=200 if payload else 404)

        monkeypatch.setattr("requests.get", fake_get)
        app = AppTest.from_file(PAGE, default_timeout=30).run()
        app.text_input[0].set_value("42").run()

        assert not app.exception
        assert "authoritative version" in page_text(app)

    def test_the_report_body_is_rendered(self, monkeypatch):
        app = run_page(monkeypatch, routes={"evaluation": EVALUATION, "report": REPORT})
        assert "The report body." in page_text(app)
