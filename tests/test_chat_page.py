"""The Chat page, rendered headlessly (spec 7.13).

The page's one job beyond drawing a conversation is to **show which tool
answered**. A number computed from the data and a sentence retrieved from the
report are different kinds of claim, and presenting them identically would hide
the distinction this whole section is built around.

So these tests are mostly about the attribution: that a calculation shows the
expression it ran, that a retrieved answer says where it came from, and that a
run with nothing indexed explains itself instead of offering an input box that
cannot work.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from streamlit.testing.v1 import AppTest

PAGE = str(Path(__file__).resolve().parents[1] / "ui" / "pages" / "4_Chat.py")

READY = {"job_id": 42, "indexed_passages": 24, "ready": True}
NOT_READY = {"job_id": 42, "indexed_passages": 0, "ready": False}

HISTORY = [
    {
        "id": 1,
        "question": "What is the average age?",
        "answer": "The average age is 47.626.",
        "route": "pandas",
        "grounding": "query: df['age'].mean()",
    },
    {
        "id": 2,
        "question": "Why was support_calls important?",
        "answer": "It carries 37.5% of the model's explained influence.",
        "route": "rag",
        "grounding": "chunks: 9, 8",
    },
]


class FakeResponse:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code

    def json(self):
        return {"detail": "Not found."} if self._payload is None else self._payload


def run_page(monkeypatch, *, status=READY, history=None):
    """Render the page for job 42 with the given chat state."""

    def fake_get(url, **_kwargs):
        if url.endswith("/chat/status"):
            return FakeResponse(status)
        if url.endswith("/chat"):
            return FakeResponse(history if history is not None else [])
        return FakeResponse(None, status_code=404)

    monkeypatch.setattr("requests.get", fake_get)
    monkeypatch.setattr("requests.post", lambda *a, **k: FakeResponse(None, status_code=404))

    app = AppTest.from_file(PAGE, default_timeout=30).run()
    app.text_input[0].set_value("42").run()
    return app


def page_text(app: AppTest) -> str:
    """Everything rendered, including inside chat messages and expanders."""
    found: list[str] = []

    def visit(node) -> None:
        for attribute in ("value", "label", "body"):
            value = getattr(node, attribute, None)
            if isinstance(value, str):
                found.append(value)
        if type(node).__name__ == "Dataframe" and isinstance(node.value, pd.DataFrame):
            found.append(node.value.to_json(orient="records"))
        children = getattr(node, "children", None) or {}
        for child in children.values() if isinstance(children, dict) else children:
            visit(child)

    visit(app.main)
    return "\n".join(found)


class TestThePageWorks:
    def test_no_job_id_asks_for_one(self, monkeypatch):
        monkeypatch.setattr("requests.get", lambda *a, **k: FakeResponse(None, status_code=404))
        app = AppTest.from_file(PAGE, default_timeout=30).run()
        assert not app.exception
        assert any("Enter a job ID" in info.value for info in app.info)

    def test_a_ready_run_offers_the_question_box(self, monkeypatch):
        app = run_page(monkeypatch)
        assert not app.exception
        assert app.chat_input, "there is no way to ask a question"

    def test_it_says_how_much_it_can_answer_from(self, monkeypatch):
        assert "24 passages" in page_text(run_page(monkeypatch))

    def test_the_guidance_shows_both_kinds_of_question(self, monkeypatch):
        """The routing is automatic, so the page has to teach what it covers."""
        text = page_text(run_page(monkeypatch))
        assert "Questions about meaning" in text
        assert "Questions about numbers" in text


class TestAnUnindexedRun:
    def test_it_explains_itself_rather_than_offering_a_dead_box(self, monkeypatch):
        app = run_page(monkeypatch, status=NOT_READY)
        assert not app.exception
        assert not app.chat_input, "a run with nothing indexed cannot answer"
        assert "no searchable text" in page_text(app)

    def test_it_says_how_to_fix_it(self, monkeypatch):
        assert "re-running the job" in page_text(run_page(monkeypatch, status=NOT_READY))


class TestShowingWhichToolAnswered:
    def test_a_calculation_shows_the_expression_it_ran(self, monkeypatch):
        """The pandas failure mode is answering a subtly different question."""
        text = page_text(run_page(monkeypatch, history=HISTORY))
        assert "computed from the data" in text
        assert "df['age'].mean()" in text

    def test_a_retrieved_answer_says_where_it_came_from(self, monkeypatch):
        text = page_text(run_page(monkeypatch, history=HISTORY))
        assert "from the run's written output" in text
        assert "chunks: 9, 8" in text

    def test_both_answers_are_rendered(self, monkeypatch):
        text = page_text(run_page(monkeypatch, history=HISTORY))
        assert "The average age is 47.626." in text
        assert "37.5%" in text

    def test_the_questions_are_rendered_too(self, monkeypatch):
        text = page_text(run_page(monkeypatch, history=HISTORY))
        assert "What is the average age?" in text
        assert "Why was support_calls important?" in text

    def test_internal_markers_are_not_shown_to_the_reader(self, monkeypatch):
        """'no-passages' is for the transcript; the answer explains itself."""
        history = [
            {
                "id": 1,
                "question": "Something unasked",
                "answer": "Nothing in this run's output addresses that.",
                "route": "rag",
                "grounding": "no-passages",
            }
        ]
        text = page_text(run_page(monkeypatch, history=history))
        assert "no-passages" not in text
        assert "Nothing in this run's output" in text

    def test_a_refusal_is_labelled_as_one(self, monkeypatch):
        history = [
            {
                "id": 1,
                "question": "Who won the world cup?",
                "answer": "That is outside what this analysis covers.",
                "route": "refused",
                "grounding": "not about this dataset",
            }
        ]
        assert "not answered" in page_text(run_page(monkeypatch, history=history))
