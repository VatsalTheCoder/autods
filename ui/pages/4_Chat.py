"""Chat page -- ask a finished run questions (spec 7.13).

The page's one real job beyond rendering a conversation is to **show which tool
answered**. A number computed from the data and a sentence retrieved from the
report are different kinds of claim: the first is exact and could be answering a
subtly different question, the second is grounded in text that a reader can go
and check. Presenting both as undifferentiated chat would hide the distinction
that the whole section is built around.

So every answer carries its route and its grounding -- the pandas expression that
ran, or the passages the answer came from.
"""

from __future__ import annotations

import os

import requests
import streamlit as st

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

st.set_page_config(page_title="Chat · AutoDS", page_icon="💬", layout="wide")

st.title("💬 Ask about this run")

# How each route is labelled. The wording states what the answer *is*, rather
# than naming the tool -- "computed from the data" tells a reader something,
# "pandas" tells them which library was imported.
ROUTE_LABEL = {
    "pandas": ("🧮", "computed from the data"),
    "rag": ("📄", "from the run's written output"),
    "refused": ("🚫", "not answered"),
}

default_job = ""
if "confirmed" in st.session_state:
    default_job = str(st.session_state["confirmed"]["id"])

job_id = st.text_input("Job ID", value=default_job, placeholder="e.g. 42")
if not job_id:
    st.info("Enter a job ID to ask questions about it.")
    st.stop()


def api(path: str, method: str = "get", **kwargs):
    """Call the chat API, returning (payload, error_message)."""
    try:
        call = requests.post if method == "post" else requests.get
        # Generous: a question is two model calls, and the second one can retry
        # through a rate limit before answering.
        resp = call(f"{API_BASE_URL}/jobs/{job_id}/{path}", timeout=120, **kwargs)
    except requests.exceptions.RequestException as exc:
        return None, f"Could not reach the API: {exc}"
    if resp.status_code == 404:
        return None, resp.json().get("detail", "Not found.")
    if resp.status_code != 200:
        return None, f"Unexpected response (HTTP {resp.status_code})."
    return resp.json(), None


status, status_error = api("chat/status")
if status is None:
    st.warning(status_error)
    st.caption("If the pipeline is still running, watch it on the Progress page.")
    st.stop()

if not status["ready"]:
    st.warning(
        "This run has no searchable text yet, so there is nothing to ask about. "
        "Indexing is the last step of the pipeline — if the run has finished, it "
        "may have failed at that step, and re-running the job will index it."
    )
    st.stop()

st.caption(
    f"Answering from {status['indexed_passages']} passages of this run's output, "
    "and from the cleaned dataset for anything arithmetic."
)

with st.expander("What can I ask?"):
    st.markdown(
        "**Questions about meaning** are answered from the report, the review, "
        "the cluster profiles and the explainability results:\n"
        "- Why was *(a column)* important?\n"
        "- Which features drive the outcome?\n"
        "- Explain the model simply.\n"
        "- What did the review find wrong?\n\n"
        "**Questions about numbers** are answered by running a calculation over "
        "the cleaned dataset:\n"
        "- What is the average *(a column)*?\n"
        "- How many rows are there?\n"
        "- How many rows have *(a value)*?\n\n"
        "The two are routed automatically, and each answer says which route it "
        "took."
    )


def render_answer(message: dict) -> None:
    """The assistant's half: the answer, then how it was arrived at."""
    with st.chat_message("assistant"):
        st.markdown(message["answer"])
        icon, label = ROUTE_LABEL.get(message["route"], ("•", message["route"]))
        grounding = message.get("grounding") or ""
        if grounding.startswith("query: "):
            # The expression is shown as code, because it is code, and because a
            # calculation that answered the wrong question is only catchable if
            # the reader can see what was actually computed.
            st.caption(f"{icon} {label}")
            st.code(grounding.removeprefix("query: "), language="python")
        else:
            # Internal markers ("no-passages", "not-indexed") are for the
            # transcript, not the reader -- the answer already explains itself.
            detail = f" · {grounding}" if grounding and not grounding.startswith("no") else ""
            st.caption(f"{icon} {label}{detail}")


def render(message: dict) -> None:
    """One full exchange."""
    with st.chat_message("user"):
        st.markdown(message["question"])
    render_answer(message)


history, history_error = api("chat")
if history is None:
    st.warning(history_error)
    st.stop()

for message in history:
    render(message)

question = st.chat_input("Ask about this run…")
if question:
    with st.chat_message("user"):
        st.markdown(question)
    with st.spinner("Working out how to answer…"):
        answer, answer_error = api("chat", method="post", json={"question": question})

    if answer is None:
        st.error(answer_error)
    else:
        # Only the answer is re-rendered: the question was drawn above, and the
        # history was drawn before that. A rerun would refetch every message in
        # the conversation to display one.
        render_answer(answer)
