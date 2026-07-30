"""Progress page.

Training runs in a background worker because it takes minutes and an HTTP
request cannot (Section 4). This page is the other half of that design: it polls
``GET /jobs/{id}`` and shows the pipeline ticking through its nodes, so the user
sees live progress instead of a spinner that might be doing nothing.

In Section 4 the nodes only sleep, so what this proves is the *plumbing* -- the
API queued the job, the worker picked it up, and status flows back to the
browser. Section 5 puts real work behind the same bars.
"""

from __future__ import annotations

import os
import time

import requests
import streamlit as st

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

st.set_page_config(page_title="Progress · AutoDS", page_icon="📊", layout="wide")

# Streamlit runs each page as its own script, so a page cannot import a sibling
# module without ui/ being importable. The path work has to happen before the
# import, which is what the noqa below is for.
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

sys.path.append(str(Path(__file__).resolve().parents[1]))

from auth import require_access  # noqa: E402

require_access()

st.title("Pipeline progress")

# Terminal states stop the auto-refresh loop; everything else is still moving.
TERMINAL = {"completed", "failed"}
NODE_ICON = {
    "pending": "⬜",
    "running": "🔄",
    "completed": "✅",
    "failed": "❌",
    "skipped": "⏭️",
}

# Pre-fill from a job just confirmed on the Upload page, if there is one.
default_job = ""
if "confirmed" in st.session_state:
    default_job = str(st.session_state["confirmed"]["id"])

job_id = st.text_input("Job ID", value=default_job, placeholder="e.g. 42")
auto = st.checkbox("Auto-refresh while running", value=True)

if not job_id:
    st.info("Enter a job ID to watch its pipeline.")
    st.stop()

try:
    resp = requests.get(f"{API_BASE_URL}/jobs/{job_id}", timeout=10)
except requests.exceptions.RequestException as exc:
    st.error("Could not reach the API.")
    st.code(str(exc), language=None)
    st.stop()

if resp.status_code == 404:
    st.warning(f"No job {job_id}.")
    st.stop()
if resp.status_code != 200:
    st.error(f"Unexpected response (HTTP {resp.status_code}).")
    st.stop()

job = resp.json()
status = job["status"]
runs = job.get("agent_runs", [])

# ---- Overall status ---------------------------------------------------------

top = st.columns(3)
top[0].metric("Status", status)
top[1].metric("Target", job.get("target_column") or "—")
top[2].metric("Task", job.get("task_type") or "—")

if runs:
    # A skipped step is settled, not outstanding. Counting only "completed"
    # would leave a finished run whose planner turned two steps off stuck at
    # 8/10 forever, which reads as a stall rather than as a decision.
    done = sum(1 for r in runs if r["status"] in ("completed", "skipped"))
    skipped = sum(1 for r in runs if r["status"] == "skipped")
    label = f"{done} / {len(runs)} steps settled"
    if skipped:
        label += f" ({skipped} skipped by the plan)"
    st.progress(done / len(runs), text=label)

if status == "failed":
    st.error(f"Pipeline failed: {job.get('error_message') or 'unknown error'}")
elif status == "completed":
    st.success("Pipeline completed.")

# ---- Per-node timeline ------------------------------------------------------

st.subheader("Steps")
if not runs:
    st.caption("No pipeline steps yet — the job has not been queued.")
else:
    for run in runs:
        icon = NODE_ICON.get(run["status"], "•")
        line = f"{icon} **{run['name']}** — {run['status']}"
        # A skipped step carries its reason in the same column a failure does;
        # the status beside it is what says which of the two happened. Showing
        # the reason is the whole point of marking a step skipped rather than
        # dropping it from the list -- the pipeline adapted, and says how.
        if run["status"] in ("failed", "skipped") and run.get("error_message"):
            line += f" — {run['error_message']}"
        st.markdown(line)

# ---- Auto-refresh -----------------------------------------------------------
# Streamlit has no timer, so we sleep and rerun while the job is still moving.
# The loop ends the moment the job reaches a terminal state.
if auto and status not in TERMINAL:
    time.sleep(2)
    st.rerun()
