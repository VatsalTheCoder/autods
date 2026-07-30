"""Streamlit frontend — the landing page.

What this page leads with is a deliberate choice. It used to open with a health
check and a thirteen-row build tracker, which are the two things *this project's
author* wants to know and the two things a visitor does not. Someone arriving
here needs to learn what the thing does and how to start it; whether the storage
container is reachable is a question for when something has already gone wrong.

So the order is: what it does, how to begin, how it works, then diagnostics --
with the developer-facing material still present, still one click away, and no
longer standing between a visitor and the product.
"""

from __future__ import annotations

import os

import requests
import streamlit as st

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
TIMEOUT_SECONDS = 5

st.set_page_config(page_title="AutoDS", page_icon="🔬", layout="wide")


def fetch_health() -> tuple[bool, dict | str]:
    """Call the API health endpoint. Returns (reachable, payload_or_error)."""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=TIMEOUT_SECONDS)
        return True, response.json()
    except requests.exceptions.RequestException as exc:
        return False, str(exc)


# ---- What it does -----------------------------------------------------------

st.title("AutoDS")
# Regular weight, not a heading. Set as `#####` this ran as bold serif across two
# full lines, which read as a second headline competing with the title rather
# than as the sentence explaining it.
st.markdown(
    "<p style='font-size:1.15rem; line-height:1.6; max-width:62ch; margin:0;'>"
    "Upload a CSV. Get back a cleaned dataset, exploratory analysis, a "
    "cross-validated model, SHAP explanations of its behaviour, a written "
    "report, and a chat interface over all of it.</p>",
    unsafe_allow_html=True,
)

st.write("")

# The health check runs before the call to action so a broken stack is visible
# where it changes what the visitor should do, rather than as a green tick far
# below the button they are about to press.
reachable, payload = fetch_health()
healthy = reachable and isinstance(payload, dict) and payload.get("status") == "healthy"

if not healthy:
    st.warning(
        "The API is not reachable, so uploading will not work yet. "
        "Check the stack is up with `docker compose ps`, then refresh.",
        icon="⚠️",
    )

# A real primary button rather than st.page_link. The link renders as small
# inline text, which is the wrong weight for the one action this page exists to
# prompt -- and the lone emoji on it was the only one on the page.
if st.button("Upload a dataset", type="primary", disabled=not healthy):
    st.switch_page("pages/1_Upload.py")

st.caption(
    "Two example datasets ship with the project: `data/examples/customer_churn.csv` "
    "for classification and `data/examples/house_prices.csv` for regression."
)

st.divider()

# ---- How it works -----------------------------------------------------------

st.subheader("How it works")
st.caption(
    "A dozen specialised agents coordinated by LangGraph. The language model "
    "makes decisions; ordinary Python carries them out."
)

steps = st.columns(4)
for column, (heading, body) in zip(
    steps,
    [
        (
            "1 · You confirm the schema",
            "Detection reads the file and proposes a target, a task type and "
            "which columns look like personal data. Nothing runs until you agree.",
        ),
        (
            "2 · It plans and prepares",
            "A planner writes the route through the graph, so different datasets "
            "genuinely take different paths. Cleaning and feature choices follow.",
        ),
        (
            "3 · It trains and explains",
            "Preparation is fitted **inside** each cross-validation fold, never "
            "before the split. SHAP then explains the winner in your own column names.",
        ),
        (
            "4 · It reviews itself",
            "A critic checks the run against measured thresholds before the report "
            "is written, so a problem lands in the summary rather than a footnote.",
        ),
    ],
    strict=True,
):
    with column:
        st.markdown(f"**{heading}**")
        st.caption(body)

st.divider()

# ---- Diagnostics, below the fold and collapsed ------------------------------

status_label = (
    "System status — all services healthy" if healthy else "System status — needs attention"
)
with st.expander(status_label, expanded=not healthy):
    col_left, col_right = st.columns([1, 2])

    with col_left:
        st.markdown("**API endpoint**")
        st.code(API_BASE_URL, language=None)
        st.button("Refresh", use_container_width=True)

    with col_right:
        if not reachable:
            st.error("Cannot reach the API.")
            st.caption(
                "If you are running outside Docker, check the API container is up "
                "with `docker compose ps`."
            )
            with st.expander("Error detail"):
                st.code(payload, language=None)
        else:
            overall = payload.get("status", "unknown")
            if overall == "healthy":
                st.success(f"API reachable — status: **{overall}**")
            else:
                st.warning(f"API reachable but degraded — status: **{overall}**")

            deps = payload.get("dependencies", {})
            dep_cols = st.columns(max(len(deps), 1))
            for column, (name, ok) in zip(dep_cols, deps.items(), strict=False):
                column.metric(
                    label=name.capitalize(),
                    value="up" if ok else "down",
                    delta="connected" if ok else "unreachable",
                    delta_color="normal" if ok else "inverse",
                )

            st.caption(
                "Redis is not listed: the API does not talk to it. A dead Redis "
                "shows up as jobs sitting in *queued*, not as an unhealthy API."
            )

            with st.expander("Raw response"):
                st.json(payload)

# Hand-maintained, and it went stale once already: this table stopped at
# Section 7 while Sections 7-10 were merging, so the landing page told visitors
# feature engineering had not been started for four sections. Every section is
# listed now, so falling behind shows up as a wrong status rather than a missing
# row.
#
# Three places state build status and they have to move together:
#   - this table
#   - the "Build progress" list in README.md
#   - the published progress report (see docs/RUNBOOK.md for the URL)
with st.expander("Build progress — 11 of 13 sections merged"):
    st.markdown(
        """
| Section | Status |
|---|---|
| 0 · Skeleton | ✅ done |
| 1 · Upload | ✅ done |
| 2 · LLM client | ✅ done |
| 3 · Schema detection & human checkpoint | ✅ done |
| 4 · Background worker | ✅ done |
| 5 · Vertical slice *(milestone M1)* | ✅ done |
| 6 · EDA & clustering *(milestone M2)* | ✅ done |
| 7 · Feature engineering *(milestone M3)* | ✅ done |
| 8 · Final training, SHAP & prediction *(milestone M4)* | ✅ done |
| 9 · Critic & report *(milestone M5)* | ✅ done |
| 10 · RAG chat *(milestone M6)* | ✅ done |
| 11 · AWS deployment *(milestone M7)* | 🟡 in progress — runbook written, hosting pending |
| 12 · Testing & docs | 🟡 in progress |

See `BUILD_PLAN.md` for the full plan.
"""
    )
