"""Streamlit frontend.

Section 0 renders a single page that calls the API's /health endpoint. Trivial
on purpose: it proves the UI container can reach the API container across the
Docker network, which is exactly the integration that tends to break first.

The Upload, Progress and Results pages arrived in Sections 1, 4 and 5, which
together make the product demoable end to end; the Chat page follows in Section 10.
"""

from __future__ import annotations

import os

import requests
import streamlit as st

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
TIMEOUT_SECONDS = 5

st.set_page_config(page_title="AutoDS", page_icon="🔬", layout="wide")

st.title("🔬 AutoDS")
st.caption("Multi-agent autonomous data scientist")

st.divider()
st.subheader("System status")


def fetch_health() -> tuple[bool, dict | str]:
    """Call the API health endpoint. Returns (reachable, payload_or_error)."""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=TIMEOUT_SECONDS)
        return True, response.json()
    except requests.exceptions.RequestException as exc:
        return False, str(exc)


col_left, col_right = st.columns([1, 2])

with col_left:
    st.markdown("**API endpoint**")
    st.code(API_BASE_URL, language=None)
    refresh = st.button("Refresh", use_container_width=True)

with col_right:
    reachable, payload = fetch_health()

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

        with st.expander("Raw response"):
            st.json(payload)

st.divider()

st.subheader("Build progress")
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
