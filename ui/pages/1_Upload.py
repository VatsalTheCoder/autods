"""Upload page.

Pick a CSV, send it to the API, and show what came back. The preview matters
more than it looks: it is the user's only chance to notice their file parsed
wrongly (a semicolon delimiter, a stray header row) before anything downstream
is built on it.

Then the human checkpoint (Section 3): the detected schema is shown with every
guess editable -- target, task type, and which columns are personal data to
exclude -- and confirming it saves the approved version against the job. The
upload response carries the schema, so the form renders with no second request.
"""

from __future__ import annotations

import os

import pandas as pd
import requests
import streamlit as st

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
UPLOAD_TIMEOUT_SECONDS = 120

st.set_page_config(page_title="Upload · AutoDS", page_icon="📤", layout="wide")

st.title("📤 Upload a dataset")
st.caption("Upload a CSV to start a new analysis job, then confirm its schema.")


def _do_upload(name: str, data: bytes) -> None:
    """Upload the file and stash the response for the confirmation step."""
    with st.spinner("Uploading and detecting schema…"):
        try:
            response = requests.post(
                f"{API_BASE_URL}/upload",
                files={"file": (name, data, "text/csv")},
                timeout=UPLOAD_TIMEOUT_SECONDS,
            )
        except requests.exceptions.RequestException as exc:
            st.error("Could not reach the API.")
            st.code(str(exc), language=None)
            return

    if response.status_code == 201:
        # Persist across reruns: editing a widget below reruns the script, and
        # the payload must survive so the form does not vanish mid-confirmation.
        st.session_state["upload"] = response.json()
        st.session_state.pop("confirmed", None)
    elif response.status_code == 422:
        st.error(response.json().get("detail", "The file could not be used."))
    else:
        st.error(f"Upload failed (HTTP {response.status_code}).")
        st.code(response.text, language=None)


uploaded = st.file_uploader(
    "Choose a CSV file",
    type=["csv"],
    help="UTF-8 encoded, at least two columns.",
)

if uploaded is not None:
    st.write(f"**{uploaded.name}** — {uploaded.size / 1024:.1f} KB")
    if st.button("Upload", type="primary"):
        _do_upload(uploaded.name, uploaded.getvalue())


# ---- Confirmation step ------------------------------------------------------


def _render_confirmation(payload: dict) -> None:
    report = payload["schema_report"]
    job_id = payload["job_id"]
    columns = report["columns"]
    names = [c["name"] for c in columns]

    st.success(f"Created job **{job_id}**")

    a, b, c = st.columns(3)
    a.metric("Rows", f"{report['n_rows']:,}")
    b.metric("Columns", report["n_columns"])
    c.metric("Size", f"{payload['size_bytes'] / 1024:.1f} KB")

    st.subheader("Preview")
    st.dataframe(pd.DataFrame(payload["preview"]["rows"]), use_container_width=True)

    if not report["llm_enriched"]:
        st.caption(
            "ℹ️ Column meanings were not generated (no LLM configured); the "
            "suggestions below are from automatic profiling only."
        )

    st.subheader("Confirm the schema")
    st.caption("Everything here is a suggestion. Correct anything before continuing.")

    suggested_target = report.get("suggested_target") or names[-1]
    target = st.selectbox(
        "Target column — the thing to predict",
        options=names,
        index=names.index(suggested_target) if suggested_target in names else len(names) - 1,
    )

    task_default = report.get("task_type") or "classification"
    task_type = st.radio(
        "Task type",
        options=["classification", "regression"],
        index=0 if task_default == "classification" else 1,
        horizontal=True,
    )

    if report.get("class_balance") and report["class_balance"]["imbalanced"]:
        ratio = report["class_balance"]["imbalance_ratio"]
        st.warning(
            f"The target looks imbalanced ({ratio:.1f}:1). SMOTE (Section 7) will "
            "address this inside cross-validation."
        )

    st.markdown("**Columns** — tick personal data, and anything to exclude from modelling.")
    editor_rows = [
        {
            "column": col["name"],
            "type": col["semantic_type"],
            "meaning": col.get("meaning") or "",
            "PII": col["is_pii"],
            "exclude": col["exclude"],
        }
        for col in columns
    ]
    edited = st.data_editor(
        pd.DataFrame(editor_rows),
        use_container_width=True,
        hide_index=True,
        disabled=["column", "type", "meaning"],
        column_config={
            "PII": st.column_config.CheckboxColumn(help="Is this personal data?"),
            "exclude": st.column_config.CheckboxColumn(help="Leave this column out of the model?"),
        },
        key=f"editor_{job_id}",
    )

    if st.button("Confirm schema", type="primary"):
        body = {
            "job_id": job_id,
            "target_column": target,
            "task_type": task_type,
            "columns": [
                {"name": row["column"], "is_pii": bool(row["PII"]), "exclude": bool(row["exclude"])}
                for row in edited.to_dict(orient="records")
            ],
        }
        try:
            resp = requests.post(f"{API_BASE_URL}/jobs", json=body, timeout=30)
        except requests.exceptions.RequestException as exc:
            st.error("Could not reach the API.")
            st.code(str(exc), language=None)
            return

        if resp.status_code == 200:
            st.session_state["confirmed"] = resp.json()
        elif resp.status_code == 422:
            st.error(resp.json().get("detail", "The confirmation was rejected."))
        else:
            st.error(f"Confirmation failed (HTTP {resp.status_code}).")
            st.code(resp.text, language=None)


if "upload" in st.session_state:
    st.divider()
    _render_confirmation(st.session_state["upload"])

if "confirmed" in st.session_state:
    job = st.session_state["confirmed"]
    st.success(
        f"✅ Job **{job['id']}** confirmed — target **{job['target_column']}**, "
        f"task **{job['task_type']}**. The pipeline launches from here in Section 4."
    )


# ---- Previous jobs ----------------------------------------------------------

st.divider()
st.subheader("Previous jobs")

try:
    jobs = requests.get(f"{API_BASE_URL}/jobs", timeout=10).json()
except requests.exceptions.RequestException:
    st.caption("Could not load jobs — is the API running?")
    jobs = []

if not jobs:
    st.caption("No jobs yet.")
else:
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "ID": job["id"],
                    "File": job["original_filename"],
                    "Status": job["status"],
                    "Target": job.get("target_column") or "—",
                    "Task": job.get("task_type") or "—",
                    "Rows": job["n_rows"],
                    "Columns": job["n_columns"],
                    "Uploaded": job["created_at"][:19].replace("T", " "),
                }
                for job in jobs
            ]
        ),
        use_container_width=True,
        hide_index=True,
    )
