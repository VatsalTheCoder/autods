"""Upload page.

Pick a CSV, send it to the API, and show what came back. The preview matters
more than it looks: it is the user's only chance to notice their file parsed
wrongly (a semicolon delimiter, a stray header row) before anything downstream
is built on it.
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
st.caption("Upload a CSV to start a new analysis job.")

uploaded = st.file_uploader(
    "Choose a CSV file",
    type=["csv"],
    help="UTF-8 encoded, at least two columns.",
)

if uploaded is not None:
    st.write(f"**{uploaded.name}** — {uploaded.size / 1024:.1f} KB")

    if st.button("Upload", type="primary"):
        with st.spinner("Uploading and inspecting…"):
            try:
                response = requests.post(
                    f"{API_BASE_URL}/upload",
                    files={"file": (uploaded.name, uploaded.getvalue(), "text/csv")},
                    timeout=UPLOAD_TIMEOUT_SECONDS,
                )
            except requests.exceptions.RequestException as exc:
                st.error("Could not reach the API.")
                st.code(str(exc), language=None)
                st.stop()

        if response.status_code == 201:
            payload = response.json()
            preview = payload["preview"]

            st.success(f"Created job **{payload['job_id']}**")

            a, b, c = st.columns(3)
            a.metric("Rows", f"{preview['n_rows']:,}")
            b.metric("Columns", preview["n_columns"])
            c.metric("Size", f"{payload['size_bytes'] / 1024:.1f} KB")

            st.subheader("Preview")
            st.dataframe(pd.DataFrame(preview["rows"]), use_container_width=True)

            st.subheader("Columns")
            st.write(", ".join(f"`{column}`" for column in preview["columns"]))

            st.info(
                "Next: schema detection will suggest which column to predict "
                "and flag any personal data. That arrives in Section 3."
            )

        elif response.status_code == 422:
            # The API's validation messages are written to be shown as-is.
            st.error(response.json().get("detail", "The file could not be used."))
        else:
            st.error(f"Upload failed (HTTP {response.status_code}).")
            st.code(response.text, language=None)

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
