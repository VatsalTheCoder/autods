"""Results page -- the far end of the vertical slice.

This is the screen that makes Section 5's "Done when" true: a CSV uploaded in the
browser produces a cross-validated score and a readable report, without anyone
opening a terminal.

It shows the headline metric first, the fold-by-fold detail underneath, and then
the report itself. The per-fold table is not decoration -- it is where a viewer
can see that each fold was fitted on a subset and scored on the rest, which is
the claim the whole project rests on.
"""

from __future__ import annotations

import os

import requests
import streamlit as st

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

st.set_page_config(page_title="Results · AutoDS", page_icon="📈", layout="wide")

st.title("📈 Results")

# Metrics worth showing as headline tiles, in the order they read best.
HEADLINE = {
    "classification": ["f1_macro", "accuracy", "roc_auc", "pr_auc"],
    "regression": ["r2", "rmse", "mae"],
}
LABEL = {
    "accuracy": "Accuracy",
    "precision_macro": "Precision (macro)",
    "recall_macro": "Recall (macro)",
    "f1_macro": "F1 (macro)",
    "roc_auc": "ROC-AUC",
    "pr_auc": "PR-AUC",
    "mae": "MAE",
    "mse": "MSE",
    "rmse": "RMSE",
    "r2": "R²",
}

default_job = ""
if "confirmed" in st.session_state:
    default_job = str(st.session_state["confirmed"]["id"])

job_id = st.text_input("Job ID", value=default_job, placeholder="e.g. 42")
if not job_id:
    st.info("Enter a job ID to see its results.")
    st.stop()


def fetch(path: str):
    """GET a results endpoint, returning (payload, error_message)."""
    try:
        resp = requests.get(f"{API_BASE_URL}/jobs/{job_id}/{path}", timeout=15)
    except requests.exceptions.RequestException as exc:
        return None, f"Could not reach the API: {exc}"
    if resp.status_code == 404:
        return None, resp.json().get("detail", "Not found.")
    if resp.status_code != 200:
        return None, f"Unexpected response (HTTP {resp.status_code})."
    return resp.json(), None


evaluation, error = fetch("evaluation")
if evaluation is None:
    # The commonest reason to be here early is a job that is still running, so
    # point at the Progress page rather than treating it as an error.
    st.warning(error)
    st.caption("If the pipeline is still running, watch it on the Progress page.")
    st.stop()

# ---- Headline ---------------------------------------------------------------

task_type = evaluation["task_type"]
metrics = evaluation["metrics"]

st.subheader(f"{evaluation['model_name']} · predicting `{evaluation['target_column']}`")
st.caption(
    f"{task_type} · {evaluation['n_rows']:,} rows · {evaluation['n_features']} features · "
    f"{evaluation['n_folds']}-fold {evaluation['cv_strategy']}"
)

shown = [name for name in HEADLINE.get(task_type, []) if name in metrics]
if shown:
    for column, name in zip(st.columns(len(shown)), shown, strict=True):
        summary = metrics[name]
        column.metric(
            LABEL.get(name, name),
            f"{summary['mean']:.4f}",
            # The delta slot is borrowed to show the spread across folds. It is
            # not a change over time, so the arrow colouring is turned off --
            # a green "up" arrow would imply an improvement that is not there.
            delta=f"± {summary['std']:.4f}",
            delta_color="off",
        )

# ---- All metrics ------------------------------------------------------------

with st.expander("All metrics", expanded=True):
    st.dataframe(
        [
            {
                "Metric": LABEL.get(name, name),
                "Mean": round(summary["mean"], 6),
                "Std. dev. across folds": round(summary["std"], 6),
            }
            for name, summary in metrics.items()
        ],
        hide_index=True,
        width="stretch",
    )

# ---- Folds ------------------------------------------------------------------

primary = evaluation.get("primary_metric", "")
st.subheader("Folds")
st.caption(
    "Each fold's pipeline was fitted on its training rows only and scored on the "
    "held-out rows -- no imputation, scaling or encoding happened across the whole "
    "dataset first."
)
st.dataframe(
    [
        {
            "Fold": fold["fold"],
            "Rows fitted on": fold["n_train"],
            "Rows scored on": fold["n_test"],
            LABEL.get(primary, primary): round(fold["metrics"].get(primary, float("nan")), 6),
        }
        for fold in evaluation["folds"]
    ],
    hide_index=True,
    width="stretch",
)

if evaluation.get("warnings"):
    with st.expander("Caveats"):
        for warning in evaluation["warnings"]:
            st.markdown(f"- {warning}")

# ---- What the data looks like ----------------------------------------------
# Charts come through the API rather than a presigned URL: those are signed
# against the storage endpoint, which a browser cannot resolve locally.

st.divider()
eda, _ = fetch("eda")
if eda:
    st.subheader("What the data looks like")

    if eda.get("class_balance", {}) and eda["class_balance"].get("imbalanced"):
        st.warning(
            f"The target classes are imbalanced "
            f"({eda['class_balance']['imbalance_ratio']:.1f} to 1). Accuracy "
            "flatters a model on skewed data — read the macro F1 above instead."
        )

    plots = eda.get("plots", [])
    # The cluster scatter belongs with the cluster descriptions further down.
    chart_names = [p for p in plots if p != "cluster_scatter.png"]
    for left, right in zip(chart_names[::2], chart_names[1::2] + [None], strict=False):
        columns = st.columns(2)
        for column, name in zip(columns, (left, right), strict=False):
            if name:
                column.image(f"{API_BASE_URL}/jobs/{job_id}/artifacts/{name}/content")

    if eda.get("top_correlations"):
        with st.expander("Columns that move together"):
            st.dataframe(
                [
                    {
                        "Column": pair["left"],
                        "Column ": pair["right"],
                        "Correlation": round(pair["correlation"], 3),
                    }
                    for pair in eda["top_correlations"]
                ],
                hide_index=True,
                width="stretch",
            )

# ---- Groups found in the data ----------------------------------------------

clustering, _ = fetch("clustering")
if clustering and clustering.get("k"):
    st.divider()
    st.subheader("Natural groups in the data")
    st.caption(
        f"{clustering['method']} · {clustering['k']} groups · silhouette "
        f"{clustering['silhouette']:.2f} (1.0 would be perfectly separated, "
        "0 no better than arbitrary)"
    )

    scatter, descriptions = st.columns([3, 2])
    if clustering.get("scatter_plot"):
        scatter.image(
            f"{API_BASE_URL}/jobs/{job_id}/artifacts/{clustering['scatter_plot']}/content"
        )

    with descriptions:
        for profile in clustering.get("profiles", []):
            text = profile.get("description") or ""
            descriptions.markdown(
                f"**Group {profile['cluster']}** — {profile['size']:,} rows "
                f"({profile['share']:.0%})"
            )
            if text:
                descriptions.caption(text)
            for name, detail in profile.get("distinguishing_features", {}).items():
                descriptions.markdown(f"- `{name}`: {detail}")

    # The guardrail, said where a reader of the results will see it.
    st.info(
        "These groups describe the data — they were **not** given to the model as "
        "an input. They are computed using every row, including the rows held out "
        "for testing, so using them as a feature would inflate the scores above."
    )

    if clustering.get("warnings"):
        with st.expander("Clustering caveats"):
            for warning in clustering["warnings"]:
                st.markdown(f"- {warning}")

# ---- The report -------------------------------------------------------------

st.divider()
report, report_error = fetch("report")
if report is None:
    st.info(report_error)
else:
    st.download_button(
        "Download report (Markdown)",
        data=report["markdown"],
        file_name=f"autods_job_{job_id}_report.md",
        mime="text/markdown",
    )
    st.markdown(report["markdown"])
