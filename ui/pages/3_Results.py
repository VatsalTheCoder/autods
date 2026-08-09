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

# Streamlit runs each page as its own script, so a page cannot import a sibling
# module without ui/ being importable. The path work has to happen before the
# import, which is what the noqa below is for.
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

sys.path.append(str(Path(__file__).resolve().parents[1]))

from auth import require_access  # noqa: E402

require_access()

st.title("Results")

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

# ---- The leaderboard --------------------------------------------------------

leaderboard, _ = fetch("leaderboard")
if leaderboard and leaderboard.get("entries"):
    st.subheader("Models compared")
    st.caption(
        f"All {len(leaderboard['entries'])} models were scored on the same "
        f"{leaderboard['n_folds']} folds, so the ranking compares like with like. "
        "The spread matters as much as the score: a small lead between models "
        "whose folds swing widely is not really a lead."
    )
    # The winner is the best *real* model, which is not always rank 1: the
    # featureless baseline is ranked with everything else and can top the board.
    # Labelling rank 1 "winner" would then point at a model that was never served.
    served = next(
        (e["rank"] for e in leaderboard["entries"] if not e["is_baseline"] and not e["error"]),
        None,
    )
    st.dataframe(
        [
            {
                "Rank": entry["rank"],
                "Model": entry["model_name"],
                LABEL.get(entry["primary_metric"], entry["primary_metric"]): (
                    "—" if entry["error"] else round(entry["score"], 4)
                ),
                "Spread across folds": "—" if entry["error"] else round(entry["std"], 4),
                "Time": f"{entry['fit_seconds']:.1f}s",
                "Note": entry["error"]
                or (
                    "baseline — not served"
                    if entry["is_baseline"]
                    else ("winner" if entry["rank"] == served else "")
                ),
            }
            for entry in leaderboard["entries"]
        ],
        hide_index=True,
        width="stretch",
    )
    if any(e["is_baseline"] for e in leaderboard["entries"]):
        st.caption(
            "The baseline ignores every feature and always predicts the same answer. "
            "It is there to show what the other scores are worth — a model that cannot "
            "beat it has learned nothing from the data, however its score reads on its own."
        )
    # An absent or empty value means the same as "none" here; without the
    # emptiness check the page renders a bare "Class balancing: ."
    resampling = (leaderboard.get("resampling") or "none").strip()
    if resampling != "none":
        st.caption(f"Class balancing: {resampling}.")
    for warning in leaderboard.get("warnings", []):
        st.warning(warning)

# ---- How each column was prepared -------------------------------------------

features, _ = fetch("features")
if features and features.get("columns"):
    chosen = "the strategy model" if features["source"] == "llm" else "the built-in defaults"
    with st.expander(f"How each column was prepared — chosen by {chosen}", expanded=False):
        st.dataframe(
            [
                {
                    "Column": column["column"],
                    "Treated as": column["role"],
                    "Blanks filled with": column["impute"],
                    "Encoded": column["encode"],
                    "Scaled": column["scale"],
                }
                for column in features["columns"]
            ],
            hide_index=True,
            width="stretch",
        )

        # The point of showing this at all: the model proposed, and code checked.
        # A run where nothing was overruled should look different from one where
        # something was, so both are stated rather than only the interesting case.
        if features.get("rejected_columns"):
            st.warning(
                "Columns the model invented, which were rejected: "
                + ", ".join(f"`{name}`" for name in features["rejected_columns"])
            )
        if features.get("overrides"):
            st.caption("Choices the code overruled, because the data could not support them:")
            st.dataframe(
                [
                    {
                        "Column": o["column"],
                        "Setting": o["field"],
                        "Asked for": o["requested"],
                        "Used instead": o["applied"],
                        "Why": o["reason"],
                    }
                    for o in features["overrides"]
                ],
                hide_index=True,
                width="stretch",
            )
        elif features["source"] == "llm":
            st.caption(
                "Every choice the model made was buildable as stated; nothing was overruled."
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

# ---- Why the model predicts what it does ------------------------------------
# Directly after the scores, because "how well does it do" and "why does it do
# it" are the two questions a reader has, in that order (Section 8).

explainability, _ = fetch("explainability")
if explainability and explainability.get("global_importance"):
    st.divider()
    st.subheader("Why the model predicts what it does")
    st.caption(
        f"SHAP ({explainability['explainer']}) over "
        f"{explainability['n_rows_explained']:,} rows. The model works in "
        f"{explainability['n_encoded_features']:,} encoded features; every one has been "
        "traced back to the column it came from, so what follows is in your column names."
    )

    chart_column, table_column = st.columns([3, 2])
    if "shap_importance.png" in explainability.get("plots", []):
        chart_column.image(f"{API_BASE_URL}/jobs/{job_id}/artifacts/shap_importance.png/content")
    table_column.dataframe(
        [
            {
                "Column": item["feature"],
                "Influence": round(item["importance"], 4),
                "Share": f"{item['share']:.0%}",
                "How it acts": item["direction"] or "—",
            }
            for item in explainability["global_importance"][:12]
        ],
        hide_index=True,
        width="stretch",
    )

    for name in ("shap_summary.png", "shap_dependence.png"):
        if name in explainability.get("plots", []):
            st.image(f"{API_BASE_URL}/jobs/{job_id}/artifacts/{name}/content")

    examples = explainability.get("examples", [])
    if examples:
        st.markdown("**Individual predictions, explained**")
        st.caption(
            "SHAP is additive: the baseline plus every column's contribution equals "
            "the model's output for that row. That is what makes these an account of "
            "the prediction rather than an illustration of it."
        )
        for index, example in enumerate(examples, start=1):
            confidence = (
                f" · {example['probability']:.0%} confidence"
                if example.get("probability") is not None
                else ""
            )
            with st.expander(f"Row {example['row_label']} → {example['predicted']}{confidence}"):
                chart = f"shap_explanation_{index}.png"
                if chart in explainability.get("plots", []):
                    st.image(f"{API_BASE_URL}/jobs/{job_id}/artifacts/{chart}/content")
                st.dataframe(
                    [
                        {
                            "Column": c["feature"],
                            "Value": c["value"],
                            "Pushed the prediction by": round(c["contribution"], 4),
                        }
                        for c in example["contributions"]
                    ],
                    hide_index=True,
                    width="stretch",
                )
                if example.get("explained_class"):
                    # On a binary model the contributions push towards the
                    # positive class whichever way the row came out, so the
                    # direction has to be named or the signs read backwards.
                    st.caption(
                        "Positive bars push towards "
                        f"`{evaluation['target_column']} = {example['explained_class']}`."
                    )
                st.caption(
                    f"baseline {example['base_value']:+.4f} + contributions "
                    f"= {example['output_value']:+.4f}"
                )

    with st.expander("How encoded features map back to your columns"):
        st.caption(
            "The translation the section above rests on. It is published so it can "
            "be checked rather than trusted — a mislabelled feature is worse than a "
            "missing one, because it gets believed."
        )
        st.dataframe(
            [
                {"Encoded feature": encoded, "Came from": origin}
                for encoded, origin in explainability.get("feature_name_mapping", {}).items()
            ],
            hide_index=True,
            width="stretch",
        )

    if explainability.get("warnings"):
        with st.expander("Explainability caveats"):
            for warning in explainability["warnings"]:
                st.markdown(f"- {warning}")
elif explainability and explainability.get("warnings"):
    st.divider()
    st.subheader("Why the model predicts what it does")
    for warning in explainability["warnings"]:
        st.warning(warning)

# ---- Try a prediction -------------------------------------------------------

model_info, _ = fetch("model")
if model_info and model_info.get("feature_columns"):
    st.divider()
    st.subheader("Try a prediction")
    st.caption(
        f"**{model_info['model_name']}**, refitted on all {model_info['n_rows']:,} rows and "
        "saved to object storage. The form below sends your columns to the API, which "
        "loads that saved pipeline and runs it — preprocessing included, exactly as it "
        "was fitted."
    )

    with st.form("predict"):
        supplied: dict[str, str] = {}
        # Only the columns the recipe consumes. A column excluded at the
        # checkpoint is still part of the model's input frame, but asking for a
        # customer ID the model discards would imply it matters.
        columns = [column for column in model_info["feature_columns"] if column.get("used", True)]
        # Three across: a wide dataset otherwise produces a form the length of
        # the page. Pre-filled with a real value from the training data so the
        # units are never a guess.
        for start in range(0, len(columns), 3):
            for slot, column in zip(st.columns(3), columns[start : start + 3], strict=False):
                supplied[column["name"]] = slot.text_input(
                    column["name"],
                    value=column.get("example", ""),
                    help=f"{column['role']} · {column['dtype']}",
                )
        submitted = st.form_submit_button("Predict")

    if submitted:
        rows = [{name: value for name, value in supplied.items() if value != ""}]
        try:
            response = requests.post(
                f"{API_BASE_URL}/jobs/{job_id}/predict", json={"rows": rows}, timeout=30
            )
        except requests.exceptions.RequestException as exc:
            st.error(f"Could not reach the API: {exc}")
        else:
            if response.status_code != 200:
                st.error(response.json().get("detail", f"HTTP {response.status_code}"))
            else:
                payload = response.json()
                result = payload["predictions"][0]
                st.success(f"**{payload['target_column']} = {result['prediction']}**")
                if result.get("probabilities"):
                    st.bar_chart(result["probabilities"], horizontal=True)
                if payload.get("missing_columns"):
                    st.info(
                        "Left blank and filled in by the pipeline's imputers: "
                        + ", ".join(f"`{name}`" for name in payload["missing_columns"])
                    )
                if payload.get("unexpected_columns"):
                    st.warning(
                        "Not columns this model knows, so they were ignored: "
                        + ", ".join(f"`{name}`" for name in payload["unexpected_columns"])
                    )

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

# ---- What the review found --------------------------------------------------

critic, _ = fetch("critic")
if critic and (critic.get("findings") or critic.get("verdict")):
    st.subheader("What the review found")
    if critic.get("verdict"):
        st.markdown(critic["verdict"])

    if critic.get("source") == "default":
        st.caption(
            "These are automated threshold checks rather than a written review — "
            "no language model was available for this run."
        )

    if critic.get("strengths"):
        with st.expander("Held up well"):
            for item in critic["strengths"]:
                st.markdown(f"- {item}")

    if critic.get("findings"):
        # Worst first. A review sorted by the order checks happened to run in
        # buries the blocker under three notes.
        rank = {"blocker": 0, "concern": 1, "note": 2}
        findings = sorted(critic["findings"], key=lambda f: rank.get(f["severity"], 3))
        st.dataframe(
            [
                {
                    "Severity": f["severity"],
                    "Area": f["area"],
                    "Finding": f["finding"],
                    "Recommendation": f["recommendation"] or "—",
                    # A threshold result and a model's opinion are both worth
                    # reading and are not worth the same.
                    "Source": "measured" if f.get("measured") else "review",
                }
                for f in findings
            ],
            hide_index=True,
            width="stretch",
        )

    if critic.get("recommended_next_steps"):
        st.markdown("**The reviewer suggests:**")
        for item in critic["recommended_next_steps"]:
            st.markdown(f"- {item}")

    if critic.get("omissions"):
        with st.expander("What the review did not see"):
            st.caption(
                "The reviewer reads a summary sized to fit the model's token limit. "
                "On a wide dataset that summary is capped, and a critique of part of "
                "a run should say so."
            )
            for note in critic["omissions"]:
                st.markdown(f"- {note}")

    st.divider()

# ---- The report -------------------------------------------------------------

report, report_error = fetch("report")
if report is None:
    st.info(report_error)
else:
    markdown_column, pdf_column = st.columns(2)
    markdown_column.download_button(
        "Download report (Markdown)",
        data=report["markdown"],
        file_name=f"autods_job_{job_id}_report.md",
        mime="text/markdown",
        width="stretch",
    )

    # Fetched rather than linked: a download_button needs the bytes, and the
    # PDF is best-effort, so its absence has to degrade to a caption rather
    # than a dead button.
    try:
        pdf = requests.get(f"{API_BASE_URL}/jobs/{job_id}/report/pdf", timeout=30)
    except requests.exceptions.RequestException:
        pdf = None
    if pdf is not None and pdf.status_code == 200:
        pdf_column.download_button(
            "Download report (PDF)",
            data=pdf.content,
            file_name=f"autods_job_{job_id}_report.pdf",
            mime="application/pdf",
            type="primary",
            width="stretch",
        )
    else:
        pdf_column.caption(
            "No PDF for this run. The Markdown report beside it is the authoritative version."
        )

    st.markdown(report["markdown"])
