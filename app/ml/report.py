"""The Markdown report -- deterministic in Section 5, LLM-written in Section 9.

Section 5's report exists to close the loop: a CSV goes in the browser and a
readable document comes out the other end, so the product is demoable end to end
(build-plan Section 5's "Done when"). It is assembled from the artifacts by plain
string formatting -- no LLM. The spec's Report Agent (7.12) is an LLM that writes
an executive summary over a much richer set of inputs (EDA, clustering, SHAP,
critic feedback); it arrives in Section 9 and replaces this function's prose
while reusing the same artifacts.

Writing it deterministically now is not a shortcut, it is the correct order: a
report that hallucinates a number is worse than no report, and until there is a
critic in place the safest generator is one that can only restate what the
artifacts say.

The report states its own methodology and its own limitations. Both matter for a
portfolio piece -- the first is the claim an examiner will probe (spec 8), and
the second stops a weak first slice from reading as an overclaim.
"""

from __future__ import annotations

from app.ml.contracts import (
    CleaningReport,
    ClusteringReport,
    EdaReport,
    EvaluationReport,
    ExplainabilityReport,
    FinalModelInfo,
    Leaderboard,
    PlannerPlan,
    PreprocessingSpec,
)

AGENT_NAME = "report"

# Metrics where a bigger number is better read as a proportion; used only to
# decide how many decimals look sensible, never to change a value.
_PROPORTION_METRICS = {
    "accuracy",
    "precision_macro",
    "recall_macro",
    "f1_macro",
    "roc_auc",
    "pr_auc",
    "r2",
}


def build_markdown_report(
    *,
    filename: str,
    plan: PlannerPlan,
    cleaning: CleaningReport,
    preprocessing: PreprocessingSpec,
    evaluation: EvaluationReport,
    eda: EdaReport | None = None,
    clustering: ClusteringReport | None = None,
    leaderboard: Leaderboard | None = None,
    final_model: FinalModelInfo | None = None,
    explainability: ExplainabilityReport | None = None,
    sampling_note: str = "",
) -> str:
    """Assemble the job's report from its artifacts.

    ``eda``, ``clustering``, ``leaderboard`` and the two Section 8 arguments are
    optional so the report still builds for a run whose EDA stage was skipped or
    whose model could not be explained -- those are descriptive, and a missing
    chart should not cost the reader the model results.
    """
    sections = [
        _heading(filename, evaluation),
        _headline(evaluation),
        _methodology(evaluation),
        _metrics_table(evaluation),
        _leaderboard_table(leaderboard),
        _folds_table(evaluation),
        _why_it_predicts(explainability),
        _data_quality(cleaning),
        _what_the_data_looks_like(eda),
        _groups(clustering),
        _preparation(plan, preprocessing, sampling_note),
        _served_model(final_model),
        _caveats(evaluation),
        _limitations(),
    ]
    return "\n\n".join(section.strip() for section in sections if section.strip()) + "\n"


def _what_the_data_looks_like(eda: EdaReport | None) -> str:
    """The descriptive findings a reader wants before trusting a score."""
    if eda is None:
        return ""
    lines = ["## What the data looks like", ""]

    if eda.class_balance:
        balance = eda.class_balance
        counts = ", ".join(f"`{k}` ({v:,})" for k, v in balance.counts.items())
        lines.append(f"- Target classes: {counts}")
        if balance.imbalanced:
            lines.append(
                f"- The classes are imbalanced ({balance.imbalance_ratio:.1f} to 1). "
                "Accuracy flatters a model on skewed data, which is why the "
                "headline metric above is macro F1 rather than accuracy."
            )

    if eda.top_correlations:
        lines.append("- Numeric columns that move together most strongly:")
        lines.extend(
            f"  - `{pair.left}` and `{pair.right}`: {pair.correlation:+.2f}"
            for pair in eda.top_correlations[:5]
        )

    outliers = {
        column.name: column.numeric.outlier_count
        for column in eda.columns
        if column.numeric and column.numeric.outlier_count
    }
    if outliers:
        listed = ", ".join(f"`{name}` ({count:,})" for name, count in sorted(outliers.items()))
        lines.append(
            f"- Unusually distant values were counted in {listed}. They were left "
            "in the data: an outlier is often the most interesting record in a "
            "dataset, so removing one is a decision to make deliberately."
        )

    if eda.plots:
        lines.append(f"- {_count(len(eda.plots), 'chart')} were produced; see the Results page.")

    return "\n".join(lines) if len(lines) > 2 else ""


def _groups(clustering: ClusteringReport | None) -> str:
    """The clustering findings, with the guardrail stated where a reader sees it."""
    if clustering is None or clustering.k == 0:
        return ""

    lines = [
        "## Natural groups in the data",
        "",
        f"Using {clustering.method}, the rows fall into "
        f"{_count(clustering.k, 'group')} (silhouette {clustering.silhouette:.2f}, "
        "where 1.0 would be perfectly separated and 0 no better than arbitrary).",
        "",
    ]

    for profile in clustering.profiles:
        headline = profile.description or "No notable differences from the dataset average."
        lines.append(
            f"- **Group {profile.cluster}** — {profile.size:,} rows "
            f"({profile.share:.0%}). {headline}"
        )
        for name, description in profile.distinguishing_features.items():
            lines.append(f"  - `{name}`: {description}")

    lines.append("")
    lines.append(
        "> These groups describe the data; they were **not** given to the model as "
        "an input. They are computed using every row, including the rows held out "
        "for testing, so feeding them to the model would inflate its scores the "
        "same way preparing the data up front would."
    )
    return "\n".join(lines)


def _heading(filename: str, evaluation: EvaluationReport) -> str:
    return (
        f"# Analysis of `{filename}`\n\n"
        f"- **Target column:** `{evaluation.target_column}`\n"
        f"- **Task:** {evaluation.task_type}\n"
        f"- **Rows modelled:** {evaluation.n_rows:,}\n"
        f"- **Feature columns:** {evaluation.n_features}\n"
        f"- **Model:** {evaluation.model_name}"
    )


def _headline(evaluation: EvaluationReport) -> str:
    score = evaluation.primary_score()
    if score is None:
        return "## Result\n\nNo metric could be computed for this dataset."

    summary = evaluation.metrics[evaluation.primary_metric]
    return (
        "## Result\n\n"
        f"**{_label(evaluation.primary_metric)}: {_fmt(evaluation.primary_metric, score)}** "
        f"(± {_fmt(evaluation.primary_metric, summary.std)} across "
        f"{evaluation.n_folds} folds)\n\n"
        f"{_spread_note(summary.std, evaluation.primary_metric)}"
    )


def _spread_note(std: float, metric: str) -> str:
    """Say what the fold-to-fold spread means, since that is the point of CV."""
    if metric not in _PROPORTION_METRICS:
        return "The ± figure is the spread across folds; smaller means more consistent."
    if std < 0.02:
        return "The score barely moved between folds, so this estimate is stable."
    if std < 0.08:
        return "There is moderate variation between folds, which is normal."
    return (
        "The score varies considerably between folds -- treat the average as a "
        "rough estimate. More data would tighten it."
    )


def _methodology(evaluation: EvaluationReport) -> str:
    """The leakage statement.

    Stated in the report itself, in plain English, because it is the difference
    between a score that means something and one that does not -- and because a
    reader should not have to open the source to find out whether the evaluation
    was done properly.
    """
    return (
        "## How this was validated\n\n"
        f"The dataset was split into {evaluation.n_folds} folds using "
        f"`{evaluation.cv_strategy}`. For each fold in turn, the *entire* "
        "preparation pipeline -- missing-value imputation, scaling and category "
        "encoding -- was fitted on that fold's training rows only, and then "
        "applied to the held-out rows before scoring. Nothing was imputed, scaled "
        "or encoded across the whole dataset beforehand.\n\n"
        "This matters: preparing the data up front would let information from the "
        "held-out rows influence the model's training, which inflates every score "
        "below without anything appearing to go wrong. The numbers here are "
        "measured on rows the model had never seen."
    )


def _metrics_table(evaluation: EvaluationReport) -> str:
    if not evaluation.metrics:
        return ""
    lines = [
        "## Metrics",
        "",
        "| Metric | Mean | Std. dev. across folds |",
        "| --- | --- | --- |",
    ]
    for name, summary in evaluation.metrics.items():
        lines.append(f"| {_label(name)} | {_fmt(name, summary.mean)} | {_fmt(name, summary.std)} |")
    return "\n".join(lines)


def _folds_table(evaluation: EvaluationReport) -> str:
    """Per-fold detail, including the row counts that make the split checkable.

    The train/test row counts are here so a reader can verify the split was real:
    each fold trains on roughly (k-1)/k of the data and is scored on the rest.
    """
    if not evaluation.folds:
        return ""
    primary = evaluation.primary_metric
    lines = [
        "## Folds",
        "",
        f"| Fold | Rows fitted on | Rows scored on | {_label(primary)} |",
        "| --- | --- | --- | --- |",
    ]
    for fold in evaluation.folds:
        value = fold.metrics.get(primary)
        shown = _fmt(primary, value) if value is not None else "—"
        lines.append(f"| {fold.fold} | {fold.n_train:,} | {fold.n_test:,} | {shown} |")
    return "\n".join(lines)


def _why_it_predicts(explainability: ExplainabilityReport | None) -> str:
    """The SHAP account, in the user's columns rather than the model's (spec 7.10).

    Placed directly after the scores on purpose. "How well does it do" and "why
    does it do it" are the two questions a reader has, in that order, and putting
    the explanation after the data-quality appendix would bury the answer to the
    second one.
    """
    if explainability is None:
        return ""
    if not explainability.global_importance:
        # An unexplainable model says so here rather than leaving a gap the
        # reader has to interpret.
        if not explainability.warnings:
            return ""
        stated = "\n".join(f"- {warning}" for warning in explainability.warnings)
        return f"## Why the model predicts what it does\n\n{stated}"

    lines = [
        "## Why the model predicts what it does",
        "",
        (
            f"Measured with SHAP ({explainability.explainer}) over "
            f"{explainability.n_rows_explained:,} rows. The model works in "
            f"{explainability.n_encoded_features:,} encoded features -- one-hot columns, "
            "calendar parts, frequency encodings -- and every one of them has been "
            "traced back to the column it came from, so the table below is in your "
            "column names and not the pipeline's."
        ),
        "",
        "| Column | Influence | Share | How it acts |",
        "| --- | ---: | ---: | --- |",
    ]
    for item in explainability.top_features(10):
        lines.append(
            f"| `{item.feature}` | {item.importance:.4f} | {item.share:.0%} | "
            f"{item.direction or '—'} |"
        )

    lines += [
        "",
        (
            "Influence is the mean absolute SHAP value: how far, on average, that "
            "column moved the model's output away from its baseline. It says "
            "**how much** a column matters, not whether the relationship is causal "
            "-- a column can be influential because it is a proxy for something "
            "the dataset does not contain."
        ),
    ]

    if explainability.examples:
        example = explainability.examples[0]
        confidence = (
            f" with {example.probability:.0%} confidence" if example.probability is not None else ""
        )
        # Said explicitly, because on a binary model the contributions push
        # towards the positive class whichever way the row came out -- a reader
        # of a negative row would otherwise read every sign backwards.
        direction = (
            f" The contributions below are pushes towards `{example.explained_class}`."
            if example.explained_class
            else ""
        )
        lines += [
            "",
            "### One prediction, in full",
            "",
            (
                f"Row {example.row_label} was predicted `{example.predicted}`{confidence}. "
                "SHAP decomposes that answer additively: start at the baseline the "
                "model gives an average row, then add what each column "
                f"contributed.{direction}"
            ),
            "",
            f"- Baseline: {example.base_value:+.4f}",
        ]
        lines += [
            f"- `{c.feature}` = {c.value}: {c.contribution:+.4f}" for c in example.contributions[:8]
        ]
        if example.other_contribution:
            lines.append(f"- Everything else, combined: {example.other_contribution:+.4f}")
        lines.append(f"- **Model output: {example.output_value:+.4f}**")

    if explainability.sampling_note:
        lines += ["", explainability.sampling_note]
    if explainability.warnings:
        lines += [""] + [f"- {warning}" for warning in explainability.warnings]

    return "\n".join(lines)


def _served_model(final: FinalModelInfo | None) -> str:
    """What is actually saved, and the sentence that keeps its score honest."""
    if final is None:
        return ""

    lines = [
        "## The model that gets served",
        "",
        (
            f"**{final.model_name}**, refitted on all {final.n_rows:,} rows of the "
            f"cleaned dataset and saved as `{final.artifact or 'final_model.pkl'}`. "
            "This is the only estimator in the run fitted on every row, and it is "
            "the one a live prediction request loads."
        ),
    ]

    if final.primary_metric and final.cv_score is not None:
        lines += [
            "",
            (
                f"Its {_label(final.primary_metric)} is not measured again here. The "
                f"{_fmt(final.primary_metric, final.cv_score)} reported above is the "
                "cross-validated estimate for this configuration, and a model fitted "
                "on every row has no unseen data left to score itself against -- "
                "which is precisely why the cross-validation came first."
            ),
        ]

    if final.resampling and final.resampling != "none":
        lines += ["", f"Resampling: {final.resampling}."]
    if final.warnings:
        lines += [""] + [f"- {warning}" for warning in final.warnings]

    return "\n".join(lines)


def _data_quality(cleaning: CleaningReport) -> str:
    lines = [
        "## Data quality",
        "",
        f"- Rows: {cleaning.n_rows_before:,} → {cleaning.n_rows_after:,}",
        f"- Columns: {cleaning.n_columns_before} → {cleaning.n_columns_after}",
    ]
    if cleaning.duplicate_rows_removed:
        lines.append(
            f"- Removed {_count(cleaning.duplicate_rows_removed, 'exactly-duplicated row')}"
        )
    if cleaning.missing_target_rows_removed:
        lines.append(
            f"- Removed {_count(cleaning.missing_target_rows_removed, 'row')} with no target value"
        )
    if cleaning.dtype_corrections:
        lines.append("- Corrected column types:")
        lines.extend(
            f"  - `{c.name}`: {c.from_dtype} → {c.to_dtype}" for c in cleaning.dtype_corrections
        )
    if cleaning.dropped_columns:
        lines.append("- Dropped columns:")
        lines.extend(f"  - `{c.name}` — {c.reason}" for c in cleaning.dropped_columns)

    gaps = cleaning.missing_values_left_to_the_pipeline
    if gaps:
        total = sum(gaps.values())
        listed = ", ".join(f"`{name}` ({count:,})" for name, count in sorted(gaps.items()))
        was = "was" if total == 1 else "were"
        lines.append(
            f"- {_count(total, 'missing value')} {was} left in place, in {listed}. "
            "Gaps are filled inside each cross-validation fold rather than "
            "beforehand, which is what keeps the scores above honest."
        )
    else:
        lines.append("- No missing values remained after cleaning.")
    return "\n".join(lines)


def _leaderboard_table(leaderboard: Leaderboard | None) -> str:
    """Every candidate that was tried, ranked, with the spread beside the score.

    The spread is in the table rather than a footnote because it is what decides
    whether the ranking is worth acting on: a 0.02 lead between two models whose
    folds swing 0.09 is not a lead, and a table of means alone hides that.
    """
    if leaderboard is None or not leaderboard.entries:
        return ""

    metric = _label(leaderboard.primary_metric)
    lines = [
        "## Models compared",
        "",
        f"| Rank | Model | {metric} | Spread across folds | Time |",
        "| --- | --- | --- | --- | --- |",
    ]
    for entry in leaderboard.entries:
        if entry.error:
            lines.append(f"| {entry.rank} | {entry.model_name} | — | — | could not be trained |")
            continue
        lines.append(
            f"| {entry.rank} | {entry.model_name} | {entry.score:.3f} | "
            f"± {entry.std:.3f} | {entry.fit_seconds:.1f}s |"
        )

    lines.append("")
    lines.append(
        f"All {len(leaderboard.entries)} models were scored on the same "
        f"{leaderboard.n_folds} folds, so the ranking compares like with like."
    )
    if leaderboard.resampling != "none":
        lines.append(f"Class balancing: {leaderboard.resampling}.")
    return "\n".join(lines)


def _preparation(
    plan: PlannerPlan, preprocessing: PreprocessingSpec, sampling_note: str = ""
) -> str:
    lines = ["## How the data was prepared", ""]

    if preprocessing.numeric_columns:
        lines.append(
            f"- **{_count(len(preprocessing.numeric_columns), 'numeric column')}**: "
            f"{preprocessing.numeric_strategy}"
        )
    if preprocessing.categorical_columns:
        lines.append(
            f"- **{_count(len(preprocessing.categorical_columns), 'categorical column')}**: "
            f"{preprocessing.categorical_strategy}"
        )
    if preprocessing.unhandled_columns:
        lines.append("- Left out of the model for now:")
        lines.extend(f"  - `{c.name}` — {c.reason}" for c in preprocessing.unhandled_columns)

    if preprocessing.ordinal_columns:
        lines.append(
            f"- **{_count(len(preprocessing.ordinal_columns), 'ordered column')}**: "
            "encoded in rank order rather than as unrelated labels"
        )
    if preprocessing.datetime_columns:
        lines.append(
            f"- **{_count(len(preprocessing.datetime_columns), 'date column')}**: "
            "split into calendar features (year, month, day, weekday, hour)"
        )
    if preprocessing.text_columns:
        lines.append(
            f"- **{_count(len(preprocessing.text_columns), 'high-variety column')}**: "
            "replaced with how often each value occurs, learned per fold"
        )
    if preprocessing.feature_selection:
        lines.append(f"- **Feature selection**: {preprocessing.feature_selection}")

    # What the pipeline decided *not* to do. A run that silently does less is an
    # optimisation; one that says which steps it routed around is a decision the
    # reader can disagree with (spec 11).
    turned_off = [
        label
        for label, on in (
            ("oversampling the rare outcome", plan.use_smote),
            ("selecting a subset of features", plan.run_feature_selection),
            ("training on a sample of the rows", plan.run_sampling),
        )
        if not on
    ]
    if turned_off:
        lines.append("")
        lines.append(f"Steps not used on this dataset: {', '.join(turned_off)}.")
    if sampling_note:
        lines.append("")
        lines.append(sampling_note)

    # Both halves have to read as a sentence. The LLM branch had never actually
    # run before Section 7 was tested against a live model, and produced
    # "came from chosen by the planning model".
    source = (
        "were chosen by the planning model"
        if plan.source == "llm"
        else "came from the built-in defaults"
    )
    lines.append("")
    lines.append(f"Preparation steps {source}.")
    if preprocessing.strategy_source == "llm":
        lines.append(
            "Per-column preparation was chosen by the feature strategy model and "
            "checked against the real columns before being built."
        )
    if plan.rationale:
        lines.append(f"> {plan.rationale}")
    return "\n".join(lines)


def _caveats(evaluation: EvaluationReport) -> str:
    if not evaluation.warnings:
        return ""
    lines = ["## Caveats", ""]
    lines.extend(f"- {warning}" for warning in evaluation.warnings)
    return "\n".join(lines)


def _limitations() -> str:
    """State what this version does not do, so a weak slice is not read as a strong one."""
    return (
        "## What this version does not do yet\n\n"
        "This is the first end-to-end version of the pipeline. It trains a single "
        "model with a fixed preparation strategy, so treat the score as a baseline "
        "rather than the best achievable result. Exploratory analysis and plots, "
        "model comparison across several algorithms, class-imbalance resampling, "
        "learned feature engineering, and per-prediction explanations are all still "
        "to come."
    )


def _count(n: int, noun: str) -> str:
    """ "1 row" / "12 rows" -- the report is prose, so it should read like prose.

    Cheap, but "1 missing value(s)" in the one document a human actually reads
    undercuts the care taken everywhere else.
    """
    return f"{n:,} {noun}" if n == 1 else f"{n:,} {noun}s"


def _label(metric: str) -> str:
    """Human-readable metric names, so the report does not read like a log line."""
    return {
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
    }.get(metric, metric)


def _fmt(metric: str, value: float) -> str:
    """Format a metric for reading, not for round-tripping.

    Proportions get four decimals; regression errors are in the target's own
    units and can be enormous, so those get thousands separators instead of a
    long decimal tail nobody reads.
    """
    if metric in _PROPORTION_METRICS or abs(value) < 1000:
        return f"{value:.4f}"
    return f"{value:,.2f}"
