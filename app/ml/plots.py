"""Chart rendering -- headless matplotlib to PNG bytes (Section 6).

Pure functions: a DataFrame in, ``Chart`` objects out. Nothing here touches S3 or
the database -- the graph node stores what this returns -- which keeps every chart
testable by rendering it and inspecting the bytes.

**Matplotlib only, no seaborn.** The chart set is small and fixed, so seaborn
would add a dependency and a second styling system to earn nothing.

**The colour choices are computed, not taste.** Each chart's palette follows the
job its colour does:

* *Single-series* charts (histograms, boxplots, target counts, missingness) use
  one hue. There is nothing to tell apart, so a second colour would imply a
  distinction that does not exist.
* The *correlation heatmap* encodes **polarity** -- correlation runs -1 to +1
  through a meaningful zero -- so it uses a diverging blue↔red ramp with a neutral
  grey midpoint. A rainbow, or a one-hue ramp, would hide the sign.
* The *cluster scatter* encodes **identity**, so it uses a categorical palette
  whose colours were validated for colour-vision deficiency at every pair, not
  just adjacent ones (a scatter puts all groups on screen at once). Only four
  hues clear that bar, so past four clusters identity is carried by hue × marker
  shape, and every cluster is labelled at its centroid regardless -- colour is
  never the only thing telling two groups apart.

Charts render on an explicit light surface rather than a transparent background:
a transparent PNG with dark axis text is unreadable if the page around it is
dark, and rendering every chart twice to theme it is not worth the complexity for
static images.
"""

from __future__ import annotations

import io
import logging
import math
from dataclasses import dataclass

import matplotlib

# Must precede the pyplot import: the worker has no display, and the default
# backend would try to find one.
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402 - deliberately after use("Agg")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from app.agents.schema_models import TaskType  # noqa: E402
from app.core.config import get_settings  # noqa: E402
from app.ml.statistics import heatmap_columns  # noqa: E402

logger = logging.getLogger(__name__)

# ---- The palette ------------------------------------------------------------
# Validated with the data-viz palette checker. The four categorical hues are the
# largest subset that passes CVD and normal-vision separation across *every*
# pair, which is the bar a scatter has to clear; five hues fail it.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_MUTED = "#52514e"
GRID = "#dcdbd6"

SERIES = "#2a78d6"  # single-hue default (blue)
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7"]  # blue, orange, aqua, violet
# Used when there are more clusters than validated hues -- see the module note.
SHAPES = ["o", "^", "s", "D"]

# Diverging: blue ↔ red through a neutral grey. Equal steps per arm, so a
# correlation of -0.4 and +0.4 are equally strong-looking in opposite directions.
DIVERGING = LinearSegmentedColormap.from_list(
    "autods_diverging",
    ["#184f95", "#5598e7", "#f0efec", "#ea7f7e", "#b52c2c"],
)

# Sequential: one hue, light to dark, for a quantity with no meaningful zero --
# the SHAP beeswarm colours points by how high a feature's value was (Section 8).
# The diverging ramp would be wrong there: its neutral midpoint reads as a
# boundary, and "average" is not one.
SEQUENTIAL = LinearSegmentedColormap.from_list(
    "autods_sequential",
    ["#dbe7f7", "#5598e7", "#184f95"],
)

# The two ends of the diverging ramp, used on their own where the quantity is
# purely directional: a SHAP contribution either raises or lowers the prediction.
POSITIVE = "#b52c2c"
NEGATIVE = "#184f95"


@dataclass(slots=True)
class Chart:
    """A rendered chart: the artifact name it will be stored under, and its bytes."""

    name: str
    title: str
    png: bytes


def render_charts(
    frame: pd.DataFrame,
    *,
    target: str,
    task_type: TaskType,
) -> list[Chart]:
    """Render the standard EDA chart set for a cleaned dataset.

    Charts that do not apply are skipped rather than emitted empty -- a dataset
    with no missing values should not produce a blank "missing values" chart.
    """
    charts: list[Chart] = []
    for build in (
        lambda: _target_distribution(frame, target, task_type),
        lambda: _numeric_distributions(frame, target),
        lambda: _numeric_boxplots(frame, target),
        lambda: _correlation_heatmap(frame),
        lambda: _missingness(frame),
    ):
        try:
            chart = build()
        except Exception:  # pragma: no cover - a chart must never fail the job
            # EDA is descriptive. A dataset that breaks one renderer should still
            # get the other charts and, more importantly, still get modelled.
            logger.exception("A chart failed to render; continuing without it")
            continue
        if chart is not None:
            charts.append(chart)

    logger.info("Rendered %d chart(s)", len(charts))
    return charts


# ---- Individual charts ------------------------------------------------------


def _target_distribution(frame: pd.DataFrame, target: str, task_type: TaskType) -> Chart | None:
    """What we are predicting, and how it is distributed.

    The most important chart in the set: it is where an imbalanced target becomes
    visible, and imbalance explains more surprising scores than anything else.
    """
    if target not in frame.columns:
        return None
    values = frame[target].dropna()
    if values.empty:
        return None

    if task_type == "classification":
        counts = values.astype(str).value_counts().sort_values(ascending=True)
        # Height follows the category count, so a two-class target gets two
        # readable bars rather than two slabs filling a fixed-height figure.
        fig, ax = _figure(height=max(2.2, 0.42 * len(counts) + 1.3))
        positions = np.arange(len(counts))
        ax.barh(positions, counts.to_numpy(), color=SERIES, height=0.65)
        ax.set_yticks(positions, [_truncate(str(i)) for i in counts.index])
        ax.set_xlabel("Rows", color=INK_MUTED, fontsize=9)
        # Direct labels: with a handful of bars the count belongs on the bar, not
        # on an axis the reader has to trace back to.
        span = float(counts.max()) or 1.0
        for pos, value in zip(positions, counts.to_numpy(), strict=True):
            ax.text(value + span * 0.01, pos, f"{int(value):,}", va="center", fontsize=8, color=INK)
        ax.set_xlim(0, span * 1.12)
    else:
        fig, ax = _figure()
        ax.hist(values.to_numpy(), bins=_bin_count(len(values)), color=SERIES)
        ax.set_xlabel(_truncate(target), color=INK_MUTED, fontsize=9)
        ax.set_ylabel("Rows", color=INK_MUTED, fontsize=9)

    _style(
        ax,
        f"Distribution of {_truncate(target)}",
        grid_axis="x" if task_type == "classification" else "y",
    )
    return _finish(fig, "target_distribution.png", f"Distribution of {target}")


def _numeric_distributions(frame: pd.DataFrame, target: str) -> Chart | None:
    """Histograms as small multiples -- one panel per column, own x scale.

    Small multiples rather than one overlaid axes: income and age share no
    meaningful scale, and plotting them together would make one of them a flat
    line against the other's range.
    """
    columns = _numeric_feature_columns(frame, target)
    if not columns:
        return None

    fig, axes = _grid(len(columns))
    for ax, name in zip(axes, columns, strict=False):
        values = frame[name].dropna().to_numpy()
        ax.hist(values, bins=_bin_count(len(values)), color=SERIES)
        _style(ax, _truncate(name, 28), grid_axis="y", small=True)
    _blank_unused(axes, len(columns))

    fig.suptitle("Numeric distributions", color=INK, fontsize=12, y=0.995)
    return _finish(fig, "numeric_distributions.png", "Numeric distributions")


def _numeric_boxplots(frame: pd.DataFrame, target: str) -> Chart | None:
    """Boxplots as small multiples, for the same scale reason as the histograms.

    A boxplot answers a different question from a histogram -- where the middle
    half sits and what lies beyond the whiskers -- which is why both are drawn
    rather than picking one.
    """
    columns = _numeric_feature_columns(frame, target)
    if not columns:
        return None

    fig, axes = _grid(len(columns))
    for ax, name in zip(axes, columns, strict=False):
        ax.boxplot(
            frame[name].dropna().to_numpy(),
            vert=True,
            widths=0.5,
            patch_artist=True,
            boxprops={"facecolor": SERIES, "edgecolor": SERIES, "linewidth": 1.2},
            medianprops={"color": SURFACE, "linewidth": 1.6},
            whiskerprops={"color": INK_MUTED, "linewidth": 1.2},
            capprops={"color": INK_MUTED, "linewidth": 1.2},
            flierprops={
                "marker": "o",
                "markersize": 3,
                "markerfacecolor": INK_MUTED,
                "markeredgecolor": "none",
                "alpha": 0.5,
            },
        )
        ax.set_xticks([])
        _style(ax, _truncate(name, 28), grid_axis="y", small=True)
    _blank_unused(axes, len(columns))

    fig.suptitle("Numeric spread and outliers", color=INK, fontsize=12, y=0.995)
    return _finish(fig, "numeric_boxplots.png", "Numeric spread and outliers")


def _correlation_heatmap(frame: pd.DataFrame) -> Chart | None:
    """Correlation as a diverging map, fixed to -1..+1.

    The scale is pinned rather than fitted to the data: a matrix whose strongest
    correlation is 0.2 would otherwise render that 0.2 in full saturation and
    imply a relationship that is not there.
    """
    columns = heatmap_columns(frame)
    if len(columns) < 2:
        return None

    matrix = frame[columns].corr(numeric_only=True)
    size = max(4.0, min(11.0, 0.55 * len(columns) + 2.2))
    fig, ax = plt.subplots(figsize=(size, size * 0.85), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)

    # Show the lower triangle only. The diagonal is always 1.0 -- a column
    # correlates perfectly with itself -- and would otherwise be the strongest
    # colour on the chart while saying nothing; the upper triangle is its mirror.
    # Dropping both halves the ink and lets the real correlations be the loudest
    # thing on the plot.
    #
    # The first row and last column are dropped with them: after masking they
    # hold no cells at all, and an axis label pointing at empty space reads as a
    # rendering fault rather than a deliberate omission.
    panel = matrix.iloc[1:, :-1]
    values = np.ma.masked_array(
        panel.to_numpy(), mask=np.triu(np.ones_like(panel, dtype=bool), k=1)
    )
    image = ax.imshow(values, cmap=DIVERGING, vmin=-1.0, vmax=1.0)

    ax.set_xticks(
        range(panel.shape[1]),
        [_truncate(str(c), 16) for c in panel.columns],
        rotation=45,
        ha="right",
        fontsize=8,
        color=INK_MUTED,
    )
    ax.set_yticks(
        range(panel.shape[0]),
        [_truncate(str(i), 16) for i in panel.index],
        fontsize=8,
        color=INK_MUTED,
    )

    # Annotate only when the cells are big enough to hold a number legibly.
    if len(columns) <= 12:
        for i in range(panel.shape[0]):
            for j in range(i + 1):  # lower triangle only, matching the mask
                value = panel.iat[i, j]
                if pd.isna(value):
                    continue
                # Ink colour follows cell darkness, not the series hue.
                shade = SURFACE if abs(value) > 0.55 else INK
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=7, color=shade)

    bar = fig.colorbar(image, ax=ax, shrink=0.75)
    bar.ax.tick_params(labelsize=8, colors=INK_MUTED)
    bar.outline.set_visible(False)
    bar.set_label("Pearson correlation", color=INK_MUTED, fontsize=9)

    ax.set_title("How numeric columns move together", color=INK, fontsize=12, pad=12)
    for spine in ax.spines.values():
        spine.set_visible(False)
    return _finish(fig, "correlation_heatmap.png", "Correlation heatmap")


def _missingness(frame: pd.DataFrame) -> Chart | None:
    """Which columns have gaps -- omitted entirely when there are none."""
    missing = frame.isna().sum()
    missing = missing[missing > 0].sort_values(ascending=True)
    if missing.empty:
        return None

    fig, ax = _figure(height=max(2.4, 0.36 * len(missing) + 1.4))
    positions = np.arange(len(missing))
    ax.barh(positions, missing.to_numpy(), color=SERIES, height=0.6)
    ax.set_yticks(positions, [_truncate(str(i), 24) for i in missing.index])
    ax.set_xlabel("Missing values", color=INK_MUTED, fontsize=9)

    total = len(frame)
    span = float(missing.max()) or 1.0
    for pos, value in zip(positions, missing.to_numpy(), strict=True):
        share = value / total if total else 0.0
        ax.text(
            value + span * 0.01,
            pos,
            f"{int(value):,} ({share:.0%})",
            va="center",
            fontsize=8,
            color=INK,
        )
    ax.set_xlim(0, span * 1.2)

    _style(ax, "Missing values by column", grid_axis="x")
    return _finish(fig, "missing_values.png", "Missing values by column")


def render_cluster_scatter(
    coordinates: np.ndarray,
    labels: np.ndarray,
    *,
    explained_variance: tuple[float, float] | None = None,
) -> Chart | None:
    """The PCA scatter: every row placed in two dimensions, coloured by group.

    Identity here is carried by three channels at once, on purpose. Only four
    hues clear colour-vision separation at *every* pair, which is the bar a
    scatter has to meet because all groups share one plot -- so past four
    clusters, hue is combined with marker shape (the documented composite
    encoding), and every cluster is labelled at its own centroid regardless.
    A reader never has to distinguish two groups by colour alone.

    The axes are unlabelled beyond their variance share by design: principal
    components have no units and no meaning to name. What the chart is for is
    whether the groups separate at all.
    """
    if coordinates.size == 0 or len(labels) == 0:
        return None

    unique = sorted({int(v) for v in labels})
    fig, ax = _figure(height=5.0)

    for index, cluster in enumerate(unique):
        mask = labels == cluster
        ax.scatter(
            coordinates[mask, 0],
            coordinates[mask, 1],
            s=26,
            c=CATEGORICAL[index % len(CATEGORICAL)],
            marker=SHAPES[(index // len(CATEGORICAL)) % len(SHAPES)],
            # A thin surface-coloured ring keeps overlapping points readable
            # instead of merging into one dense blob.
            linewidths=0.6,
            edgecolors=SURFACE,
            alpha=0.85,
            label=f"Cluster {cluster}",
        )

    # Direct labels at each centroid -- the reason colour never has to work alone.
    for cluster in unique:
        mask = labels == cluster
        cx = float(np.median(coordinates[mask, 0]))
        cy = float(np.median(coordinates[mask, 1]))
        ax.text(
            cx,
            cy,
            str(cluster),
            fontsize=11,
            fontweight="bold",
            ha="center",
            va="center",
            color=INK,
            bbox={
                "boxstyle": "circle,pad=0.28",
                "facecolor": SURFACE,
                "edgecolor": INK_MUTED,
                "linewidth": 0.8,
                "alpha": 0.9,
            },
        )

    if explained_variance:
        ax.set_xlabel(
            f"Component 1 ({explained_variance[0]:.0%} of variance)", color=INK_MUTED, fontsize=9
        )
        ax.set_ylabel(
            f"Component 2 ({explained_variance[1]:.0%} of variance)", color=INK_MUTED, fontsize=9
        )

    _style(ax, f"{len(unique)} groups found in the data", grid_axis="both")
    # A legend as well as the centroid labels: identity is never colour-alone,
    # and with more than four groups the shape is part of the key.
    ax.legend(frameon=False, fontsize=8, labelcolor=INK_MUTED, loc="best", handletextpad=0.4)
    return _finish(fig, "cluster_scatter.png", "Clusters in two dimensions")


# ---- SHAP (Section 8) -------------------------------------------------------
#
# Four charts, and the palette follows the same rule as everything above -- what
# the colour is *for* decides which ramp it gets:
#
# * The importance bars encode **magnitude only**, so one hue.
# * The beeswarm colours points by the feature's own value, which is a quantity
#   with no meaningful zero once it has been scaled -- a sequential ramp, not the
#   diverging one, or the midpoint would imply a boundary that is not there.
# * A waterfall's bars encode **polarity**: a contribution either raises or lowers
#   the prediction. That is exactly what the diverging palette's two arms are for,
#   and the two colours are its endpoints rather than a third scheme.
#
# Rendered here rather than with ``shap.summary_plot`` deliberately. SHAP's own
# plotting is built around its ``Explanation`` object and its own styling, which
# would put a second visual language on the Results page next to the EDA charts
# for no gain -- these are four ordinary matplotlib charts over arrays.


def render_shap_charts(
    *,
    importance: list,
    shap_values: np.ndarray,
    feature_values: np.ndarray,
    encoded_names: list[str],
    origins: list[str],
    examples: list,
    class_label: str = "",
    target: str = "",
) -> list[Chart]:
    """Every SHAP chart for one job, skipping any the data cannot support.

    ``shap_values`` and ``feature_values`` are both ``(rows, encoded features)``
    for the single output being visualised; the aggregated, all-classes ranking
    arrives separately as ``importance`` because a bar chart of source columns
    and a beeswarm of encoded features are answering different questions.
    """
    charts: list[Chart] = []
    for chart in (
        _shap_importance(importance, target=target),
        _shap_beeswarm(shap_values, feature_values, encoded_names, class_label),
        _shap_dependence(shap_values, feature_values, encoded_names, origins, class_label),
    ):
        if chart is not None:
            charts.append(chart)

    for index, example in enumerate(examples, start=1):
        chart = _shap_waterfall(example, index, target=target)
        if chart is not None:
            charts.append(chart)
    return charts


def _shap_importance(importance: list, *, target: str) -> Chart | None:
    """Which columns move the model most, in the user's own column names."""
    if not importance:
        return None

    ranked = list(reversed(importance))  # largest at the top of a horizontal bar
    labels = [_truncate(item.feature, 28) for item in ranked]
    scores = [item.importance for item in ranked]

    fig, ax = plt.subplots(figsize=(7.0, max(2.6, 0.34 * len(ranked) + 1.0)), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.barh(labels, scores, color=SERIES, height=0.7)
    ax.set_xlabel("mean |SHAP value|", color=INK_MUTED, fontsize=9)

    about = f" on {target}" if target else ""
    _style(ax, f"What drives the prediction{about}", grid_axis="x")
    return _finish(fig, "shap_importance.png", "Global feature importance")


def _shap_beeswarm(
    shap_values: np.ndarray,
    feature_values: np.ndarray,
    encoded_names: list[str],
    class_label: str,
) -> Chart | None:
    """One dot per row per feature: how much it moved, and from what value.

    The chart the bar chart cannot be. A bar says ``support_calls`` matters; this
    says *high* ``support_calls`` pushes towards churn and low values push away,
    which is the part a reader can act on. Encoded features, not source columns,
    because a point's colour is one feature's value -- ``city`` has no single
    value to colour by, but ``city_London`` does.
    """
    if shap_values.size == 0:
        return None

    top = _top_encoded(shap_values, limit=10)
    if not top:
        return None

    fig, ax = plt.subplots(figsize=(7.0, max(2.8, 0.42 * len(top) + 1.0)), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)

    rng = np.random.default_rng(get_settings().random_seed)
    scatter = None
    for row, index in enumerate(reversed(top)):
        contributions = shap_values[:, index]
        # Colour by the feature's *rank* rather than its raw value: one extreme
        # outlier would otherwise flatten every other point to the same shade,
        # and the question the colour answers ("was this a high or low value for
        # this feature?") is ordinal anyway.
        shades = _rank_scale(feature_values[:, index])
        jitter = rng.uniform(-0.16, 0.16, size=contributions.shape[0])
        scatter = ax.scatter(
            contributions,
            np.full(contributions.shape[0], row) + jitter,
            c=shades,
            cmap=SEQUENTIAL,
            s=9,
            alpha=0.75,
            linewidths=0,
        )

    ax.set_yticks(range(len(top)))
    ax.set_yticklabels([_truncate(encoded_names[i], 26) for i in reversed(top)])
    ax.axvline(0, color=INK_MUTED, linewidth=0.8)
    ax.set_xlabel("SHAP value (push on the prediction)", color=INK_MUTED, fontsize=9)

    if scatter is not None:
        bar = fig.colorbar(scatter, ax=ax, pad=0.02, fraction=0.03)
        bar.set_label("feature value (low → high)", color=INK_MUTED, fontsize=8)
        bar.ax.tick_params(labelsize=0, length=0)
        bar.outline.set_edgecolor(GRID)

    _style(ax, _titled("How each feature's value moved the prediction", class_label), grid_axis="x")
    return _finish(fig, "shap_summary.png", "SHAP summary")


def _shap_dependence(
    shap_values: np.ndarray,
    feature_values: np.ndarray,
    encoded_names: list[str],
    origins: list[str],
    class_label: str,
) -> Chart | None:
    """The strongest continuous feature's value against its own contribution.

    Only drawn for a feature with enough distinct values for the shape to mean
    something. A one-hot column takes two values and its "dependence plot" is two
    vertical stacks -- true, and no more informative than the beeswarm row above
    it, so it is skipped rather than padded out.
    """
    if shap_values.size == 0:
        return None

    for index in _top_encoded(shap_values, limit=len(encoded_names)):
        column = feature_values[:, index]
        if len(np.unique(column)) < 8:
            continue

        fig, ax = _figure()
        ax.scatter(column, shap_values[:, index], color=SERIES, s=12, alpha=0.7, linewidths=0)
        ax.axhline(0, color=INK_MUTED, linewidth=0.8)
        ax.set_xlabel(
            f"{_truncate(encoded_names[index], 40)} (as the model sees it)",
            color=INK_MUTED,
            fontsize=9,
        )
        ax.set_ylabel("SHAP value", color=INK_MUTED, fontsize=9)
        _style(
            ax,
            _titled(f"How {_truncate(origins[index], 24)} changes the prediction", class_label),
        )
        return _finish(fig, "shap_dependence.png", "SHAP dependence")

    return None


def _shap_waterfall(example, index: int, *, target: str) -> Chart | None:
    """Why one row got the answer it got, as an addition a reader can follow.

    The bars are the contributions and they sum, with the base value, to the
    model's output -- which is stated on the chart. An explanation that does not
    add up is a picture, not an account, so the arithmetic is shown rather than
    implied.
    """
    contributions = list(getattr(example, "contributions", []))
    if not contributions:
        return None

    items = list(reversed(contributions))
    labels = [f"{_truncate(c.feature, 22)} = {_truncate(c.value, 14)}" for c in items]
    scores = [c.contribution for c in items]
    colours = [POSITIVE if score >= 0 else NEGATIVE for score in scores]

    fig, ax = plt.subplots(figsize=(7.0, max(2.6, 0.36 * len(items) + 1.2)), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.barh(labels, scores, color=colours, height=0.68)
    ax.axvline(0, color=INK_MUTED, linewidth=0.8)
    # The axis has to name the direction, not just the quantity. On a binary
    # model the contributions push towards the *positive* class whichever way the
    # row was predicted, so a bare "push on the prediction" would leave a reader
    # of a negative row's chart reading every bar backwards.
    ax.set_xlabel(
        f"push towards {target} = {example.explained_class} (SHAP value)"
        if example.explained_class
        else "push on the prediction (SHAP value)",
        color=INK_MUTED,
        fontsize=9,
    )

    about = f"{target} = {example.predicted}" if target else str(example.predicted)
    subtitle = f"row {example.row_label} → {_truncate(about, 40)}"
    if example.probability is not None:
        subtitle += f" ({example.probability:.0%} confidence)"

    _style(ax, f"Why this prediction: {subtitle}", grid_axis="x", small=True)
    ax.annotate(
        f"baseline {example.base_value:.3f} + contributions = {example.output_value:.3f}",
        xy=(0.0, -0.22),
        xycoords="axes fraction",
        color=INK_MUTED,
        fontsize=8,
    )
    return _finish(fig, f"shap_explanation_{index}.png", f"Why row {example.row_label}")


def _top_encoded(shap_values: np.ndarray, *, limit: int) -> list[int]:
    """Indices of the encoded features with the largest mean absolute contribution."""
    magnitude = np.abs(shap_values).mean(axis=0)
    order = np.argsort(-magnitude)
    return [int(i) for i in order[:limit] if magnitude[int(i)] > 0]


def _rank_scale(column: np.ndarray) -> np.ndarray:
    """Values as their position in the column's own order, in [0, 1].

    A constant column has no order, so every point gets the midpoint rather than
    a division by zero.
    """
    order = np.argsort(np.argsort(column)).astype(float)
    if len(order) < 2:
        return np.full(len(order), 0.5)
    return order / (len(order) - 1)


def _titled(text: str, class_label: str) -> str:
    """Name the class a chart is about, when there is more than one to confuse."""
    return f"{text} ({class_label})" if class_label else text


# ---- Shared plumbing --------------------------------------------------------


def _numeric_feature_columns(frame: pd.DataFrame, target: str) -> list[str]:
    """Numeric columns worth charting, capped so a wide dataset stays readable."""
    numeric = frame.select_dtypes(include=[np.number])
    columns = [str(c) for c in numeric.columns if str(c) != target]
    limit = get_settings().max_distribution_plots
    return columns[:limit]


def _figure(height: float = 3.6):
    fig, ax = plt.subplots(figsize=(7.0, height), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    return fig, ax


def _grid(count: int):
    """A small-multiples grid sized to the panel count."""
    columns = min(3, count)
    rows = math.ceil(count / columns)
    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(3.2 * columns, 2.5 * rows + 0.4),
        facecolor=SURFACE,
        squeeze=False,
    )
    flat = [ax for row in axes for ax in row]
    for ax in flat:
        ax.set_facecolor(SURFACE)
    return fig, flat


def _blank_unused(axes: list, used: int) -> None:
    """Hide the leftover cells of a grid rather than leaving empty boxes."""
    for ax in axes[used:]:
        ax.set_visible(False)


def _style(ax, title: str, *, grid_axis: str = "y", small: bool = False) -> None:
    """Recessive axes and grid, ink-coloured text -- the data is the loudest thing."""
    ax.set_title(title, color=INK, fontsize=10 if small else 12, pad=8)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_MUTED, labelsize=8)


def _finish(fig, name: str, title: str) -> Chart:
    """Serialise a figure to PNG bytes and release it.

    Closing matters: the worker renders charts for job after job in one
    long-lived process, and matplotlib keeps every unclosed figure alive.
    """
    fig.tight_layout()
    buffer = io.BytesIO()
    fig.savefig(
        buffer,
        format="png",
        dpi=get_settings().plot_dpi,
        facecolor=SURFACE,
        bbox_inches="tight",
    )
    plt.close(fig)
    return Chart(name=name, title=title, png=buffer.getvalue())


def _bin_count(n: int) -> int:
    """Sturges' rule, clamped. Enough bins to show shape, not so many it is noise."""
    if n <= 1:
        return 1
    return max(5, min(40, int(math.log2(n)) + 1))


def _truncate(text: str, limit: int = 32) -> str:
    return text if len(text) <= limit else text[: limit - 1] + "…"
