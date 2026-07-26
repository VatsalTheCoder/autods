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
