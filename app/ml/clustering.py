"""Clustering -- finding natural groupings of rows (spec 9, build-plan Section 6).

**The guardrail, first, because it is the point of this module's design:**
cluster labels are computed over *every* row and are therefore contaminated by
the rows that later land in a test fold. Feeding them to the model as a feature
would leak exactly like fitting a scaler across the whole dataset does (spec 9).
So the labels never enter the DataFrame. They are returned in a separate object,
used to colour a scatter plot and to compute per-group summaries, and discarded.
``graph.py`` asserts that the frame leaving this stage is unchanged, and
``tests/test_leakage.py`` asserts a cluster column never reaches the model.

That constraint is also what makes everything else here safe. Imputing and
scaling across the whole dataset -- which would be leakage in the modelling path
-- is fine here precisely because nothing downstream of it touches the model.

Method selection follows spec 9: K-Means for all-numeric data, K-Prototypes when
categories are present. The Planner states a preference and this module overrides
it when the data cannot support it, because the failure is not cosmetic --
K-Means on categorical data requires inventing Euclidean distances between labels
that have no distance between them. One-hot encoding then running K-Means is the
usual workaround and it is exactly the fabricated geometry the spec rejects.

``k`` is chosen by silhouette score across a range, so the number of groups is
searched rather than assumed. For mixed data the silhouette is measured on a
one-hot encoding, since silhouette needs a metric space -- an approximation, and
noted as one in the report; the clustering itself uses K-Prototypes' own mixed
distance.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from app.core.config import get_settings
from app.ml.contracts import ClusteringMethod, ClusteringReport, ClusterProfile, PlannerPlan
from app.ml.plots import Chart, render_cluster_scatter

logger = logging.getLogger(__name__)

AGENT_NAME = "clustering"

# Categorical columns with more distinct values than this are left out of
# clustering: they explode K-Prototypes' mode computation and contribute noise
# rather than structure. Same reasoning as the preprocessing cardinality cap.
_MAX_CLUSTER_CARDINALITY = 30
# A cluster needs at least this many rows before "the average of this group"
# means anything worth describing.
_MIN_ROWS_PER_CLUSTER = 5
# How far from the dataset average a feature must sit to be worth naming as
# distinguishing, in standard deviations.
_DISTINCTIVE_Z = 0.5
_MAX_DISTINGUISHING = 4


@dataclass(slots=True)
class ClusteringResult:
    """Labels, the report, and the scatter -- with the labels kept out of the data.

    ``labels`` is deliberately a bare array rather than a column on the frame.
    Handing back a DataFrame with a ``cluster`` column would make leaking it into
    the model a one-line mistake somebody makes later; handing back an array
    makes using it as a feature a deliberate act.
    """

    report: ClusteringReport
    labels: np.ndarray | None = None
    scatter: Chart | None = None
    skipped_reason: str | None = None
    warnings: list[str] = field(default_factory=list)


def choose_method(
    frame: pd.DataFrame, requested: ClusteringMethod
) -> tuple[ClusteringMethod, str | None]:
    """Honour the planner's choice unless the data cannot support it.

    Returns the method to use and, when it differs from the request, why -- so
    the artifact records that the plan was overridden instead of implying it was
    followed.
    """
    has_categorical = bool(_categorical_columns(frame))
    if has_categorical and requested == "kmeans":
        return "kprototypes", (
            "Planner asked for K-Means, but the data has categorical columns; "
            "used K-Prototypes so category distances are not fabricated."
        )
    if not has_categorical and requested == "kprototypes":
        return "kmeans", (
            "Planner asked for K-Prototypes, but every column is numeric; "
            "used K-Means, which is the right tool for that."
        )
    return requested, None


def run_clustering(
    frame: pd.DataFrame,
    *,
    target: str,
    plan: PlannerPlan | None = None,
) -> ClusteringResult:
    """Cluster the rows, profile the groups, and draw them. Never raises.

    Clustering is descriptive: a dataset it cannot handle should cost the run its
    scatter plot, not its model. Every failure path returns a result carrying a
    reason instead of propagating.
    """
    plan = plan or PlannerPlan()
    settings = get_settings()
    warnings: list[str] = []

    if not plan.run_clustering:
        return _skipped("The plan turned clustering off.")

    features = frame.drop(columns=[target], errors="ignore")
    numeric = _numeric_columns(features)
    categorical = _categorical_columns(features)
    if not numeric and not categorical:
        return _skipped("No columns suitable for clustering.")

    method, override = choose_method(features, plan.clustering_method)
    if override:
        logger.info("Clustering method overridden: %s", override)

    k_max = min(settings.cluster_k_max, len(frame) // _MIN_ROWS_PER_CLUSTER)
    if k_max < settings.cluster_k_min:
        return _skipped(
            f"Only {len(frame)} rows: too few to split into groups of at least "
            f"{_MIN_ROWS_PER_CLUSTER}."
        )

    try:
        prepared = _prepare(features, numeric, categorical)
        chosen_k, silhouettes = _select_k(
            prepared,
            method=method,
            k_min=settings.cluster_k_min,
            k_max=k_max,
            sample_size=settings.cluster_sample_size,
            random_seed=settings.random_seed,
        )
        labels = _fit_labels(prepared, method=method, k=chosen_k, random_seed=settings.random_seed)
    except Exception as exc:  # pragma: no cover - defensive; EDA must not fail a job
        logger.exception("Clustering failed")
        return _skipped(f"Clustering could not be completed: {exc}")

    if method == "kprototypes":
        warnings.append(
            "Cluster quality for mixed data is measured on a one-hot encoding, "
            "which approximates the mixed distance the clustering itself uses."
        )

    profiles = profile_clusters(features, labels, numeric=numeric, categorical=categorical)
    scatter = _scatter(prepared.matrix, labels)

    report = ClusteringReport(
        method=method,
        k=chosen_k,
        silhouette=silhouettes.get(chosen_k, 0.0),
        silhouette_by_k=silhouettes,
        profiles=profiles,
        scatter_plot=scatter.name if scatter else None,
        method_override_reason=override,
        warnings=warnings,
    )
    if report.is_weak():
        report.warnings.append(
            f"A silhouette of {report.silhouette:.2f} is below the 0.5 mark for "
            "reasonably separated groups. Clustering always returns groups, even "
            "in data that has none -- treat these as a loose description rather "
            "than distinct segments."
        )

    logger.info("Clustering: %s k=%d silhouette=%.3f", method, chosen_k, report.silhouette)
    return ClusteringResult(report=report, labels=labels, scatter=scatter, warnings=warnings)


# ---- Preparing the matrix ---------------------------------------------------


@dataclass(slots=True)
class _Prepared:
    """The numeric matrix used for distances, plus what K-Prototypes needs."""

    matrix: np.ndarray  # fully numeric: scaled numbers + one-hot categories
    mixed: np.ndarray | None  # scaled numbers + raw category codes, for kprototypes
    categorical_indices: list[int]


def _prepare(frame: pd.DataFrame, numeric: list[str], categorical: list[str]) -> _Prepared:
    """Build the matrices clustering needs.

    Imputation and scaling happen over the whole dataset here. That is leakage in
    the modelling path and correct here: this output never reaches the model (see
    the module docstring), and a cluster computed on a subset would describe a
    subset.
    """
    parts: list[np.ndarray] = []
    if numeric:
        block = frame[numeric].astype(float)
        block = block.fillna(block.median())
        scaled = StandardScaler().fit_transform(block.to_numpy())
        parts.append(scaled)

    mixed = None
    categorical_indices: list[int] = []
    if categorical:
        codes = frame[categorical].astype(str).fillna("missing")
        one_hot = pd.get_dummies(codes, drop_first=False).to_numpy(dtype=float)
        parts.append(one_hot)

        # K-Prototypes wants the raw categories alongside the numbers, not
        # one-hot columns -- that is the entire reason to use it.
        numeric_block = (
            StandardScaler().fit_transform(
                frame[numeric].astype(float).fillna(frame[numeric].astype(float).median())
            )
            if numeric
            else np.empty((len(frame), 0))
        )
        mixed = np.column_stack([numeric_block, codes.to_numpy()])
        categorical_indices = list(range(numeric_block.shape[1], mixed.shape[1]))

    matrix = np.column_stack(parts) if parts else np.empty((len(frame), 0))
    return _Prepared(matrix=matrix, mixed=mixed, categorical_indices=categorical_indices)


def _numeric_columns(frame: pd.DataFrame) -> list[str]:
    return [
        str(c)
        for c in frame.columns
        if pd.api.types.is_numeric_dtype(frame[c]) and not pd.api.types.is_bool_dtype(frame[c])
    ]


def _categorical_columns(frame: pd.DataFrame) -> list[str]:
    out = []
    for name in frame.columns:
        series = frame[name]
        if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
            continue
        if pd.api.types.is_datetime64_any_dtype(series):
            continue
        if int(series.nunique(dropna=True)) <= _MAX_CLUSTER_CARDINALITY:
            out.append(str(name))
    return out


# ---- Choosing k and fitting -------------------------------------------------


def _select_k(
    prepared: _Prepared,
    *,
    method: ClusteringMethod,
    k_min: int,
    k_max: int,
    sample_size: int,
    random_seed: int,
) -> tuple[int, dict[int, float]]:
    """Try every k in range and keep the best-separated one.

    Runs on a sample above a size threshold: the silhouette curve settles long
    before every row is involved, and fitting eight models over a million rows to
    pick a number between two and eight is wasted work.
    """
    matrix = prepared.matrix
    indices = np.arange(len(matrix))
    if len(matrix) > sample_size:
        rng = np.random.default_rng(random_seed)
        indices = rng.choice(len(matrix), size=sample_size, replace=False)

    sample = matrix[indices]
    mixed_sample = prepared.mixed[indices] if prepared.mixed is not None else None

    scores: dict[int, float] = {}
    for k in range(k_min, k_max + 1):
        labels = _fit_labels(
            _Prepared(sample, mixed_sample, prepared.categorical_indices),
            method=method,
            k=k,
            random_seed=random_seed,
        )
        if len(set(labels.tolist())) < 2:
            # A degenerate fit that put everything in one group tells us nothing.
            continue
        scores[k] = float(silhouette_score(sample, labels))

    if not scores:
        # Fall back to the smallest k rather than giving up: the caller has
        # already established there are enough rows to form groups.
        return k_min, {}
    best = max(scores, key=lambda key: scores[key])
    return best, scores


def _fit_labels(
    prepared: _Prepared, *, method: ClusteringMethod, k: int, random_seed: int
) -> np.ndarray:
    """Fit one model and return its labels."""
    if method == "kprototypes" and prepared.mixed is not None:
        from kmodes.kprototypes import KPrototypes

        model = KPrototypes(n_clusters=k, random_state=random_seed, n_init=3, verbose=0)
        return np.asarray(
            model.fit_predict(prepared.mixed, categorical=prepared.categorical_indices)
        )

    model = KMeans(n_clusters=k, random_state=random_seed, n_init=10)
    return np.asarray(model.fit_predict(prepared.matrix))


# ---- Describing the groups --------------------------------------------------


def profile_clusters(
    frame: pd.DataFrame,
    labels: np.ndarray,
    *,
    numeric: list[str],
    categorical: list[str],
) -> list[ClusterProfile]:
    """Summarise each group by how it departs from the dataset as a whole.

    The comparison is against the overall average rather than between clusters,
    because that is the sentence a reader wants: "this group earns much more than
    typical" beats "this group earns more than cluster 3". These measured
    differences are also what the LLM is handed, so its description is grounded
    in numbers rather than invented from a cluster number.
    """
    profiles: list[ClusterProfile] = []
    total = len(frame)

    for cluster in sorted({int(v) for v in labels}):
        mask = labels == cluster
        subset = frame[mask]
        distinguishing: dict[str, str] = {}

        scored: list[tuple[float, str, str]] = []
        for name in numeric:
            overall = frame[name].astype(float)
            spread = float(overall.std(ddof=0))
            if spread <= 0:
                continue
            delta = float(subset[name].astype(float).mean() - overall.mean())
            z = delta / spread
            if abs(z) >= _DISTINCTIVE_Z:
                direction = "higher" if z > 0 else "lower"
                scored.append(
                    (abs(z), name, f"{direction} than average ({subset[name].mean():,.1f})")
                )

        for name in categorical:
            overall_share = frame[name].astype(str).value_counts(normalize=True)
            local_share = subset[name].astype(str).value_counts(normalize=True)
            if local_share.empty:
                continue
            top = str(local_share.index[0])
            lift = float(local_share.iloc[0] - overall_share.get(top, 0.0))
            if lift >= 0.15:
                scored.append((lift, name, f"mostly {top} ({local_share.iloc[0]:.0%})"))

        scored.sort(reverse=True, key=lambda item: item[0])
        for _, name, description in scored[:_MAX_DISTINGUISHING]:
            distinguishing[name] = description

        profiles.append(
            ClusterProfile(
                cluster=cluster,
                size=int(mask.sum()),
                share=float(mask.sum() / total) if total else 0.0,
                distinguishing_features=distinguishing,
            )
        )
    return profiles


def _scatter(matrix: np.ndarray, labels: np.ndarray) -> Chart | None:
    """Project to two dimensions with PCA and draw the groups."""
    if matrix.shape[1] < 2:
        return None
    pca = PCA(n_components=2, random_state=get_settings().random_seed)
    coordinates = pca.fit_transform(matrix)
    ratio = pca.explained_variance_ratio_
    return render_cluster_scatter(
        coordinates, labels, explained_variance=(float(ratio[0]), float(ratio[1]))
    )


def _skipped(reason: str) -> ClusteringResult:
    """A result that says clustering did not happen, and why."""
    logger.info("Clustering skipped: %s", reason)
    return ClusteringResult(
        report=ClusteringReport(method="kmeans", k=0, silhouette=0.0, warnings=[reason]),
        labels=None,
        skipped_reason=reason,
    )
