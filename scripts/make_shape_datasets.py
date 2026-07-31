"""Generate the five dataset shapes the sweep had never covered (``data/shapes/``).

``scripts/sweep_manifest.json`` has carried a ``_shapes_not_yet_covered`` list
since the sweep harness landed: multiclass, numeric-only, high-cardinality text,
small-n, and messy. Each names a route through this system that no dataset had
ever taken. This script builds one file per shape, so the list can be emptied on
evidence rather than by assertion.

**One module rather than five.** ``make_house_prices.py`` and
``make_wide_telemetry.py`` each generate a dataset that stands on its own -- a
plausible domain, worth reading as a thing. These five are not that: they are
probes, each aimed at a specific branch, and they are only interesting as a set.
Keeping them together makes the coverage argument legible in one file and stops
five near-identical ``__main__`` blocks from accumulating.

**What each shape is aimed at**, with the code it is aimed at:

``multiclass_soil``
    Four target classes, so ``_classification_metrics`` takes its ``len(classes)
    > 2`` branch: ROC-AUC computed one-vs-rest and macro-averaged, and PR-AUC
    deliberately *absent* with a warning saying why
    (``app/ml/evaluation.py:112``). The classes are imbalanced 40/30/20/10 but
    the rarest still has ~70 members, which keeps SMOTE above its floor of 6 --
    so this is also the first dataset to resample a multiclass target.

``numeric_only_calibration``
    Not one object or datetime column, which is the only condition under which
    ``choose_method`` picks K-Means (``app/ml/clustering.py:89``); every previous
    dataset had a categorical somewhere and so always got K-Prototypes. It also
    puts an empty list through ``_categorical_branches``, and gives SHAP a
    feature space with no one-hot names to map back.

``support_tickets``
    Two kinds of high cardinality, which take different routes.  ``subject`` is
    near-unique free text -- the ``text`` role, where ``FrequencyEncoder``
    collapses to a near-constant column and the question is whether anything
    downstream minds.  ``customer_id`` draws 900 rows from 180 accounts and
    lands on ~160 of them, which is the case frequency encoding actually exists
    for: too many levels for one-hot, but with real repetition to learn from.

``small_pilot``
    150 rows with exactly 4 members of the rarer class, which trips two
    minimums at once: stratification drops from 5 folds to 4
    (``app/ml/modeling.py:726``) and SMOTE declines to run at all, 4 being under
    ``_MIN_MINORITY_FOR_SMOTE`` of 6 (``app/ml/modeling.py:361``). Both should
    appear as warnings in the report; neither should fail the run.

    It does *not* reach the clustering minimum, and no honest dataset would.
    That guard needs ``len(frame) // 5 < 2`` -- fewer than ten rows -- which is
    not a small dataset, it is a broken one. The guard is left to its unit test.

``messy_intake``
    Duplicate headers, an all-null column, a column mixing numbers with prose,
    unicode and whitespace-padded names, thousands separators and currency
    symbols, four spellings of one country, three date formats, and duplicate
    rows.

    **Most of the target's signal is inside the mess** -- roughly 60% from
    ``revenue`` and ``country``, both of which arrive unparseable, and 40% from a
    clean numeric column. That split is the point: if cleaning recovers the
    messy columns the score is good, and if it silently gives up on them the
    score falls to what the clean column alone supports. The dataset therefore
    reports how much was recovered, instead of only whether the run crashed.

    It ships as **two files**, which the first sweep made necessary. The
    duplicate header is terminal -- ``inspect_csv`` rejects the upload rather
    than let pandas rename it to ``amount.1`` behind the user's back -- so
    ``messy_intake.csv`` proves that one rule fires and never reaches the
    pipeline at all. ``messy_intake_renamed.csv`` is the same sheet with the
    second ``amount`` renamed, and it is the one that carries the remaining nine
    traits through cleaning, encoding and modelling.

Every output is gitignored (only ``data/examples/*.csv`` is committed) and every
manifest entry is marked ``in_repo: false``, as with the wide dataset. The
generator is the artifact worth reading; the CSV is a build product.

Usage::

    python scripts/make_shape_datasets.py            # all five
    python scripts/make_shape_datasets.py --only messy_intake
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd

OUTPUT_DIR = Path("data/shapes")

# One base seed, offset per dataset, so any one of them can be regenerated on its
# own and come out byte-identical to the run that produced the sweep numbers.
SEED = 20260731


def _rng(offset: int) -> np.random.Generator:
    return np.random.default_rng(SEED + offset)


# ---- multiclass --------------------------------------------------------------


def build_multiclass() -> pd.DataFrame:
    """Soil samples graded into four classes, 700 rows.

    The grade is cut from a latent quality score at the 40th, 70th and 90th
    percentiles, which produces the 40/30/20/10 split exactly rather than
    approximately. The ordering is real -- ``premium`` samples genuinely score
    highest -- so a model that learns nothing is visibly distinguishable from one
    that learns the ordering but confuses adjacent grades.
    """
    rng = _rng(1)
    n = 700

    regions = rng.choice(["north", "south", "coastal", "upland", "valley"], size=n)
    ph = rng.normal(6.4, 0.7, n).round(2)
    nitrogen = rng.gamma(shape=3.0, scale=14.0, size=n).round(1)
    phosphorus = rng.gamma(shape=2.2, scale=9.0, size=n).round(1)
    potassium = rng.gamma(shape=2.6, scale=22.0, size=n).round(1)
    organic_matter = rng.normal(4.1, 1.3, n).clip(0.2).round(2)
    rainfall_mm = rng.normal(880, 190, n).clip(120).round(0)

    # A handful of missing readings, because a lab sheet always has some.
    nitrogen[rng.random(n) < 0.04] = np.nan
    organic_matter[rng.random(n) < 0.03] = np.nan

    def z(values: np.ndarray) -> np.ndarray:
        centred = values - np.nanmean(values)
        return centred / (np.nanstd(values) or 1.0)

    quality = (
        1.3 * z(organic_matter)
        + 1.0 * z(nitrogen)
        + 0.6 * z(phosphorus)
        + 0.4 * z(potassium)
        # pH helps most in the middle of the range, so the relationship is not
        # monotone and a linear model cannot take the whole dataset.
        - 0.9 * np.abs(ph - 6.5)
        + np.where(regions == "valley", 0.5, 0.0)
        + np.where(regions == "upland", -0.4, 0.0)
    )
    quality = np.where(np.isnan(quality), np.nanmean(quality), quality)
    quality += rng.normal(0, 0.8, n)

    cuts = np.quantile(quality, [0.40, 0.70, 0.90])
    grade = np.select(
        [quality <= cuts[0], quality <= cuts[1], quality <= cuts[2]],
        ["poor", "fair", "good"],
        default="premium",
    )

    return pd.DataFrame(
        {
            "sample_id": [f"SOIL-{i:05d}" for i in range(1, n + 1)],
            "sampled_on": (
                pd.to_datetime("2024-04-01") + pd.to_timedelta(rng.integers(0, 700, n), unit="D")
            ).strftime("%Y-%m-%d"),
            "region": regions,
            "ph": ph,
            "nitrogen_ppm": nitrogen,
            "phosphorus_ppm": phosphorus,
            "potassium_ppm": potassium,
            "organic_matter_pct": organic_matter,
            "rainfall_mm": rainfall_mm,
            "grade": grade,
        }
    )


def describe_multiclass(frame: pd.DataFrame) -> list[str]:
    counts = frame.grade.value_counts()
    shares = ", ".join(f"{name} {count / len(frame):.0%}" for name, count in counts.items())
    return [
        f"classes: {frame.grade.nunique()} ({shares})",
        f"rarest class: {counts.min()} rows (SMOTE floor is 6)",
    ]


# ---- numeric-only ------------------------------------------------------------


def build_numeric_only() -> pd.DataFrame:
    """Instrument calibration, 600 rows, every column a float.

    The constraint that makes this dataset what it is, is negative: no strings,
    no dates, no booleans, nothing an integer code could be mistaken for a
    category. Scales differ by four orders of magnitude and two pairs are
    strongly collinear, so scaling and correlation both have something to do.
    """
    rng = _rng(2)
    n = 600

    drift_ppm = rng.normal(0, 42.0, n)
    # Collinear by construction: the coarse channel is the fine one, rounded and
    # re-noised. Correlation ~0.97.
    drift_coarse = (drift_ppm + rng.normal(0, 9.0, n)).round(0)
    ambient_c = rng.normal(21.5, 3.4, n)
    housing_c = ambient_c * 1.18 + rng.normal(0, 0.8, n)
    supply_mv = rng.normal(3300, 45, n)
    vibration_g = rng.gamma(2.0, 0.06, n)
    hours_since_service = rng.exponential(430, n)
    humidity_pct = rng.normal(47, 11, n).clip(2, 98)
    pressure_kpa = rng.normal(101.3, 1.1, n)
    gain_error = rng.normal(0, 0.0021, n)
    noise_floor_uv = rng.gamma(3.0, 0.9, n)

    # Gaps in two channels, at rates where imputing and dropping are both
    # defensible decisions for the strategy agent to make.
    gain_error[rng.random(n) < 0.09] = np.nan
    humidity_pct[rng.random(n) < 0.15] = np.nan

    def z(values: np.ndarray) -> np.ndarray:
        centred = values - np.nanmean(values)
        return centred / (np.nanstd(values) or 1.0)

    logit = (
        1.4 * np.abs(z(drift_ppm))
        + 0.9 * z(hours_since_service)
        + 0.7 * z(np.nan_to_num(gain_error, nan=float(np.nanmean(gain_error))))
        + 0.5 * z(vibration_g)
        + 0.3 * z(housing_c)
        + rng.normal(0, 0.7, n)
    )
    probability = 1 / (1 + np.exp(-(logit - 1.6)))
    out_of_spec = (rng.random(n) < probability).astype(int)

    return pd.DataFrame(
        {
            "drift_ppm": drift_ppm.round(3),
            "drift_coarse_ppm": drift_coarse,
            "ambient_c": ambient_c.round(2),
            "housing_c": housing_c.round(2),
            "supply_mv": supply_mv.round(1),
            "vibration_g": vibration_g.round(4),
            "hours_since_service": hours_since_service.round(1),
            "humidity_pct": humidity_pct.round(1),
            "pressure_kpa": pressure_kpa.round(2),
            "gain_error": gain_error.round(5),
            "noise_floor_uv": noise_floor_uv.round(3),
            "out_of_spec": out_of_spec,
        }
    )


def describe_numeric_only(frame: pd.DataFrame) -> list[str]:
    non_numeric = [c for c in frame.columns if not pd.api.types.is_numeric_dtype(frame[c])]
    return [
        f"non-numeric columns: {len(non_numeric)} (must be 0 for K-Means to be chosen)",
        f"out_of_spec: {frame.out_of_spec.mean():.1%} positive",
        f"widest scale gap: {frame.supply_mv.std() / frame.gain_error.std():,.0f}x",
    ]


# ---- high-cardinality text ---------------------------------------------------

_SUBJECT_TEMPLATES = [
    "cannot sign in to {area} since release {build}",
    "{area} export fails with error {build}",
    "billing mismatch on invoice {build} for {area}",
    "request: raise {area} rate limit above {build}",
    "{area} dashboard blank after {build} upgrade",
    "intermittent timeouts calling {area} endpoint {build}",
]


def build_high_cardinality_text() -> pd.DataFrame:
    """Support tickets, 900 rows, with two different kinds of high cardinality.

    ``subject`` is near-unique: the frequency encoder will map almost every row
    to the same tiny value, which is the honest encoding of a column that carries
    no repeated signal. ``customer_id`` is the opposite -- ~160 levels, mean ~5
    rows each -- and two of those customers escalate far more than the rest, so
    the encoding has something real to find.
    """
    rng = _rng(3)
    n = 900

    areas = ["reporting", "billing", "api", "sso", "mobile", "webhooks"]
    customers = [f"CUST-{i:04d}" for i in range(1, 181)]
    # A long-tailed customer distribution: some heavy accounts, many one-ticket
    # ones. Uniform levels would make frequency encoding a constant. The tail is
    # kept moderate on purpose -- a Pareto tail hands one account 13% of the
    # tickets, which makes the encoding a single-account detector rather than a
    # distribution.
    weights = rng.gamma(1.4, 1.0, len(customers))
    weights = weights / weights.sum()
    customer_id = rng.choice(customers, size=n, p=weights)

    product_area = rng.choice(areas, size=n)
    priority = rng.choice(["P1", "P2", "P3", "P4"], size=n, p=[0.08, 0.22, 0.45, 0.25])
    channel = rng.choice(["email", "portal", "phone"], size=n, p=[0.5, 0.35, 0.15])

    subject = [
        _SUBJECT_TEMPLATES[rng.integers(len(_SUBJECT_TEMPLATES))].format(
            area=rng.choice(areas), build=f"{rng.integers(2, 9)}.{rng.integers(0, 40)}"
        )
        for _ in range(n)
    ]

    first_response_mins = rng.gamma(2.0, 55.0, n).round(0)
    messages_exchanged = rng.poisson(4.0, n) + 1

    # Two accounts that escalate regardless of anything else -- the signal that
    # only a customer-level encoding can pick up.
    hot_accounts = set(rng.choice(customers, size=2, replace=False))
    is_hot = np.array([c in hot_accounts for c in customer_id])

    logit = (
        np.select(
            [priority == "P1", priority == "P2", priority == "P3"],
            [2.1, 1.0, 0.1],
            default=-0.6,
        )
        + 0.9 * (first_response_mins - first_response_mins.mean()) / first_response_mins.std()
        + 0.4 * (messages_exchanged - messages_exchanged.mean()) / messages_exchanged.std()
        + np.where(is_hot, 1.8, 0.0)
        + rng.normal(0, 0.8, n)
    )
    probability = 1 / (1 + np.exp(-(logit - 1.9)))

    return pd.DataFrame(
        {
            "ticket_ref": [f"TCK-{i:06d}" for i in range(1, n + 1)],
            "opened_at": (
                pd.to_datetime("2025-01-06") + pd.to_timedelta(rng.integers(0, 360, n), unit="D")
            ).strftime("%Y-%m-%d"),
            "customer_id": customer_id,
            "subject": subject,
            "product_area": product_area,
            "priority": priority,
            "channel": channel,
            "first_response_mins": first_response_mins,
            "messages_exchanged": messages_exchanged,
            "escalated": (rng.random(n) < probability).astype(int),
        }
    )


def describe_high_cardinality_text(frame: pd.DataFrame) -> list[str]:
    return [
        f"subject: {frame.subject.nunique()} distinct over {len(frame)} rows "
        f"({frame.subject.nunique() / len(frame):.0%} unique)",
        f"customer_id: {frame.customer_id.nunique()} levels, "
        f"mean {len(frame) / frame.customer_id.nunique():.1f} rows each, "
        f"busiest {frame.customer_id.value_counts().max()}",
        f"escalated: {frame.escalated.mean():.1%} positive",
    ]


# ---- small n -----------------------------------------------------------------


def build_small_n() -> pd.DataFrame:
    """A 150-row pilot study with exactly 4 responders.

    The two numbers that matter are 150 and 4, and they are chosen rather than
    sampled: 4 is below the SMOTE floor of 6 and below the 5 folds the settings
    ask for, so both guards fire on one run. Sampling the minority count would
    make which guards fire a property of the seed.
    """
    rng = _rng(4)
    n = 150
    n_responders = 4

    age = rng.normal(58, 12, n).clip(19, 92).round(0)
    baseline_score = rng.normal(42, 9, n).round(1)
    dose_mg = rng.choice([5.0, 10.0, 20.0], size=n, p=[0.4, 0.4, 0.2])
    weeks_enrolled = rng.integers(4, 25, n)
    cohort = rng.choice(["A", "B", "C"], size=n, p=[0.45, 0.35, 0.20])
    site = rng.choice(["leeds", "cardiff", "glasgow", "belfast"], size=n)

    # Responders are the 4 highest-scoring rows on a latent index, so the class
    # is learnable in principle and the model is not being asked to fit noise --
    # it is being asked to do so from four examples, which is the actual test.
    index = (
        0.9 * (dose_mg - dose_mg.mean()) / dose_mg.std()
        + 0.6 * (baseline_score - baseline_score.mean()) / baseline_score.std()
        - 0.4 * (age - age.mean()) / age.std()
        + rng.normal(0, 0.5, n)
    )
    responded = np.zeros(n, dtype=int)
    responded[np.argsort(index)[-n_responders:]] = 1

    return pd.DataFrame(
        {
            "participant": [f"P{i:03d}" for i in range(1, n + 1)],
            "cohort": cohort,
            "site": site,
            "age": age,
            "baseline_score": baseline_score,
            "dose_mg": dose_mg,
            "weeks_enrolled": weeks_enrolled,
            "responded": responded,
        }
    )


def describe_small_n(frame: pd.DataFrame) -> list[str]:
    minority = int(frame.responded.sum())
    return [
        f"{len(frame)} rows, {minority} positive ({minority / len(frame):.1%})",
        f"expect: folds reduced 5 -> {minority}, SMOTE skipped ({minority} < 6)",
        f"clustering still runs: k_max = min(8, {len(frame) // 5}) = {min(8, len(frame) // 5)}",
    ]


# ---- messy -------------------------------------------------------------------

_COUNTRY_SPELLINGS = ["UK", "uk", " Uk", "U.K."]
_READING_JUNK = ["n/a", "N/A", "twelve", "unknown", "--", ""]


def build_messy(*, duplicate_headers: bool = True) -> pd.DataFrame:
    """A 400-row intake sheet with everything wrong with it that usually is.

    Built as a DataFrame with deliberately duplicated and unicode column names,
    then written straight out -- pandas preserves both on ``to_csv``, so the file
    on disk really does have two columns called ``amount``.

    The mess is not decoration. ``revenue`` and ``country`` carry ~60% of the
    target's signal between them and both arrive unparseable, so the score says
    how much of the mess was recovered rather than merely whether the run
    survived it.

    ``duplicate_headers=False`` gives the same sheet with the second ``amount``
    renamed, and it exists because the first sweep of this dataset showed the
    duplicate is *terminal*: ``inspect_csv`` rejects the upload outright, on the
    deliberate grounds that pandas would otherwise rename it to ``amount.1`` and
    show the agents a column the user never wrote. That rejection is the correct
    behaviour and worth keeping a probe for -- but it means the file tests one
    validation rule and nothing else. The renamed variant is what actually
    carries the rest of the mess through cleaning, encoding and modelling.
    """
    rng = _rng(5)
    n = 400

    applicant_score = rng.normal(620, 85, n).round(0)
    revenue_true = rng.lognormal(mean=10.4, sigma=0.75, size=n)
    country_true = rng.choice(["UK", "IE", "FR"], size=n, p=[0.6, 0.25, 0.15])

    logit = (
        0.9 * (applicant_score - applicant_score.mean()) / applicant_score.std()
        + 1.1 * (np.log(revenue_true) - np.log(revenue_true).mean()) / np.log(revenue_true).std()
        + np.where(country_true == "UK", 0.7, 0.0)
        + rng.normal(0, 0.6, n)
    )
    approved = np.where(rng.random(n) < 1 / (1 + np.exp(-logit)), "Yes", "No")

    # revenue: thousands separators and a currency symbol, so it reads as text.
    revenue = [f"£{value:,.2f}" for value in revenue_true]

    # country: four spellings of the same thing for UK rows, which cleaning has
    # to fold together or spend three one-hot columns on.
    country = [
        _COUNTRY_SPELLINGS[rng.integers(len(_COUNTRY_SPELLINGS))] if value == "UK" else value
        for value in country_true
    ]

    # reading: numbers and prose in one column, the classic mixed dtype.
    reading: list[object] = []
    for _ in range(n):
        if rng.random() < 0.25:
            reading.append(_READING_JUNK[rng.integers(len(_READING_JUNK))])
        else:
            reading.append(round(float(rng.normal(37.2, 4.1)), 1))

    # received: three date formats in one column.
    base = pd.to_datetime("2025-02-01") + pd.to_timedelta(rng.integers(0, 300, n), unit="D")
    formats = rng.integers(0, 3, n)
    received = [
        stamp.strftime("%Y-%m-%d")
        if fmt == 0
        else stamp.strftime("%d/%m/%Y")
        if fmt == 1
        else stamp.strftime("%B %-d, %Y")
        for stamp, fmt in zip(base, formats, strict=True)
    ]

    frame = pd.DataFrame(
        {
            "c00": [f"REF{i:04d}" for i in range(1, n + 1)],
            "c01": received,
            "c02": country,
            "c03": revenue,
            "c04": applicant_score,
            "c05": reading,
            # The two columns that will both be called "amount". Different
            # meanings, identical names -- which is how it happens in the wild,
            # from a join nobody suffixed.
            "c06": (revenue_true * rng.uniform(0.02, 0.09, n)).round(2),
            "c07": rng.integers(1, 40, n),
            "c08": [np.nan] * n,
            "c09": rng.choice(["desk", "field", "portal"], size=n),
            "c10": rng.normal(0.5, 0.2, n).round(3),
            "c11": rng.integers(15, 45, n),
            "c12": approved,
        }
    )
    frame.columns = [
        "ref",
        "received",
        "country",
        "revenue",
        "applicant_score",
        "reading",
        "amount",
        "amount" if duplicate_headers else "amount_paid",
        "legacy_flag",  # all null
        " submitted_by ",  # padded header
        "naïve_score",  # unicode
        "温度",  # unicode, non-latin
        "approved",
    ]

    # Duplicate rows, which cleaning is asked to remove. Appended rather than
    # shuffled in, because their position does not matter and keeping them at the
    # end makes the file easier to eyeball.
    duplicates = frame.sample(n=int(n * 0.08), random_state=SEED)
    return pd.concat([frame, duplicates], ignore_index=True)


def build_messy_accepted() -> pd.DataFrame:
    """``build_messy`` with the duplicate header renamed, so upload accepts it."""
    return build_messy(duplicate_headers=False)


def describe_messy(frame: pd.DataFrame) -> list[str]:
    names = list(frame.columns)
    duplicated = {name for name in names if names.count(name) > 1}
    all_null = [name for name in names if frame[name].isna().all().all()]
    return [
        f"{len(frame)} rows ({int(frame.duplicated().sum())} duplicated), {len(names)} columns",
        f"duplicate headers: {sorted(duplicated) or 'none (renamed variant)'}",
        f"all-null columns: {all_null}",
        f"country spellings: {sorted(set(frame['country']))}",
        "approved: " + ", ".join(f"{k} {v}" for k, v in frame["approved"].value_counts().items()),
    ]


# ---- registry ----------------------------------------------------------------

Builder = Callable[[], pd.DataFrame]
Describer = Callable[[pd.DataFrame], list[str]]

DATASETS: dict[str, tuple[Builder, Describer]] = {
    "multiclass_soil": (build_multiclass, describe_multiclass),
    "numeric_only_calibration": (build_numeric_only, describe_numeric_only),
    "support_tickets": (build_high_cardinality_text, describe_high_cardinality_text),
    "small_pilot": (build_small_n, describe_small_n),
    "messy_intake": (build_messy, describe_messy),
    "messy_intake_renamed": (build_messy_accepted, describe_messy),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--only", choices=sorted(DATASETS), help="generate a single shape")
    parser.add_argument("--out", type=Path, default=OUTPUT_DIR, help="output directory")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    names = [args.only] if args.only else list(DATASETS)

    for name in names:
        build, describe = DATASETS[name]
        frame = build()
        path = args.out / f"{name}.csv"
        frame.to_csv(path, index=False, encoding="utf-8")

        print(f"\n{path}  ({len(frame)} rows x {frame.shape[1]} columns)")
        for line in describe(frame):
            print(f"  {line}")


if __name__ == "__main__":
    main()
