"""Generate the regression example dataset (``data/examples/house_prices.csv``).

Kept in the repo rather than only its output, so the data's properties are
inspectable: a reader can see that the signal is real, that the noise is
deliberate, and that nothing was cherry-picked to flatter a model.

The dataset exists because every end-to-end run of this pipeline so far has been
a *classification* task. Regression takes a different route through almost every
stage -- ``KFold`` rather than ``StratifiedKFold``, no resampling, r²/RMSE/MAE
rather than F1, a different SHAP branch -- and none of it had been exercised as a
whole.

The columns are chosen to make the run pass through the interesting machinery
rather than to be realistic for their own sake:

- ``listing_id`` -- an identifier, which cleaning should drop rather than model.
- ``agent_email`` -- should be flagged as PII at the schema checkpoint.
- ``listed_date`` -- a timestamp, which becomes calendar parts.
- ``neighbourhood`` -- eight levels, enough for one-hot encoding to matter.
- ``epc_rating`` -- genuinely ordinal, so the strategy agent has something to get
  right that one-hot would get wrong.
- ``condition_rating`` -- ~12% missing, so imputation happens inside the folds.
- ``sale_price`` -- continuous, skewed the way prices actually are.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SEED = 20260729
N_ROWS = 600

NEIGHBOURHOODS = {
    # name: (price multiplier, share of listings)
    "Riverside": (1.45, 0.10),
    "Old Town": (1.30, 0.12),
    "Parkview": (1.18, 0.14),
    "Meadowbank": (1.02, 0.16),
    "Eastfield": (0.95, 0.16),
    "Northgate": (0.88, 0.14),
    "Docklands": (0.82, 0.10),
    "Industrial Park": (0.70, 0.08),
}

PROPERTY_TYPES = {"flat": 0.82, "terraced": 0.95, "semi-detached": 1.10, "detached": 1.32}

# Ordinal on purpose: A is better than B is better than C. One-hot encoding this
# throws away the ordering, which is the mistake the feature strategy agent is
# supposed to avoid.
EPC_RATINGS = ["A", "B", "C", "D", "E", "F"]


def build() -> pd.DataFrame:
    rng = np.random.default_rng(SEED)

    names = list(NEIGHBOURHOODS)
    shares = np.array([NEIGHBOURHOODS[name][1] for name in names])
    neighbourhood = rng.choice(names, size=N_ROWS, p=shares / shares.sum())

    property_type = rng.choice(list(PROPERTY_TYPES), size=N_ROWS, p=[0.34, 0.28, 0.22, 0.16])

    # Floor area depends on property type, which is what makes the two columns
    # correlated -- a realistic wrinkle for the correlation reporting to find.
    base_area = {"flat": 620, "terraced": 950, "semi-detached": 1180, "detached": 1650}
    area_sqft = np.array(
        [rng.normal(base_area[t], base_area[t] * 0.18) for t in property_type]
    ).clip(320, 4200)

    bedrooms = np.clip((area_sqft / 420 + rng.normal(0, 0.5, N_ROWS)).round(), 1, 6).astype(int)
    bathrooms = np.clip((bedrooms / 2 + rng.normal(0, 0.4, N_ROWS)).round(), 1, 4).astype(int)

    year_built = rng.integers(1890, 2024, N_ROWS)
    distance_to_station_km = np.abs(rng.gamma(2.0, 0.9, N_ROWS)).round(2).clip(0.1, 12.0)

    epc_rating = rng.choice(EPC_RATINGS, size=N_ROWS, p=[0.07, 0.15, 0.26, 0.27, 0.16, 0.09])
    epc_value = np.array([len(EPC_RATINGS) - EPC_RATINGS.index(r) for r in epc_rating])

    condition_rating = rng.integers(1, 6, N_ROWS).astype(float)
    # ~12% missing, so imputation has to happen -- and has to happen per fold.
    condition_rating[rng.random(N_ROWS) < 0.12] = np.nan

    listed_date = pd.to_datetime("2023-01-01") + pd.to_timedelta(
        rng.integers(0, 730, N_ROWS), unit="D"
    )

    # The generative relationship. Written out rather than hidden in a formula
    # so the SHAP results can be checked against what actually drives the price:
    # area dominates, neighbourhood is second, condition and EPC are real but
    # small, and distance to the station works against the price.
    price = (
        118.0 * area_sqft
        + 9_500 * bedrooms
        + 7_200 * bathrooms
        - 6_400 * distance_to_station_km
        + 3_100 * epc_value
        + 4_800 * np.nan_to_num(condition_rating, nan=3.0)
        + 240 * (year_built - 1890)
    )
    price *= np.array([NEIGHBOURHOODS[n][0] for n in neighbourhood])
    price *= np.array([PROPERTY_TYPES[t] for t in property_type])
    # Multiplicative noise, which is what gives prices their right-skew.
    price *= rng.lognormal(0, 0.11, N_ROWS)

    frame = pd.DataFrame(
        {
            "listing_id": [f"LST-{i:05d}" for i in range(1, N_ROWS + 1)],
            "agent_email": [
                f"agent{rng.integers(1, 41)}@example-estates.com" for _ in range(N_ROWS)
            ],
            "listed_date": listed_date.strftime("%Y-%m-%d"),
            "neighbourhood": neighbourhood,
            "property_type": property_type,
            "area_sqft": area_sqft.round(0).astype(int),
            "bedrooms": bedrooms,
            "bathrooms": bathrooms,
            "year_built": year_built,
            "epc_rating": epc_rating,
            "condition_rating": condition_rating,
            "distance_to_station_km": distance_to_station_km,
            "sale_price": price.round(-2).astype(int),
        }
    )

    # A handful of exact duplicates, so cleaning has something real to remove and
    # the report's "rows removed" figure is not always zero.
    duplicated = frame.sample(6, random_state=SEED)
    return pd.concat([frame, duplicated], ignore_index=True)


if __name__ == "__main__":
    frame = build()
    frame.to_csv("data/examples/house_prices.csv", index=False)
    print(f"{len(frame)} rows, {frame.shape[1]} columns")
    print(f"sale_price: {frame.sale_price.min():,} to {frame.sale_price.max():,}")
    print(f"median: {frame.sale_price.median():,.0f}")
    print(f"missing condition_rating: {frame.condition_rating.isna().sum()}")
