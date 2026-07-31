"""Generate the wide dataset (``data/wide_telemetry.csv``) — 122 columns.

Committed as a generator rather than as its output, for the same reason as
``make_house_prices.py``: the generative relationship stays inspectable, so a
reader can see that the signal is real and the noise deliberate. The output is
gitignored (only ``data/examples/*.csv`` is committed) and the sweep manifest
marks it ``in_repo: false``.

**Why this dataset exists.** ``BUILD_PLAN.md`` section 9 says, of the critic and
report agents: "you've tested it on a dataset with 100+ columns, because that's
where the token cap breaks, not on your tidy 8-column demo file." Nothing has.
Every run so far has been 10–13 columns. The free tier's binding constraint is
input tokens per minute, and three things scale with column count:

1. the feature-strategy agent's *output* — one JSON decision per column, so 122
   of them in a single structured response;
2. the critic's and report writer's *input* — the summaries they review carry
   per-column detail;
3. the encoded feature space — 15 categoricals at 4–9 levels each expand to
   several hundred columns after one-hot, which is what SHAP then has to map
   back to names a person recognises.

Any of the three could give first. The point of the dataset is to find out which,
on evidence rather than by reasoning about it.

**Rows are kept low (500) on purpose.** Width is the variable under test; adding
rows would only make every run slower without probing anything new.

The columns are chosen to put pressure on specific machinery rather than to
model a real machine:

- ``sensor_id`` — identifier, which cleaning should drop rather than model.
- ``technician_email`` — PII, to be caught at the checkpoint.
- ``reading_ts`` — a timestamp, which becomes calendar parts.
- ``sig_00``–``sig_59`` — 60 numeric channels, of which **6** actually drive the
  target. The other 54 are noise, which is what feature selection is for.
- ``agg_00``–``agg_19`` — rolling aggregates *derived from* the signal columns,
  so they are collinear with them by construction. Wide real-world data is
  almost always redundant like this, and it is what makes a 122×122 correlation
  matrix worth computing.
- ``mode_00``–``mode_14`` — 15 categoricals, 4–9 levels each: the one-hot
  expansion.
- ``grade_00``–``grade_07`` — ordinal A–E, so the strategy agent has eight
  chances to get ordering right where one-hot would throw it away.
- ``sparse_00``–``sparse_09`` — 70–95% missing, the range where imputing and
  dropping are both defensible.
- ``const_*`` — three columns that never vary, which cleaning must remove.
- ``note_text``, ``batch_code`` — high-cardinality strings, for the frequency
  encoder and the ``text`` role.
- ``failed`` — binary, ~18% positive, so SMOTE has a reason to run.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SEED = 20260731
N_ROWS = 500

N_SIGNALS = 60
N_AGGREGATES = 20
N_MODES = 15
N_GRADES = 8
N_SPARSE = 10

# The six signal channels that genuinely move the target, and their weights.
# Written out rather than sampled so the SHAP output can be checked against the
# truth: if these six are not the top of the importance ranking, something in
# the modelling or the name mapping is wrong -- and on a 122-column frame that
# is a much sharper test than it is on eight columns.
DRIVERS: dict[int, float] = {3: 1.6, 11: -1.2, 19: 0.9, 28: -0.7, 41: 1.1, 52: 0.5}

GRADES = ["A", "B", "C", "D", "E"]


def build() -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    columns: dict[str, object] = {}

    columns["sensor_id"] = [f"SNS-{i:05d}" for i in range(1, N_ROWS + 1)]
    columns["technician_email"] = [
        f"tech{rng.integers(1, 25)}@example-plant.com" for _ in range(N_ROWS)
    ]
    columns["reading_ts"] = (
        pd.to_datetime("2025-03-01") + pd.to_timedelta(rng.integers(0, 400, N_ROWS), unit="D")
    ).strftime("%Y-%m-%d")

    # The signal channels. Scales deliberately differ by two orders of magnitude
    # across the block, so a run that skips scaling is visibly worse rather than
    # merely theoretically wrong.
    signals = np.zeros((N_ROWS, N_SIGNALS))
    for i in range(N_SIGNALS):
        scale = 10.0 ** rng.integers(-1, 3)
        signals[:, i] = rng.normal(0, 1, N_ROWS) * scale
        columns[f"sig_{i:02d}"] = signals[:, i].round(4)

    # Aggregates derived from the signals: collinear on purpose.
    for i in range(N_AGGREGATES):
        source = rng.choice(N_SIGNALS, size=3, replace=False)
        blended = signals[:, source].mean(axis=1) + rng.normal(0, 0.05, N_ROWS)
        columns[f"agg_{i:02d}"] = blended.round(4)

    for i in range(N_MODES):
        n_levels = int(rng.integers(4, 10))
        levels = [f"m{i:02d}_{chr(97 + j)}" for j in range(n_levels)]
        columns[f"mode_{i:02d}"] = rng.choice(levels, size=N_ROWS)

    for i in range(N_GRADES):
        columns[f"grade_{i:02d}"] = rng.choice(GRADES, size=N_ROWS, p=[0.1, 0.2, 0.4, 0.2, 0.1])

    # Sparse columns. The missing fraction is swept across the block so the
    # cleaning threshold is straddled rather than cleared or failed wholesale:
    # some of these should be dropped and some imputed, and which is which is a
    # decision the pipeline has to make 10 times.
    for i in range(N_SPARSE):
        values = rng.normal(50, 12, N_ROWS)
        missing_fraction = 0.70 + i * 0.025
        values[rng.random(N_ROWS) < missing_fraction] = np.nan
        columns[f"sparse_{i:02d}"] = values.round(3)

    columns["const_region"] = "EU-WEST"
    columns["const_schema_version"] = 4
    columns["const_calibrated"] = True

    # High cardinality, two ways: one nearly unique per row, one with real
    # repetition. They take different routes -- the first is closer to an
    # identifier, the second is genuinely a frequency-encodable category.
    columns["note_text"] = [
        f"run {rng.integers(1000, 9999)} on line {rng.integers(1, 9)} ok" for _ in range(N_ROWS)
    ]
    columns["batch_code"] = [f"B{rng.integers(1, 120):03d}" for _ in range(N_ROWS)]

    # The target. Only the six drivers and one categorical contribute; every
    # other column is noise the pipeline has to see past.
    logit = np.zeros(N_ROWS)
    for index, weight in DRIVERS.items():
        channel = signals[:, index]
        # Standardised before weighting, so the weights above mean what they say
        # regardless of the scale that channel happened to be given.
        logit += weight * (channel - channel.mean()) / (channel.std() or 1.0)

    # One categorical effect, so the target is not purely numeric-driven and the
    # encoding decisions matter to the score rather than only to the pipeline.
    mode_00 = np.asarray(columns["mode_00"])
    logit += np.where(mode_00 == np.unique(mode_00)[0], 0.8, 0.0)

    logit += rng.normal(0, 0.6, N_ROWS)
    # Shifted to land near 18% positive: imbalanced enough for SMOTE to be worth
    # running, not so imbalanced that a fold loses the minority class entirely.
    probability = 1 / (1 + np.exp(-(logit - 3.0)))
    columns["failed"] = (rng.random(N_ROWS) < probability).astype(int)

    return pd.DataFrame(columns)


if __name__ == "__main__":
    frame = build()
    frame.to_csv("data/wide_telemetry.csv", index=False)

    print(f"{len(frame)} rows, {frame.shape[1]} columns")
    print(f"failed: {frame.failed.mean():.1%} positive ({frame.failed.sum()} rows)")
    print(f"drivers: {', '.join(f'sig_{i:02d}' for i in DRIVERS)}")
    one_hot_width = sum(
        frame[c].nunique() for c in frame.columns if c.startswith(("mode_", "grade_"))
    )
    print(f"categorical levels to encode: {one_hot_width}")
    print(f"columns >50% missing: {(frame.isna().mean() > 0.5).sum()}")
