"""Row sampling -- the one reduction that is safe outside a fold (spec 7.3).

Everything else in ``app/ml`` that touches data is careful to happen inside a
cross-validation fold, because it *learns* something: a median, a scale, a
category frequency. Sampling learns nothing. It removes rows, and no quantity
computed from the discarded rows travels into the kept ones, so doing it once up
front leaves the fold discipline exactly as it was. That is why this is a plain
function over a DataFrame rather than a pipeline step.

It exists because the roster costs four models times five folds -- twenty fits --
and on a large upload that turns a demo into a coffee break. The cost is honest
and stated: a model trained on 20,000 of 200,000 rows is a model trained on
20,000 rows, and the report says so rather than implying the whole dataset was
used.

**Stratified for classification**, which is the part that matters. A uniform
random subset of a 99:1 dataset can easily contain no minority rows at all, and
the run would then either crash or report a meaningless score on a target that
no longer has two classes. Keeping the class proportions makes the sample a
smaller version of the dataset rather than a different one.
"""

from __future__ import annotations

import logging

import pandas as pd

from app.core.config import get_settings

logger = logging.getLogger(__name__)


def sample_frame(
    frame: pd.DataFrame,
    *,
    target: str,
    task_type: str,
    limit: int,
    random_seed: int | None = None,
) -> pd.DataFrame:
    """Take at most ``limit`` rows, keeping the target's shape.

    Returns the frame untouched when it is already small enough, so callers do
    not have to check first.
    """
    if len(frame) <= limit:
        return frame

    if random_seed is None:
        random_seed = get_settings().random_seed

    if task_type == "classification" and target in frame.columns:
        sampled = _stratified(frame, target=target, limit=limit, random_seed=random_seed)
        if sampled is not None:
            return sampled

    return frame.sample(n=limit, random_state=random_seed).sort_index()


def _stratified(
    frame: pd.DataFrame,
    *,
    target: str,
    limit: int,
    random_seed: int,
) -> pd.DataFrame | None:
    """A sample holding each class's share of the whole. ``None`` if impossible.

    Each class keeps its proportion, and every class keeps **at least one row**
    -- a rare class rounding to zero would silently drop a category from the
    problem, which is the failure this function exists to prevent. That floor
    means the result can be a row or two over ``limit`` on a target with many
    tiny classes; being slightly over a performance ceiling is a much better
    outcome than losing a class.
    """
    counts = frame[target].value_counts(dropna=False)
    if counts.empty:
        return None

    share = limit / len(frame)
    parts = []
    for value, count in counts.items():
        rows = frame[frame[target] == value] if pd.notna(value) else frame[frame[target].isna()]
        take = max(1, min(int(round(count * share)), len(rows)))
        parts.append(rows.sample(n=take, random_state=random_seed))

    sampled = pd.concat(parts).sort_index()
    logger.info(
        "Stratified sample: %d rows from %d, %d classes preserved",
        len(sampled),
        len(frame),
        len(counts),
    )
    return sampled
