"""Cluster profiling -- the LLM writes a sentence per discovered group (spec 9).

The division of labour is the project's usual one (spec 6.2): code measures, the
model writes. ``clustering.py`` has already computed how each group departs from
the dataset average; this agent turns those measurements into a sentence a
non-technical reader understands. The model never sees the data, never sees a
row, and never decides which features matter -- it is handed the differences and
asked to phrase them.

That matters more here than it looks. "Describe cluster 2" invites a model to
invent a persona; "cluster 2 is 33% of rows, income is higher than average and
the city is mostly Leeds -- describe it" does not leave much room to make
anything up, and anything it did invent would be visibly contradicted by the
numbers printed beside it on the Results page.

Best-effort, like every LLM call in the pipeline. Without a key the clusters are
still found, still plotted and still profiled numerically; only the prose is
missing, and the UI falls back to showing the measured differences directly.
"""

from __future__ import annotations

import logging

from pydantic import BaseModel, Field

from app.core.llm.base import LLMClient, LLMError, ModelTier, UsageCallback, system, user
from app.core.llm.structured import structured_complete
from app.ml.contracts import ClusterProfile

logger = logging.getLogger(__name__)

AGENT_NAME = "cluster_profiles"

# Long enough to be useful, short enough that it cannot drift into invention.
_MAX_WORDS = 30

_SYSTEM = (
    "You are describing groups of records that a clustering algorithm found in a "
    "dataset. For each group you are given its size and the ways it differs from "
    "the dataset average. Write one short sentence per group, in plain language a "
    "non-technical reader would understand, saying what kind of records it "
    "contains. Base every description only on the differences given -- do not "
    "invent characteristics, do not speculate about causes, and do not give "
    "advice. If a group has no listed differences, say it is unremarkable. Keep "
    f"each description under {_MAX_WORDS} words."
)


class _ClusterDescription(BaseModel):
    cluster: int
    description: str


class _ClusterDescriptions(BaseModel):
    clusters: list[_ClusterDescription] = Field(default_factory=list)


def describe_clusters(
    profiles: list[ClusterProfile],
    *,
    client: LLMClient | None = None,
    on_usage: UsageCallback | None = None,
) -> list[ClusterProfile]:
    """Fill in each profile's ``description``. Never raises.

    Returns the profiles either way -- with prose when a model was available, and
    unchanged when it was not.
    """
    if client is None or not profiles:
        logger.info("Cluster profiling: no LLM configured, keeping measured differences only")
        return profiles

    try:
        result = structured_complete(
            client,
            [system(_SYSTEM), user(_prompt(profiles))],
            _ClusterDescriptions,
            tier=ModelTier.SMALL,
            on_usage=on_usage,
        )
    except LLMError as exc:
        logger.warning("Cluster profiling skipped: %s", exc)
        return profiles
    except Exception:  # pragma: no cover - defensive; EDA must not fail a job
        logger.exception("Unexpected error while profiling clusters")
        return profiles

    # Match on cluster number and ignore anything for a group that does not
    # exist -- the same validate-against-reality rule schema detection applies to
    # the LLM's column names.
    described = {d.cluster: d.description.strip() for d in result.data.clusters}
    for profile in profiles:
        text = described.get(profile.cluster)
        if text:
            profile.description = text

    logger.info(
        "Cluster profiling: described %d of %d group(s)",
        sum(1 for p in profiles if p.description),
        len(profiles),
    )
    return profiles


def _prompt(profiles: list[ClusterProfile]) -> str:
    """The measured differences, one block per group.

    No column values, no rows -- just the comparisons code already computed. That
    keeps the prompt small (spec 10's input-token cap) and keeps the model
    describing evidence rather than imagining data.
    """
    lines: list[str] = []
    for profile in profiles:
        lines.append(
            f"Group {profile.cluster}: {profile.size:,} records "
            f"({profile.share:.0%} of the dataset)."
        )
        if profile.distinguishing_features:
            for name, description in profile.distinguishing_features.items():
                lines.append(f"  - {name}: {description}")
        else:
            lines.append("  - no notable differences from the dataset average")
        lines.append("")
    return "\n".join(lines).strip()
