"""The build status, stated once.

Three places told a visitor how far the build had got: this table on the landing
page, the "Build progress" list in ``README.md``, and the published progress
report. Nothing kept them in step, and the drift was not hypothetical -- the
landing page once spent four sections announcing that feature engineering had
not been started while Sections 7 to 10 were merging.

Two of those three now derive from the list below: the landing page renders it,
and ``scripts/sync_build_status.py`` writes it into the README between markers,
with ``--check`` failing the test suite when the two disagree. The progress
report is published artifact HTML and is still updated by hand, which is why
``REPORT_URL`` sits here next to the data it has to match rather than in a
comment somewhere else.

Living in ``ui/`` rather than ``app/`` for a boring reason: the Streamlit
frontend deliberately imports nothing from ``app``, and ``ui/`` is mounted into
both the ui and api containers, so a test can read it without a new mount.
"""

from __future__ import annotations

from dataclasses import dataclass

REPORT_URL = "https://claude.ai/code/artifact/2fb6fa9c-b8bc-448e-b653-3145e3c02794"

DONE = "done"
IN_PROGRESS = "in progress"
NOT_STARTED = "not started"

_MARKS = {DONE: "✅", IN_PROGRESS: "🟡", NOT_STARTED: "⬜"}


@dataclass(frozen=True)
class Section:
    """One build-plan section, as both surfaces describe it.

    ``note`` is the qualifier that stops a status being read as more than it is
    -- Section 11 is "in progress" with a runbook written and nothing hosted,
    and a bare 🟡 would let a reader assume the URL exists.

    Two titles, because the two surfaces phrase them differently and always did:
    the README spells a section out, the landing page's narrow status column
    abbreviates. What has to agree between them is the *status*, not the prose,
    so single-sourcing the status while keeping both wordings is the honest
    shape rather than flattening one surface into the other.
    """

    number: int
    title: str
    short_title: str
    status: str
    milestone: str | None = None
    note: str | None = None

    @property
    def mark(self) -> str:
        return _MARKS[self.status]

    @property
    def merged(self) -> bool:
        return self.status == DONE


# Ordered by section number, and every section in BUILD_PLAN.md must appear:
# omission was the original bug, so a short list is the failure to guard.
SECTIONS: tuple[Section, ...] = (
    Section(0, "Skeleton: Docker Compose, config, S3 abstraction, health check", "Skeleton", DONE),
    Section(1, "Upload: CSV to S3, jobs and artifacts tables", "Upload", DONE),
    Section(
        2, "LLM client: structured output, rate limiting, token accounting", "LLM client", DONE
    ),
    Section(
        3, "Schema detection and the human checkpoint", "Schema detection & human checkpoint", DONE
    ),
    Section(
        4, "Background worker: Celery, LangGraph, progress tracking", "Background worker", DONE
    ),
    Section(5, "Vertical slice: end-to-end demo", "Vertical slice", DONE, milestone="M1"),
    Section(6, "EDA and clustering", "EDA & clustering", DONE, milestone="M2"),
    Section(7, "Feature engineering", "Feature engineering", DONE, milestone="M3"),
    Section(
        8,
        "Final training, SHAP, prediction",
        "Final training, SHAP & prediction",
        DONE,
        milestone="M4",
    ),
    Section(9, "Critic and report", "Critic & report", DONE, milestone="M5"),
    Section(10, "RAG chat", "RAG chat", DONE, milestone="M6"),
    Section(
        11,
        "AWS deployment",
        "AWS deployment",
        IN_PROGRESS,
        milestone="M7",
        # Deliberately says "never run", not "not finished". The deployment
        # config, scripts and guide are all written and merged; what has never
        # happened is executing any of it against a real AWS account, and a
        # reader deciding whether to trust docs/DEPLOYMENT.md needs that fact
        # rather than a percentage.
        note="config and guide written; never run against real AWS",
    ),
    Section(
        12,
        "Testing and documentation",
        "Testing & docs",
        IN_PROGRESS,
        note="diagrams and API reference written; consolidation pass outstanding",
    ),
)

MERGED = sum(1 for section in SECTIONS if section.merged)
TOTAL = len(SECTIONS)
HEADLINE = f"{MERGED} of {TOTAL} sections merged"


def _milestone(section: Section) -> str:
    return f" *(milestone {section.milestone})*" if section.milestone else ""


def as_markdown_table() -> str:
    """The landing page's rendering: a table, one row per section."""
    lines = ["| Section | Status |", "|---|---|"]
    for section in SECTIONS:
        status = f"{section.mark} {section.status}"
        if section.note:
            status = f"{status} — {section.note}"
        lines.append(
            f"| {section.number} · {section.short_title}{_milestone(section)} | {status} |"
        )
    return "\n".join(lines)


def as_markdown_checklist() -> str:
    """The README's rendering: a checklist, because that is what it always was.

    Deliberately not the same shape as the table. The two surfaces read
    differently and always did; what has to agree is the *content*, and
    generating both from one list is what makes that true.
    """
    lines = []
    for section in SECTIONS:
        box = "x" if section.merged else " "
        line = f"- [{box}] **Section {section.number}** — {section.title}{_milestone(section)}"
        if section.note:
            line = f"{line} — {section.note}"
        lines.append(line)
    return "\n".join(lines)
