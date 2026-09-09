"""The build status is stated once, and the copies of it are generated.

Three surfaces tell a reader how far the build has got: the landing page's
table, the README's checklist, and the published progress report. Keeping them
in step by hand did not work -- the landing page spent four sections announcing
that feature engineering had not been started while Sections 7 to 10 merged.

``ui/build_status.py`` is now the one place the statuses live. The landing page
renders it and ``scripts/sync_build_status.py`` writes it into the README, so
what is left to check is that nobody has edited a generated copy by hand and
that the source still covers every section the plan defines.

The progress report is artifact HTML outside the repository. Nothing here can
reach it, which is exactly why it is the only copy still able to drift.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUILD_PLAN = ROOT / "BUILD_PLAN.md"
README = ROOT / "README.md"

sys.path.insert(0, str(ROOT / "ui"))
sys.path.insert(0, str(ROOT / "scripts"))

from build_status import SECTIONS, as_markdown_checklist, as_markdown_table  # noqa: E402
from sync_build_status import render  # noqa: E402


class TestTheReadmeIsGeneratedAndCurrent:
    def test_the_readme_block_matches_the_source(self):
        """The failure this catches is someone editing the README copy by hand.

        It is the natural thing to do -- the list is right there in the file
        being edited -- and it puts the two surfaces back out of step silently.
        """
        current = README.read_text()
        assert render(current) == current, (
            "README.md's build-progress list is out of date with "
            "ui/build_status.py. Run: python scripts/sync_build_status.py"
        )

    def test_the_generated_block_is_not_empty(self):
        """A marker pair with nothing between it would satisfy the test above."""
        assert "- [x] **Section 0**" in README.read_text()


class TestTheSourceCoversThePlan:
    def test_every_section_in_the_plan_has_a_status(self):
        planned = {int(m) for m in re.findall(r"^## Section (\d+)", BUILD_PLAN.read_text(), re.M)}
        stated = {section.number for section in SECTIONS}
        assert stated == planned, f"status covers {sorted(stated)}, plan has {sorted(planned)}"

    def test_the_sections_are_in_order(self):
        numbers = [section.number for section in SECTIONS]
        assert numbers == sorted(numbers)


class TestTheTwoRenderingsAgree:
    """They are worded differently on purpose; what has to match is the status."""

    def test_both_surfaces_mark_the_same_sections_finished(self):
        table = as_markdown_table()
        checklist = as_markdown_checklist()
        for section in SECTIONS:
            table_row = next(
                line for line in table.splitlines() if line.startswith(f"| {section.number} · ")
            )
            checklist_row = next(
                line
                for line in checklist.splitlines()
                if line.startswith(f"- [x] **Section {section.number}**")
                or line.startswith(f"- [ ] **Section {section.number}**")
            )
            assert ("✅" in table_row) == ("- [x]" in checklist_row), (
                f"Section {section.number} is finished on one surface and not the other:\n"
                f"  {table_row}\n  {checklist_row}"
            )
