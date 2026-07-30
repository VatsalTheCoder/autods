"""The landing page's build-progress table, rendered headlessly.

The table is hand-maintained and it went stale in the way hand-maintained tables
do: it stopped at Section 7 and stayed there while Sections 7, 8, 9 and 10
merged. For four sections the first page a visitor saw announced that feature
engineering had *not been started*, under a heading claiming to describe build
progress.

No test can check the statuses are truthful -- that is a fact about the world,
not about the code. What a test can check is the failure that actually happened:
the table quietly ending early. Every section in the build plan has to appear.
"""

from __future__ import annotations

import re
from pathlib import Path

from streamlit.testing.v1 import AppTest

# Named Home.py, not app.py: Streamlit labels the sidebar entry from the
# file stem, so the entrypoint's name is user-visible navigation text.
PAGE = str(Path(__file__).resolve().parents[1] / "ui" / "Home.py")
BUILD_PLAN = Path(__file__).resolve().parents[1] / "BUILD_PLAN.md"


def _rendered() -> str:
    app = AppTest.from_file(PAGE, default_timeout=30).run()
    assert not app.exception, app.exception
    return "\n".join(block.value for block in app.markdown)


class TestEverySectionAppears:
    def test_no_section_is_missing_from_the_table(self):
        """The bug was omission, not a wrong label, so absence is what to assert."""
        rendered = _rendered()
        for section in range(13):
            assert re.search(rf"\|\s*{section} · ", rendered), (
                f"Section {section} is missing from the build-progress table. "
                "A table that ends early reads as 'not started' to a visitor."
            )

    def test_the_table_matches_the_build_plan_it_points_at(self):
        """The page tells the reader to see BUILD_PLAN.md, so the two must agree.

        If a section is ever added to the plan, this fails rather than letting
        the landing page silently describe a shorter project than the one there.
        """
        planned = {int(m) for m in re.findall(r"^## Section (\d+)", BUILD_PLAN.read_text(), re.M)}
        rendered = _rendered()
        shown = {int(m) for m in re.findall(r"\|\s*(\d+) · ", rendered)}
        assert shown == planned, f"table shows {sorted(shown)}, plan has {sorted(planned)}"


class TestTheStatusesAreLegible:
    def test_every_row_carries_a_status(self):
        rendered = _rendered()
        rows = [line for line in rendered.splitlines() if re.match(r"\|\s*\d+ · ", line)]
        assert len(rows) == 13
        for row in rows:
            assert any(mark in row for mark in ("✅", "🟡", "⬜")), row

    def test_the_finished_sections_are_not_marked_unstarted(self):
        """Sections 0-10 are merged to main; saying otherwise is the original bug."""
        rendered = _rendered()
        for section in range(11):
            row = next(
                line for line in rendered.splitlines() if re.match(rf"\|\s*{section} · ", line)
            )
            assert "⬜" not in row, f"Section {section} is merged but shown as not started"
