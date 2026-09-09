"""Write the build-progress list in README.md from ui/build_status.py.

The README list and the landing page's table said the same thing in two places
and nothing made them say it at the same time. One of them fell four sections
behind and told visitors feature engineering had not been started.

This script makes the README the *generated* copy: ``ui/build_status.py`` holds
the statuses, the landing page renders them directly, and this rewrites the
block between the two markers in the README. ``--check`` exits non-zero instead
of writing, which is what ``tests/test_build_status.py`` runs -- so drift fails
the suite on a pull request rather than being noticed by a reader months later.

    python scripts/sync_build_status.py           # rewrite the block
    python scripts/sync_build_status.py --check   # fail if it is out of date

The third copy, the published progress report, cannot be generated from here --
it is artifact HTML living outside the repository. It stays hand-updated, and
that is now the only copy that can drift.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"

sys.path.insert(0, str(ROOT / "ui"))

from build_status import as_markdown_checklist  # noqa: E402

START = "<!-- build-status:start -->"
END = "<!-- build-status:end -->"


def render(readme: str) -> str:
    """Return ``readme`` with the marked block replaced by the generated list."""
    try:
        head, rest = readme.split(START, 1)
        _, tail = rest.split(END, 1)
    except ValueError as exc:  # pragma: no cover - a missing marker is a repo bug
        raise SystemExit(
            f"{README} is missing the {START} / {END} markers, so there is "
            "nowhere to write the build-progress list."
        ) from exc
    return f"{head}{START}\n\n{as_markdown_checklist()}\n\n{END}{tail}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="exit 1 if the README is out of date, without writing to it",
    )
    args = parser.parse_args()

    current = README.read_text()
    updated = render(current)

    if current == updated:
        return 0
    if args.check:
        print(
            "README.md's build-progress list is out of date with "
            "ui/build_status.py. Run: python scripts/sync_build_status.py",
            file=sys.stderr,
        )
        return 1
    README.write_text(updated)
    print(f"Updated the build-progress list in {README}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
