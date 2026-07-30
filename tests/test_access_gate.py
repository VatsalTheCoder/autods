"""The shared-secret gate in front of the Streamlit app.

The application has no user accounts: every job belongs to one ``DEV_USER_ID``
and ``GET /jobs`` returns all of them. On a laptop that is fine. Behind a public
URL it means anyone holding the link can read every dataset ever uploaded, so
this gate is the only thing between a tunnel and that.

A lock is worth what it is worth *closed*, so most of these tests are about it
refusing rather than admitting -- and the one that matters most is that a page
reached by deep link is gated too, because Streamlit's routing means
``/Results`` is reachable without ever passing through the front page.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

UI = Path(__file__).resolve().parents[1] / "ui"
HOME = str(UI / "Home.py")
PAGES = sorted(str(p) for p in (UI / "pages").glob("*.py"))

TOKEN = "correct-horse-battery-staple"


def _run(page: str, *, token: str | None, query: dict | None = None) -> AppTest:
    app = AppTest.from_file(page, default_timeout=30)
    if token is not None:
        app.query_params.update(query or {})
    return app


@pytest.fixture
def gated(monkeypatch):
    monkeypatch.setenv("AUTODS_ACCESS_TOKEN", TOKEN)


class TestWithNoTokenConfigured:
    """The local-development path: no secret, no gate, nothing in the way."""

    def test_the_page_renders_normally(self, monkeypatch):
        monkeypatch.delenv("AUTODS_ACCESS_TOKEN", raising=False)
        app = AppTest.from_file(HOME, default_timeout=30).run()
        assert not app.exception
        # The real page, not the lock screen.
        assert any("How it works" in s.value for s in app.subheader)

    def test_an_empty_token_is_treated_as_unset(self, monkeypatch):
        """Whitespace in a .env should not accidentally lock everyone out."""
        monkeypatch.setenv("AUTODS_ACCESS_TOKEN", "   ")
        app = AppTest.from_file(HOME, default_timeout=30).run()
        assert not app.exception
        assert any("How it works" in s.value for s in app.subheader)


class TestTheGateRefuses:
    def test_no_key_stops_the_page(self, gated):
        app = AppTest.from_file(HOME, default_timeout=30).run()
        assert not app.exception
        assert any("private" in c.value.lower() for c in app.caption)
        # The page's real content must not have rendered behind the lock.
        assert not any("How it works" in s.value for s in app.subheader)

    def test_a_wrong_key_in_the_url_stops_the_page(self, gated):
        app = AppTest.from_file(HOME, default_timeout=30)
        app.query_params["k"] = "not-the-key"
        app.run()
        assert not any("How it works" in s.value for s in app.subheader)

    def test_a_prefix_of_the_key_is_not_enough(self, gated):
        app = AppTest.from_file(HOME, default_timeout=30)
        app.query_params["k"] = TOKEN[:-1]
        app.run()
        assert not any("How it works" in s.value for s in app.subheader)

    @pytest.mark.parametrize("page", PAGES)
    def test_every_page_is_gated_not_just_the_entry_point(self, gated, page):
        """Streamlit routing makes /Results reachable without the front page.

        Gating only Home.py would gate nothing at all: an examiner -- or anyone
        with the URL -- can deep-link straight past it.
        """
        app = AppTest.from_file(page, default_timeout=30).run()
        assert not app.exception
        assert any("private" in c.value.lower() for c in app.caption), (
            f"{Path(page).name} rendered without the gate"
        )


class TestTheGateAdmits:
    def test_the_right_key_in_the_url_lets_the_page_render(self, gated):
        app = AppTest.from_file(HOME, default_timeout=30)
        app.query_params["k"] = TOKEN
        app.run()
        assert not app.exception
        assert any("How it works" in s.value for s in app.subheader)

    def test_surrounding_whitespace_is_tolerated(self, gated):
        """A key copied out of an email arrives with a space on the end."""
        app = AppTest.from_file(HOME, default_timeout=30)
        app.query_params["k"] = f"  {TOKEN} "
        app.run()
        assert any("How it works" in s.value for s in app.subheader)

    def test_the_key_is_removed_from_the_address_bar(self, gated):
        """So the secret is not left in a screenshot, screen share or history."""
        app = AppTest.from_file(HOME, default_timeout=30)
        app.query_params["k"] = TOKEN
        app.run()
        assert "k" not in app.query_params


class TestTheComparison:
    def test_it_is_constant_time(self):
        """Timing a character-by-character comparison would leak the key."""
        import inspect

        from ui import auth  # noqa: PLC0415 - imported here to keep the module optional

        assert "compare_digest" in inspect.getsource(auth._matches)
