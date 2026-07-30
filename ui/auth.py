"""A shared-secret gate for the Streamlit app.

The application has no user accounts -- every job belongs to a single
``DEV_USER_ID`` and ``GET /jobs`` returns all of them. That is fine on a laptop
and unacceptable the moment a URL exists, because anyone holding it could read
every dataset ever uploaded and spend the project's API quota.

So this is a deliberately small thing: one shared secret, checked before any page
renders. It is not authentication -- there are no users to authenticate -- it is
a lock on a door that otherwise has none.

**Only the UI is ever exposed.** The tunnel publishes port 8501 and nothing else;
the API stays on the Docker network where only the UI can reach it. That is why a
gate here is sufficient, and why there is no token plumbing through twenty
``requests`` calls.

Unset ``AUTODS_ACCESS_TOKEN`` and the gate disappears entirely, so local
development and the test suite are untouched by it.
"""

from __future__ import annotations

import hmac
import os

import streamlit as st

TOKEN_VAR = "AUTODS_ACCESS_TOKEN"
_SESSION_KEY = "_autods_access_granted"
# The query parameter, so a single link can carry the key and an examiner does
# not have to copy a token into a box.
QUERY_KEY = "k"


def _matches(supplied: str, expected: str) -> bool:
    """Constant-time comparison, so the check cannot be timed character by character."""
    return hmac.compare_digest(supplied.strip(), expected)


def require_access() -> None:
    """Stop the page unless the caller holds the shared secret.

    Call at the top of every page. Streamlit's multipage routing lets anyone
    deep-link straight to ``/Results``, so gating only the entry point would gate
    nothing at all.
    """
    expected = os.getenv(TOKEN_VAR, "").strip()
    if not expected:
        # No secret configured: the stack is private, and the gate would only be
        # in the way. This is the local-development path.
        return

    if st.session_state.get(_SESSION_KEY):
        return

    # A key in the URL, so the link itself is the credential.
    supplied = st.query_params.get(QUERY_KEY, "")
    if supplied and _matches(supplied, expected):
        st.session_state[_SESSION_KEY] = True
        # Drop it from the address bar so the secret is not left sitting in a
        # screen share, a screenshot or the browser history of a shared machine.
        try:
            del st.query_params[QUERY_KEY]
        except KeyError:  # pragma: no cover - already absent
            pass
        return

    _render_locked_page(expected)


def _render_locked_page(expected: str) -> None:
    st.title("AutoDS")
    st.caption("This deployment is private.")

    entered = st.text_input(
        "Access key", type="password", placeholder="Paste the key you were sent"
    )
    if entered:
        if _matches(entered, expected):
            st.session_state[_SESSION_KEY] = True
            st.rerun()
        else:
            st.error("That key is not right.")

    st.caption(
        "There are no user accounts — this is a single shared key protecting a "
        "demo instance. If you were sent a link containing one, open that link "
        "rather than this page."
    )
    st.stop()
