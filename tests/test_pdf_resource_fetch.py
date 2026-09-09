"""The report document cannot make the renderer fetch anything.

The filename is the one part of this document whose bytes a user chooses, and
it lands inside a Markdown code span. A backtick in the name closed the span,
and everything after it was read as Markdown -- so a file called

    report`![x](http://host/p.png)`.csv

became an ``<img>`` in the HTML handed to WeasyPrint, whose default fetcher
honours http, https and file. The renderer runs in the worker, inside the
network where Postgres and object storage live, and the render still succeeded,
so nothing surfaced.

Both halves are tested: the filename is neutralised at the source, and the
renderer refuses resources regardless of how one got into the document. Either
alone would be one edit away from being the only thing in the way.

No network is touched here. The refusal is asserted through the fetcher itself,
and the one end-to-end check intercepts name resolution at the socket layer so a
regression cannot quietly reach out during a test run.
"""

from __future__ import annotations

import socket

import pytest

from app.ml.contracts import EvaluationReport
from app.ml.pdf import RemoteResourceRefused, _refuse_remote_resources, render_pdf
from app.ml.report import _heading, _inline_code

_EVALUATION = EvaluationReport(
    task_type="regression",
    target_column="y",
    model_name="m",
    n_folds=5,
    cv_strategy="KFold",
    n_rows=10,
    n_features=2,
)


class TestTheFilenameCannotLeaveItsCodeSpan:
    def test_a_backtick_is_neutralised(self):
        assert "`" not in _inline_code("report`x`.csv")

    def test_newlines_cannot_start_a_new_block(self):
        assert "\n" not in _inline_code("report\n# Heading\n.csv")
        assert "\r" not in _inline_code("report\r.csv")

    def test_an_ordinary_name_is_left_alone(self):
        assert _inline_code("house_prices.csv") == "house_prices.csv"

    def test_the_injected_image_never_reaches_the_markdown(self):
        evil = "report`![x](http://attacker.invalid/p.png)`.csv"
        heading = _heading(evil, _EVALUATION)
        # The text survives -- a user should still see what they named the file --
        # but it cannot close the span it sits in.
        assert "attacker.invalid" in heading
        assert "`![x]" not in heading


class TestTheRendererRefusesResources:
    def test_the_fetcher_refuses_every_scheme(self):
        for url in (
            "http://attacker.invalid/p.png",
            "https://attacker.invalid/p.png",
            "file:///etc/hostname",
            "http://169.254.169.254/latest/meta-data/",
        ):
            with pytest.raises(RemoteResourceRefused):
                _refuse_remote_resources(url)

    def test_a_document_carrying_an_image_still_renders(self):
        """A refusal must not cost the run its report -- the PDF is best-effort."""
        markdown = "# Title\n\n![x](http://attacker.invalid/p.png)\n\nBody text.\n"
        assert render_pdf(markdown, title="probe")

    def test_an_image_already_in_the_document_is_not_fetched(self):
        """The fetcher's own test, independent of how the image got there.

        Added after a mutation check: removing the url_fetcher and leaving the
        filename escaping in place kept every other test green, because the only
        socket-intercepting test used a filename the other half had already
        neutralised -- so nothing reached the renderer to be fetched. This one
        puts the image in the Markdown directly, so it tests the renderer rather
        than the escaping.
        """
        attempted: list[str] = []
        real = socket.getaddrinfo

        def record(host, *args, **kwargs):
            attempted.append(host)
            raise socket.gaierror(f"[blocked in tests] {host}")

        socket.getaddrinfo = record
        try:
            render_pdf(
                "# Title\n\n![x](http://attacker.invalid/p.png)\n",
                title="probe",
            )
        finally:
            socket.getaddrinfo = real

        assert attempted == [], f"the renderer tried to resolve {attempted}"

    def test_rendering_resolves_no_hostname(self):
        """The end-to-end assertion, with the network cut at the socket layer.

        Belt and braces against the fetcher being bypassed by some future path:
        if anything resolves a name during a render, this fails rather than
        silently making a request from the worker.
        """
        attempted: list[str] = []
        real = socket.getaddrinfo

        def record(host, *args, **kwargs):
            attempted.append(host)
            raise socket.gaierror(f"[blocked in tests] {host}")

        socket.getaddrinfo = record
        try:
            evil = "report`![x](http://attacker.invalid/p.png)`.csv"
            render_pdf(_heading(evil, _EVALUATION), title="probe")
        finally:
            socket.getaddrinfo = real

        assert attempted == [], f"the renderer tried to resolve {attempted}"
