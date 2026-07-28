"""Markdown to PDF -- the report as a document a person can send (spec 7.12).

The report already exists as Markdown, assembled from the artifacts, so this is
a translation rather than a second document: markdown → HTML → PDF. That is the
whole reason WeasyPrint is here instead of ReportLab. ReportLab would mean
building the report a second time as flowables and keeping the two versions in
step by hand, and the first time someone edited only one of them the PDF and the
Markdown would start disagreeing about what the run found.

The stylesheet is inline and deliberately plain. This is a document that will be
read on a screen and occasionally printed, so it optimises for the things that
actually go wrong in a generated PDF: tables wider than the page, headings
stranded at the bottom of one, and code spans that overflow the margin. Colour
is used once, for severity in the review table, and even that degrades to
readable black if the PDF is printed without it.

**Nothing here can change what the report says.** It receives finished Markdown
and returns bytes. If this module has a bug the document looks wrong, not says
something wrong -- which is the correct place for a rendering layer's failures.
"""

from __future__ import annotations

import logging
import re
from datetime import UTC, datetime

logger = logging.getLogger(__name__)

AGENT_NAME = "pdf"

# Print CSS, and every rule in it earns its place against a failure a generated
# report actually hits.
_STYLESHEET = """
@page {
    size: A4;
    margin: 20mm 16mm 18mm 16mm;
    @bottom-center {
        content: counter(page) " of " counter(pages);
        font-family: -apple-system, "Segoe UI", Roboto, Helvetica, sans-serif;
        font-size: 8pt;
        color: #7C8481;
    }
}

body {
    font-family: -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    font-size: 10pt;
    line-height: 1.5;
    color: #15181A;
}

h1 { font-size: 20pt; line-height: 1.2; margin: 0 0 4pt; }
h2 {
    font-size: 13pt;
    margin: 16pt 0 6pt;
    padding-bottom: 3pt;
    border-bottom: 0.5pt solid #DDE0DA;
    /* A heading alone at the foot of a page reads as a missing section. */
    break-after: avoid;
    break-inside: avoid;
}
h3 { font-size: 11pt; margin: 12pt 0 4pt; break-after: avoid; }
p { margin: 0 0 6pt; }
ul, ol { margin: 0 0 8pt; padding-left: 14pt; }
li { margin-bottom: 2pt; }

/* Tables are where a generated PDF usually breaks: the leaderboard and the
   review table are both wide. Fixed layout plus wrapping keeps them inside the
   margin instead of running off the page. */
table {
    width: 100%;
    border-collapse: collapse;
    table-layout: fixed;
    font-size: 8.5pt;
    margin: 4pt 0 10pt;
    break-inside: auto;
}
th, td {
    border-bottom: 0.5pt solid #EAECE6;
    padding: 3pt 5pt;
    text-align: left;
    vertical-align: top;
    word-wrap: break-word;
    overflow-wrap: anywhere;
}
th {
    border-bottom: 0.75pt solid #DDE0DA;
    font-size: 7.5pt;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    color: #4A524F;
}
/* Repeat headers when a long table crosses a page, or the second page of the
   leaderboard is a grid of unlabelled numbers. */
thead { display: table-header-group; }
tr { break-inside: avoid; }

code {
    font-family: "SF Mono", Menlo, Consolas, monospace;
    font-size: 8.5pt;
    background: #F2F3EF;
    padding: 0.5pt 2pt;
    border-radius: 2pt;
    overflow-wrap: anywhere;
}

blockquote {
    margin: 6pt 0;
    padding-left: 8pt;
    border-left: 1.5pt solid #8C6714;
    color: #4A524F;
}

em { color: #4A524F; }

.footer {
    margin-top: 14pt;
    padding-top: 5pt;
    border-top: 0.5pt solid #DDE0DA;
    font-size: 7.5pt;
    color: #7C8481;
}
"""


class PdfError(RuntimeError):
    """The PDF could not be rendered, with a reason for the caller."""


def render_pdf(markdown_text: str, *, title: str = "AutoDS report") -> bytes:
    """Render finished Markdown to PDF bytes.

    Imports WeasyPrint lazily. It binds to system libraries at *render* time
    rather than import time, and keeping the import here means a machine without
    pango can still load this module, run the rest of the pipeline, and fail on
    the PDF alone with a message that says so.
    """
    try:
        import markdown as markdown_lib
        from weasyprint import CSS, HTML
    except ImportError as exc:  # pragma: no cover - declared dependencies
        raise PdfError(
            "PDF rendering needs `markdown` and `weasyprint`, which are declared "
            "dependencies. Run `pip install -e .` or rebuild the image."
        ) from exc

    body = markdown_lib.markdown(
        markdown_text,
        # tables for the leaderboard and the review; sane_lists so a list after a
        # paragraph is not silently swallowed into it.
        extensions=["tables", "sane_lists"],
    )
    generated = datetime.now(UTC).strftime("%d %B %Y, %H:%M UTC")
    document = (
        f"<html><head><meta charset='utf-8'><title>{_escape(title)}</title></head>"
        f"<body>{body}"
        f"<div class='footer'>Generated by AutoDS on {generated}. "
        "Every figure in this document is taken from the run's stored artifacts.</div>"
        "</body></html>"
    )

    try:
        rendered = HTML(string=document).write_pdf(stylesheets=[CSS(string=_STYLESHEET)])
    except Exception as exc:  # noqa: BLE001 - WeasyPrint raises broadly
        raise PdfError(f"The report could not be rendered as a PDF: {exc}") from exc

    if not rendered:  # pragma: no cover - defensive
        raise PdfError("PDF rendering produced no output.")

    logger.info("Rendered report PDF (%d bytes) for %r", len(rendered), title)
    return rendered


def _escape(text: str) -> str:
    """Minimal escaping for the one place user text lands in raw HTML."""
    return re.sub(r"[<>&]", lambda m: {"<": "&lt;", ">": "&gt;", "&": "&amp;"}[m.group()], text)
