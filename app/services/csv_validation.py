"""CSV validation and inspection.

Pure functions over bytes: no database, no storage, no FastAPI. That makes the
rules here directly testable, and keeps the upload route thin enough to read in
one screen.

The job is to reject a bad file *before* it reaches storage and before a row is
written, so a failed upload leaves nothing behind to clean up.
"""

from __future__ import annotations

import csv
import io
import logging
from dataclasses import dataclass
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

# Read at most this many rows for the preview. A 2 GB CSV should not be fully
# materialised just to show the user the first handful of rows.
PREVIEW_ROWS = 10

ALLOWED_EXTENSIONS = {".csv"}


class CSVValidationError(ValueError):
    """Raised when an uploaded file is not usable as a dataset."""


@dataclass(frozen=True)
class DatasetSummary:
    """What we learned about an uploaded CSV."""

    n_rows: int
    n_columns: int
    columns: list[str]
    preview: list[dict[str, Any]]

    def as_dict(self) -> dict[str, Any]:
        return {
            "n_rows": self.n_rows,
            "n_columns": self.n_columns,
            "columns": self.columns,
            "preview": self.preview,
        }


def validate_filename(filename: str | None) -> None:
    """Reject anything that is not a .csv."""
    if not filename:
        raise CSVValidationError("No filename was provided.")
    lowered = filename.lower()
    if not any(lowered.endswith(ext) for ext in ALLOWED_EXTENSIONS):
        allowed = ", ".join(sorted(ALLOWED_EXTENSIONS))
        raise CSVValidationError(f"Unsupported file type. Allowed: {allowed}")


def validate_size(size_bytes: int, max_mb: int) -> None:
    """Reject empty files and files above the configured cap."""
    if size_bytes == 0:
        raise CSVValidationError("The file is empty.")
    max_bytes = max_mb * 1024 * 1024
    if size_bytes > max_bytes:
        actual_mb = size_bytes / (1024 * 1024)
        raise CSVValidationError(f"File is {actual_mb:.1f} MB; the limit is {max_mb} MB.")


def inspect_csv(data: bytes) -> DatasetSummary:
    """Parse the CSV and describe it, or raise CSVValidationError.

    pandas raises a fairly wide range of exceptions on malformed input, and
    their messages are not written for end users. Everything is translated into
    a single error type carrying a message worth showing in the UI.
    """
    try:
        frame = pd.read_csv(io.BytesIO(data))
    except pd.errors.EmptyDataError as exc:
        raise CSVValidationError("The file contains no data.") from exc
    except pd.errors.ParserError as exc:
        raise CSVValidationError(
            "The file could not be parsed as CSV. Check the delimiter and quoting."
        ) from exc
    except UnicodeDecodeError as exc:
        raise CSVValidationError(
            "The file is not valid UTF-8 text. Re-save it as UTF-8 encoded CSV."
        ) from exc
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception("Unexpected error parsing CSV")
        raise CSVValidationError(f"Could not read the file: {exc}") from exc

    if frame.empty:
        raise CSVValidationError("The file has headers but no rows.")
    if frame.shape[1] < 2:
        raise CSVValidationError(
            "A dataset needs at least two columns: something to predict, "
            "and something to predict it from."
        )

    # Checked against the raw header, not the parsed frame: pandas silently
    # renames a repeated "age" to "age.1" while reading, so by the time we hold
    # a DataFrame the duplicates have already been papered over. Downstream
    # agents would then see a phantom column the user never wrote.
    duplicates = _duplicate_headers(data)
    if duplicates:
        raise CSVValidationError(f"Duplicate column names: {', '.join(duplicates)}")

    return DatasetSummary(
        n_rows=int(frame.shape[0]),
        n_columns=int(frame.shape[1]),
        columns=[str(column) for column in frame.columns],
        preview=_json_safe_preview(frame.head(PREVIEW_ROWS)),
    )


def _duplicate_headers(data: bytes) -> list[str]:
    """Return column names that appear more than once in the raw header row."""
    first_line = data.split(b"\n", 1)[0]
    try:
        header = next(csv.reader(io.StringIO(first_line.decode("utf-8"))))
    except (UnicodeDecodeError, StopIteration):
        return []

    seen: set[str] = set()
    duplicates: list[str] = []
    for name in header:
        if name in seen and name not in duplicates:
            duplicates.append(name)
        seen.add(name)
    return duplicates


def _json_safe_preview(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Convert preview rows into JSON-serialisable dicts.

    NaN is not valid JSON, and NumPy scalars are not serialisable by the
    standard encoder, so both are converted rather than left to blow up at
    response-encoding time.
    """
    return [
        {str(column): _json_safe_value(value) for column, value in row.items()}
        for row in frame.to_dict(orient="records")
    ]


def _json_safe_value(value: Any) -> Any:
    if pd.isna(value):
        return None
    if hasattr(value, "item"):  # NumPy scalar
        return value.item()
    return value
