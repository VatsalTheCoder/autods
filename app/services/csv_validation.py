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

# The strings ``pd.read_csv`` turns into NaN unless told otherwise. Written out
# here rather than imported from pandas because the reader below is defined as a
# deliberate *subtraction* from this set: if pandas quietly added or removed a
# token, importing it would quietly change which values survive a load, which is
# exactly the class of silent behaviour this exists to prevent.
_PANDAS_DEFAULT_NA = frozenset(
    {
        "",
        "#N/A",
        "#N/A N/A",
        "#NA",
        "-1.#IND",
        "-1.#QNAN",
        "-NaN",
        "-nan",
        "1.#IND",
        "1.#QNAN",
        "<NA>",
        "N/A",
        "NA",
        "NULL",
        "NaN",
        "None",
        "n/a",
        "nan",
        "null",
    }
)

# The tokens above that a human plausibly *typed* to mean "this row has none of
# that thing" rather than "this value is unknown". They are kept as ordinary
# category labels; everything else in the default set stays missing.
#
# This distinction is not cosmetic. On the Ames house-prices dataset every one of
# ``PoolQC``, ``Alley``, ``Fence`` and ``MiscFeature`` is 80-100% the literal
# string "NA" -- meaning no pool, no alley, no fence -- and reading those as gaps
# made all four look almost entirely empty, so the mostly-empty rule in cleaning
# deleted four complete columns that had no missing data at all. The columns that
# survived were worse off: ``FireplaceQu``'s 690 "NA"s mean *no fireplace*, and
# imputing them with the modal category told the model that 690 fireplace-less
# houses had a fireplace of average quality.
#
# "NULL"/"null" are deliberately not here. Those come out of database exports and
# mean missing; nobody labels a category "null".
NOT_APPLICABLE = frozenset({"NA", "N/A", "n/a", "None"})

# What is still read as missing: pandas' defaults, less the tokens above. The
# empty cell survives in this set, so a genuinely blank field is still a gap.
_MISSING_TOKENS = sorted(_PANDAS_DEFAULT_NA - NOT_APPLICABLE)

# Fraction of a column's non-sentinel values that must parse as numbers for the
# sentinels in it to be read as gaps rather than labels. Not 1.0, so that one
# stray "unknown" in a numeric column does not flip the whole verdict.
_NUMERIC_RECOVERY_THRESHOLD = 0.95


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


def read_frame(data: bytes) -> pd.DataFrame:
    """Parse CSV bytes into a DataFrame, keeping "not applicable" labels intact.

    The one place in the codebase that turns bytes into a DataFrame, so that the
    upload preview, the schema profile and the modelling pipeline all see the
    same values. A column cannot be a category on one screen and a hole on the
    next.

    Differs from a bare ``pd.read_csv`` in one respect: the strings "NA", "N/A",
    "n/a" and "None" are read as *values*, not as missing data (see
    ``NOT_APPLICABLE``). Columns that are genuinely numeric are then converted
    back, so a numeric column carrying "NA" for unknown still ends up with real
    gaps -- see ``_recover_numeric``.
    """
    frame = pd.read_csv(io.BytesIO(data), keep_default_na=False, na_values=_MISSING_TOKENS)
    return _recover_numeric(frame)


def _recover_numeric(frame: pd.DataFrame) -> pd.DataFrame:
    """Re-read columns whose preserved sentinels turned out to mean "unknown".

    ``read_frame`` keeps "NA" as a label, which is right for a quality grade and
    wrong for a measurement: Ames stores both ``PoolQC`` ("NA" = no pool, a real
    category) and ``LotFrontage`` ("NA" = frontage not recorded, a real gap) the
    same way, and only the column's other values say which is which.

    The rule is the dtype the rest of the column implies. If everything that is
    not a sentinel parses as a number, the column is a measurement and its
    sentinels are gaps. If not, the column is labels and the sentinel is one of
    them -- which is also the safe reading when "NA" really did mean unknown,
    since an explicit "unknown" level carries more information for a tree than a
    modal category imputed over it.

    Only columns holding a preserved token are touched; pandas' own dtype
    verdict stands everywhere else.
    """
    for name in frame.columns:
        series = frame[name]
        if not _is_text(series):
            continue

        sentinel = series.isin(NOT_APPLICABLE)
        if not sentinel.any():
            continue

        others = series.where(~sentinel)
        present = int(others.notna().sum())
        if not present:
            # Nothing but sentinels. There is no other evidence about the
            # column, and cleaning drops it as constant either way.
            continue

        numbers = pd.to_numeric(others, errors="coerce")
        if int(numbers.notna().sum()) / present >= _NUMERIC_RECOVERY_THRESHOLD:
            frame[name] = numbers

    return frame


def _is_text(series: pd.Series) -> bool:
    """True for a column of strings, under either pandas' old or new dtype.

    pandas 2 parsed a text column as ``object``; pandas 3 gives it a dedicated
    ``str`` dtype, and ``is_object_dtype`` is False for it. Checking only the
    former silently skips every string column on pandas 3 -- which is to say it
    turns ``_recover_numeric`` into a no-op. Mirrors ``cleaning._is_object_like``.
    """
    return pd.api.types.is_object_dtype(series) or isinstance(series.dtype, pd.StringDtype)


def inspect_csv(data: bytes) -> DatasetSummary:
    """Parse the CSV and describe it, or raise CSVValidationError.

    pandas raises a fairly wide range of exceptions on malformed input, and
    their messages are not written for end users. Everything is translated into
    a single error type carrying a message worth showing in the UI.
    """
    try:
        frame = read_frame(data)
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
