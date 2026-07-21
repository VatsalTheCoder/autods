"""Tests for CSV validation.

Pure functions over bytes, so these need no database, storage or network. They
pin down exactly which files are rejected and why -- the rules a user hits
first, and the ones most likely to be quietly loosened later.
"""

from __future__ import annotations

import pytest

from app.services.csv_validation import (
    CSVValidationError,
    inspect_csv,
    validate_filename,
    validate_size,
)

VALID_CSV = b"age,city,income\n34,London,52000\n28,Leeds,41000\n45,Bristol,68000\n"


class TestFilename:
    def test_accepts_csv(self):
        validate_filename("data.csv")
        validate_filename("DATA.CSV")  # case-insensitive

    @pytest.mark.parametrize("name", ["data.xlsx", "data.json", "data", "data.csv.exe"])
    def test_rejects_non_csv(self, name):
        with pytest.raises(CSVValidationError):
            validate_filename(name)

    def test_rejects_missing_filename(self):
        with pytest.raises(CSVValidationError, match="No filename"):
            validate_filename(None)


class TestSize:
    def test_accepts_a_file_within_the_limit(self):
        validate_size(1024, max_mb=200)

    def test_rejects_empty_file(self):
        with pytest.raises(CSVValidationError, match="empty"):
            validate_size(0, max_mb=200)

    def test_rejects_oversized_file(self):
        with pytest.raises(CSVValidationError, match="limit is 1 MB"):
            validate_size(2 * 1024 * 1024, max_mb=1)

    def test_boundary_is_inclusive(self):
        """Exactly at the limit is allowed; one byte over is not."""
        validate_size(1024 * 1024, max_mb=1)
        with pytest.raises(CSVValidationError):
            validate_size(1024 * 1024 + 1, max_mb=1)


class TestInspect:
    def test_reports_shape_and_columns(self):
        summary = inspect_csv(VALID_CSV)

        assert summary.n_rows == 3
        assert summary.n_columns == 3
        assert summary.columns == ["age", "city", "income"]

    def test_preview_is_capped(self):
        rows = b"\n".join(b"%d,x,%d" % (i, i) for i in range(500))
        data = b"a,b,c\n" + rows + b"\n"

        summary = inspect_csv(data)

        assert summary.n_rows == 500
        assert len(summary.preview) == 10

    def test_preview_rows_are_json_safe(self):
        """NaN is not valid JSON and would fail at response encoding."""
        summary = inspect_csv(b"a,b\n1,\n2,5\n")

        assert summary.preview[0]["b"] is None
        assert summary.preview[1]["b"] == 5

    def test_rejects_empty_bytes(self):
        with pytest.raises(CSVValidationError):
            inspect_csv(b"")

    def test_rejects_headers_with_no_rows(self):
        with pytest.raises(CSVValidationError, match="no rows"):
            inspect_csv(b"age,city,income\n")

    def test_rejects_single_column(self):
        """One column cannot be a supervised learning problem."""
        with pytest.raises(CSVValidationError, match="at least two columns"):
            inspect_csv(b"age\n34\n28\n")

    def test_rejects_duplicate_column_names(self):
        """pandas silently renames these to age/age.1, which would confuse
        every downstream agent."""
        with pytest.raises(CSVValidationError, match="Duplicate column"):
            inspect_csv(b"age,age\n1,2\n3,4\n")

    def test_rejects_non_utf8(self):
        with pytest.raises(CSVValidationError, match="UTF-8"):
            inspect_csv(b"a,b\n\xff\xfe,2\n")

    def test_accepts_missing_values(self):
        """Blanks are the cleaning agent's problem, not a reason to reject."""
        summary = inspect_csv(b"a,b,c\n1,,3\n,5,6\n")
        assert summary.n_rows == 2
