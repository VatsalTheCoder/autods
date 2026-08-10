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
    read_frame,
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


class TestNotApplicableLabels:
    """A sentinel like "NA" in a label column is a value, not a gap.

    The case that motivated this: on Ames house prices, ``PoolQC`` is 99.5% the
    literal string "NA" meaning *no pool*. Read as missing, the column looked
    almost entirely empty and cleaning deleted it -- along with ``Alley``,
    ``Fence`` and ``MiscFeature``, none of which had a single missing value.
    """

    def test_keeps_na_as_a_category_in_a_label_column(self):
        frame = read_frame(b"quality\nGd\nEx\nNA\nNA\nTA\n")
        assert frame["quality"].isna().sum() == 0
        assert (frame["quality"] == "NA").sum() == 2
        assert frame["quality"].nunique() == 4

    def test_keeps_none_as_a_category(self):
        """Ames spells MasVnrType's "no veneer" as the string "None"."""
        frame = read_frame(b"veneer\nBrkFace\nNone\nNone\nStone\n")
        assert frame["veneer"].isna().sum() == 0
        assert (frame["veneer"] == "None").sum() == 2

    def test_reads_na_as_a_gap_in_a_numeric_column(self):
        """The other half: "NA" among numbers is genuinely unknown."""
        frame = read_frame(b"frontage\n65\n80\nNA\n70\n")
        assert frame["frontage"].dtype.kind == "f"
        assert frame["frontage"].isna().sum() == 1
        assert frame["frontage"].sum() == 215

    def test_a_blank_cell_is_still_missing(self):
        """Narrowing the token list must not stop an empty field being a gap."""
        frame = read_frame(b"a,b\n1,x\n,y\n")
        assert frame["a"].isna().sum() == 1

    @pytest.mark.parametrize("token", [b"NaN", b"null", b"NULL", b"#N/A", b"<NA>"])
    def test_machine_written_tokens_are_still_missing(self, token):
        """Nobody labels a category "NaN"; those come from an exporter."""
        frame = read_frame(b"label\nx\n" + token + b"\ny\n")
        assert frame["label"].isna().sum() == 1

    def test_one_stray_word_does_not_make_a_numeric_column_categorical(self):
        """Below the recovery threshold the column is text; above it, numbers."""
        rows = b"\n".join(str(i).encode() for i in range(40))
        frame = read_frame(b"n\n" + rows + b"\nNA\nunknown\n")
        assert frame["n"].dtype.kind == "f"
        assert frame["n"].isna().sum() == 2

    def test_a_column_of_only_sentinels_is_left_alone(self):
        """No other values means no evidence; cleaning drops it as constant."""
        frame = read_frame(b"a,b\n1,NA\n2,NA\n")
        assert frame["b"].isna().sum() == 0

    def test_the_preview_shows_the_same_values_the_model_will_see(self):
        summary = inspect_csv(b"a,quality\n1,Gd\n2,NA\n")
        assert summary.preview[1]["quality"] == "NA"
