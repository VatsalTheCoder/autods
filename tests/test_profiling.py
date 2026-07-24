"""Tests for deterministic profiling.

Pure functions over DataFrames -- no LLM, no DB, no storage -- so these always
run and can assert exact values. This is where the build plan wants coverage
concentrated: the measurable half of schema detection, checked precisely.
"""

from __future__ import annotations

import pandas as pd

from app.services.profiling import profile_dataset, read_csv_frame


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "age": [34, 28, 45, 52, 19, 63],
            "city": ["London", "Leeds", "London", "Bristol", "Leeds", "London"],
            "email": [
                "a@x.com",
                "b@y.com",
                "c@z.com",
                "d@x.com",
                "e@y.com",
                "f@z.com",
            ],
            "churn": ["yes", "no", "no", "yes", "no", "no"],
        }
    )


class TestColumnTyping:
    def test_numeric_column(self):
        report = profile_dataset(_frame())
        age = report.column("age")
        assert age.semantic_type == "numeric"
        assert age.n_unique == 6
        assert age.null_count == 0

    def test_low_cardinality_string_is_categorical(self):
        report = profile_dataset(_frame())
        assert report.column("city").semantic_type == "categorical"

    def test_zero_one_integer_is_boolean(self):
        frame = pd.DataFrame({"flag": [0, 1, 1, 0], "x": [1.5, 2.5, 3.5, 4.5]})
        report = profile_dataset(frame)
        assert report.column("flag").semantic_type == "boolean"

    def test_high_cardinality_string_is_text(self):
        frame = pd.DataFrame(
            {
                "note": [f"a unique sentence number {i}" for i in range(50)],
                "y": list(range(50)),
            }
        )
        assert profile_dataset(frame).column("note").semantic_type == "text"

    def test_datetime_strings_are_detected(self):
        frame = pd.DataFrame(
            {
                "signup": ["2024-01-01", "2024-02-15", "2024-03-20", "2024-04-10"],
                "y": [1, 0, 1, 0],
            }
        )
        assert profile_dataset(frame).column("signup").semantic_type == "datetime"


class TestNullRates:
    def test_null_rate_is_a_fraction(self):
        frame = pd.DataFrame({"a": [1, None, None, 4], "b": [1, 2, 3, 4]})
        report = profile_dataset(frame)
        a = report.column("a")
        assert a.null_count == 2
        assert a.null_rate == 0.5


class TestPIIDetection:
    def test_email_column_is_flagged_and_excluded_by_default(self):
        report = profile_dataset(_frame())
        email = report.column("email")
        assert email.is_pii is True
        assert email.pii_type == "email"
        assert email.exclude is True  # safe default

    def test_ssn_pattern_is_detected(self):
        frame = pd.DataFrame({"ssn": ["123-45-6789", "987-65-4321", "111-22-3333"], "y": [1, 0, 1]})
        assert profile_dataset(frame).column("ssn").pii_type == "ssn"

    def test_plain_numbers_are_not_flagged_as_phone(self):
        """The loose phone pattern must not swallow an ordinary integer column."""
        frame = pd.DataFrame({"age": [34, 28, 45, 52], "y": [1, 0, 1, 0]})
        assert profile_dataset(frame).column("age").is_pii is False

    def test_non_pii_column_is_not_flagged(self):
        report = profile_dataset(_frame())
        assert report.column("city").is_pii is False
        assert report.column("city").exclude is False


class TestTargetAndTask:
    def test_last_non_pii_column_is_suggested(self):
        report = profile_dataset(_frame())
        assert report.suggested_target == "churn"

    def test_pii_columns_are_skipped_as_target(self):
        # If email were last, it must still not be chosen as the target.
        frame = pd.DataFrame(
            {"y": [1, 0, 1, 0], "email": ["a@x.com", "b@y.com", "c@z.com", "d@x.com"]}
        )
        assert profile_dataset(frame).suggested_target == "y"

    def test_categorical_target_is_classification(self):
        report = profile_dataset(_frame())
        assert report.task_type == "classification"

    def test_continuous_numeric_target_is_regression(self):
        frame = pd.DataFrame({"x": list(range(100)), "price": [i * 1.7 + 3 for i in range(100)]})
        assert profile_dataset(frame).task_type == "regression"


class TestClassBalance:
    def test_imbalance_is_measured_for_classification(self):
        # churn: 4 "no", 2 "yes" -> ratio 2.0, imbalanced.
        report = profile_dataset(_frame())
        balance = report.class_balance
        assert balance is not None
        assert balance.counts == {"no": 4, "yes": 2}
        assert balance.imbalance_ratio == 2.0
        assert balance.imbalanced is True

    def test_no_balance_for_regression(self):
        frame = pd.DataFrame({"x": list(range(50)), "price": [i * 2.0 for i in range(50)]})
        assert profile_dataset(frame).class_balance is None


def test_read_csv_frame_round_trips():
    data = b"a,b\n1,2\n3,4\n"
    frame = read_csv_frame(data)
    assert list(frame.columns) == ["a", "b"]
    assert frame.shape == (2, 2)
