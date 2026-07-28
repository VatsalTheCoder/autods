"""The pandas tool's sandbox (spec 7.13).

``eval`` on the output of a language model is a remote code execution primitive.
The tool exists because arithmetic cannot be answered from retrieved text, and it
is safe because the expression is checked against an allowlist grammar *before*
it runs -- not because the model is expected to behave.

These tests are mostly attacks. The point of a sandbox is what it refuses, and a
test suite that only checked the happy path would pass just as well against
``eval`` with no checks at all.
"""

from __future__ import annotations

import pandas as pd
import pytest

from app.ml.pandas_tool import UnsafeExpression, run_query, validate


@pytest.fixture
def frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "age": [20, 30, 40, 50],
            "city": ["A", "B", "A", "B"],
            "churn": [0, 1, 0, 1],
            "charge": [10.5, 20.5, 30.5, 40.5],
        }
    )


class TestItRefusesTheWaysOut:
    """Each of these is a route from 'evaluate an expression' to 'run anything'."""

    @pytest.mark.parametrize(
        "attack",
        [
            # The canonical sandbox escape: walk the type hierarchy to something
            # that can open a file or spawn a process.
            "df.__class__",
            "df.__class__.__bases__[0].__subclasses__()",
            "df.age.values.__array_interface__",
            # Direct reaches for the outside world.
            "__import__('os').system('id')",
            "open('/etc/passwd').read()",
            "exec('import os')",
            "eval('1+1')",
            "getattr(df, 'apply')",
            "globals()",
            "vars()",
            # Anything that takes a callable is a way to smuggle one in.
            "df.apply(lambda r: 1, axis=1)",
            "df.pipe(print)",
            "df.map(print)",
            "df.transform(print)",
            # pandas' own evaluators, which would undo the whole check.
            "df.query('age > 1')",
            "df.eval('age * 2')",
            # Writing anywhere.
            "df.to_csv('/tmp/out.csv')",
            "df.to_pickle('/tmp/out.pkl')",
            # Syntax that permits arbitrary execution.
            "[x for x in df.columns]",
            "(lambda: 1)()",
            "{k: v for k, v in df.items()}",
        ],
    )
    def test_an_attack_is_refused_without_running(self, frame, attack):
        with pytest.raises(UnsafeExpression):
            run_query(frame, attack)

    def test_a_dunder_is_refused_even_if_it_looks_harmless(self, frame):
        """Checked before the method allowlist -- it is the general escape."""
        with pytest.raises(UnsafeExpression, match="Private attributes"):
            run_query(frame, "df.__len__()")

    def test_an_unknown_name_is_refused(self, frame):
        with pytest.raises(UnsafeExpression, match="not available"):
            run_query(frame, "other_frame.mean()")

    def test_statements_are_not_expressions(self, frame):
        with pytest.raises(UnsafeExpression):
            run_query(frame, "x = df['age'].mean()")

    def test_an_absurdly_long_expression_is_refused(self, frame):
        with pytest.raises(UnsafeExpression, match="too long"):
            run_query(frame, "df['age'].mean() + " * 200 + "1")

    def test_an_empty_expression_is_refused(self, frame):
        with pytest.raises(UnsafeExpression, match="empty"):
            run_query(frame, "   ")


class TestItStillAnswersQuestions:
    """A sandbox that refuses everything is safe and useless."""

    @pytest.mark.parametrize(
        ("expression", "expected"),
        [
            ("df['age'].mean()", "35"),
            ("len(df)", "4"),
            ("df['city'].nunique()", "2"),
            ("df['churn'].sum()", "2"),
            ("df[df['age'] > 25]['age'].median()", "40"),
            ("round(df['charge'].mean(), 2)", "25.5"),
        ],
    )
    def test_ordinary_arithmetic_works(self, frame, expression, expected):
        assert run_query(frame, expression).value == expected

    def test_grouping_works(self, frame):
        result = run_query(frame, "df.groupby('city')['age'].mean()")
        assert "30" in result.value and "40" in result.value

    def test_value_counts_works(self, frame):
        assert "2" in run_query(frame, "df['churn'].value_counts()").value

    def test_the_expression_is_reported_back(self, frame):
        """The pandas path's failure mode is answering a different question."""
        result = run_query(frame, "df['age'].mean()")
        assert result.expression == "df['age'].mean()"


class TestHowResultsAreShown:
    def test_nan_is_explained_rather_than_printed(self, frame):
        """'nan' in an answer reads as a bug; saying what happened does not."""
        value = run_query(frame, "df[df['age'] > 1000]['age'].mean()").value
        assert "undefined" in value
        assert "nan" not in value.lower()

    def test_an_empty_selection_says_so(self, frame):
        assert "no rows matched" in run_query(frame, "df[df['age'] > 1000]").value

    def test_a_long_result_is_truncated(self):
        """A thousand-row answer was the wrong question, and will not fit a prompt."""
        big = pd.DataFrame({"n": range(200)})
        value = run_query(big, "df['n']").value
        assert "more" in value
        assert len(value.splitlines()) < 40

    def test_large_integers_are_readable(self):
        big = pd.DataFrame({"n": range(50_000)})
        assert run_query(big, "len(df)").value == "50,000"


class TestValidateOnItsOwn:
    def test_it_can_reject_without_a_frame(self):
        """So a caller can check an expression before having data to hand."""
        with pytest.raises(UnsafeExpression):
            validate("__import__('os')")

    def test_it_accepts_a_reasonable_query(self):
        assert validate("df['age'].mean()") is not None
