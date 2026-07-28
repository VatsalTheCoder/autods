"""Arithmetic over the cleaned dataset (spec 7.13, build-plan Section 10).

"What's the average age?" cannot be answered by searching text. No passage
contains that number, and a retrieval system asked for it will either find a
passage that mentions age and paraphrase something adjacent, or invent it. The
build plan is explicit: arithmetic goes to pandas instead.

**The model writes an expression; this module decides whether to run it.** That
split matters. Handing a language model the ability to execute code against a
DataFrame is the single most dangerous thing in this project -- ``eval`` on model
output is a remote code execution primitive, and "it only ever writes pandas" is
an assumption about a model, not a property of the system.

So the expression is checked against a grammar before it runs, and the check is
an allowlist:

- it must parse as a single Python expression;
- every node in its AST must be of a permitted type;
- every name in it must be ``df`` or a permitted method;
- attribute access is restricted to a fixed set of DataFrame and Series methods;
- dunder attributes are rejected outright, which closes the usual escape from a
  sandbox like this (``df.__class__.__bases__`` and onwards).

Anything else is refused without being executed. The result is that a compromised
or simply confused model can produce a wrong *number*, which is recoverable, but
cannot open a file, import a module, or reach the network.
"""

from __future__ import annotations

import ast
import logging
import math
from dataclasses import dataclass

import pandas as pd

logger = logging.getLogger(__name__)


class UnsafeExpression(RuntimeError):
    """The expression is not something this module is willing to evaluate."""


# The frame is bound to this name.
FRAME_NAME = "df"

# The only other names an expression may use. Each is a pure builtin that cannot
# reach the filesystem, the network or the import system, and each is bound
# explicitly at evaluation time rather than inherited from the real builtins --
# so this tuple is the whole vocabulary, not a filter over a larger one.
# ``len(df)`` is the reason this exists: "how many rows" is one of the commonest
# questions, and refusing it to keep the allowlist tidy would be the wrong call.
SAFE_BUILTINS = {
    "len": len,
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "sum": sum,
    "sorted": sorted,
}

# Methods an expression may call. Aggregations, selection and grouping -- the
# vocabulary of "what is the average X by Y", and nothing that writes, executes,
# or reaches outside the frame. Notably absent: ``apply``, ``map``, ``eval``,
# ``query``, ``pipe``, ``transform``, ``to_csv``, ``read_*`` -- every one of which
# either takes a callable or touches the world outside this process.
ALLOWED_METHODS = frozenset(
    {
        # Aggregation
        "mean",
        "median",
        "sum",
        "min",
        "max",
        "std",
        "var",
        "count",
        "nunique",
        "quantile",
        "mode",
        "corr",
        "cov",
        "sem",
        "skew",
        "kurt",
        "prod",
        # Shape and selection
        "head",
        "tail",
        "sort_values",
        "sort_index",
        "groupby",
        "value_counts",
        "unique",
        "describe",
        "abs",
        "round",
        "rank",
        "size",
        "idxmax",
        "idxmin",
        "nlargest",
        "nsmallest",
        "drop_duplicates",
        "dropna",
        "fillna",
        "notna",
        "isna",
        "isnull",
        "notnull",
        "between",
        "isin",
        "astype",
        "reset_index",
        # Presentation
        "to_dict",
        "to_list",
        "tolist",
        "item",
        "squeeze",
        "agg",
    }
)

# Attributes (not calls) an expression may read.
ALLOWED_ATTRIBUTES = frozenset(
    {"columns", "index", "values", "shape", "dtypes", "size", "str", "dt", "loc", "iloc"}
)

# AST nodes an expression may contain. Comprehensions, lambdas, calls to
# arbitrary names, imports, assignments and f-strings are all absent -- each is a
# route to executing something that is not arithmetic.
ALLOWED_NODES = (
    ast.Expression,
    ast.Call,
    ast.Attribute,
    ast.Name,
    ast.Load,
    ast.Constant,
    ast.Subscript,
    ast.Slice,
    ast.Tuple,
    ast.List,
    ast.Dict,
    ast.Set,
    ast.BinOp,
    ast.UnaryOp,
    ast.Compare,
    ast.BoolOp,
    ast.keyword,
    # Operators
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.FloorDiv,
    ast.Mod,
    ast.Pow,
    ast.USub,
    ast.UAdd,
    ast.Not,
    ast.Invert,
    ast.Eq,
    ast.NotEq,
    ast.Lt,
    ast.LtE,
    ast.Gt,
    ast.GtE,
    ast.In,
    ast.NotIn,
    ast.And,
    ast.Or,
    ast.BitAnd,
    ast.BitOr,
    ast.BitXor,
)

# How much of a result is worth returning. A question whose answer is a
# thousand-row frame was the wrong question for this tool, and the answer would
# not fit in a prompt anyway.
MAX_RESULT_ROWS = 25


@dataclass(frozen=True)
class QueryResult:
    """What running an expression produced."""

    expression: str
    value: str
    raw: object

    def __str__(self) -> str:  # pragma: no cover - convenience
        return self.value


def validate(expression: str) -> ast.Expression:
    """Parse and check an expression, or raise ``UnsafeExpression``.

    Separate from ``run_query`` so the check is testable on its own, and so a
    caller can reject an expression without having a DataFrame to hand.
    """
    expression = expression.strip()
    if not expression:
        raise UnsafeExpression("The expression is empty.")
    if len(expression) > 500:
        raise UnsafeExpression("The expression is too long to be a simple query.")

    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise UnsafeExpression(f"The expression does not parse: {exc.msg}") from exc

    for node in ast.walk(tree):
        if not isinstance(node, ALLOWED_NODES):
            raise UnsafeExpression(f"{type(node).__name__} is not allowed in a data query.")

        if isinstance(node, ast.Attribute):
            # Checked before the allowlist, because the dunder route is the one
            # that turns a restricted expression into arbitrary execution.
            if node.attr.startswith("_"):
                raise UnsafeExpression("Private attributes are not allowed.")
            if node.attr not in ALLOWED_METHODS and node.attr not in ALLOWED_ATTRIBUTES:
                raise UnsafeExpression(f"'{node.attr}' is not an allowed operation.")

        if isinstance(node, ast.Name) and node.id != FRAME_NAME and node.id not in SAFE_BUILTINS:
            allowed = ", ".join(sorted(SAFE_BUILTINS))
            raise UnsafeExpression(
                f"'{node.id}' is not available. Only '{FRAME_NAME}' and "
                f"these functions can be referenced: {allowed}."
            )

    return tree


def run_query(frame: pd.DataFrame, expression: str) -> QueryResult:
    """Validate an expression and evaluate it against the frame.

    ``__builtins__`` is emptied rather than left to default. Without that, an
    expression could reach ``open`` or ``__import__`` through the globals dict
    even with every AST check above passing -- the checks constrain the syntax,
    and this constrains what the syntax can resolve to.
    """
    tree = validate(expression)

    try:
        raw = eval(  # noqa: S307 - constrained by validate() and an empty builtins
            compile(tree, "<query>", "eval"),
            {"__builtins__": {}, **SAFE_BUILTINS},
            {FRAME_NAME: frame},
        )
    except Exception as exc:
        raise UnsafeExpression(f"The query could not be run: {exc}") from exc

    return QueryResult(expression=expression, value=_present(raw), raw=raw)


def _present(value: object) -> str:
    """Render a result compactly enough to put in a prompt and an answer."""
    if isinstance(value, pd.DataFrame):
        if value.empty:
            return "(no rows matched)"
        shown = value.head(MAX_RESULT_ROWS)
        suffix = (
            f"\n... and {len(value) - MAX_RESULT_ROWS} more rows"
            if len(value) > MAX_RESULT_ROWS
            else ""
        )
        return shown.to_string() + suffix

    if isinstance(value, pd.Series):
        if value.empty:
            return "(no rows matched)"
        shown = value.head(MAX_RESULT_ROWS)
        suffix = (
            f"\n... and {len(value) - MAX_RESULT_ROWS} more" if len(value) > MAX_RESULT_ROWS else ""
        )
        return shown.to_string() + suffix

    if isinstance(value, float):
        # NaN reaches here whenever a question asks for a statistic of an empty
        # selection. "nan" in an answer reads as a bug; saying so does not.
        if math.isnan(value):
            return "undefined (no values to compute it from)"
        return f"{value:,.4f}".rstrip("0").rstrip(".")

    if isinstance(value, int):
        return f"{value:,}"

    return str(value)
