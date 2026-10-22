"""Deciding whether a predicted query is correct: execution match (EX).

Two SQL strings can look nothing alike and still be equally right, so
text-to-SQL is graded by running both queries and comparing the rows.

Two modes:

- ``set`` (default): the rows as a set, ignoring order and duplicates. This
  is what SkyRL-SQL and OmniSQL use (`frozenset(rows)`), so our numbers are
  comparable to theirs, and it is the training reward.
- ``strict``: duplicates count, and order counts when the gold query sorts
  at the top level (ORDER BY). Reported alongside, because ``set`` accepts
  e.g. a missing DISTINCT or a wrong sort direction.

Column order is compared as returned in both modes: the question asks for
specific columns, and the prompt tells the model which ones.
"""

from __future__ import annotations

from collections import Counter
from typing import Literal

import sqlglot

Mode = Literal["set", "strict"]


def has_top_level_order_by(sql: str) -> bool:
    """True if the outermost query sorts its result."""
    try:
        tree = sqlglot.parse_one(sql, read="sqlite")
    except (sqlglot.errors.ParseError, sqlglot.errors.TokenError, ValueError):
        # Unparseable gold SQL is rare; a plain substring check is a safe
        # fallback because it only decides whether order is compared.
        return "order by" in sql.lower()
    return tree is not None and tree.args.get("order") is not None


def rows_match(pred: tuple[tuple, ...], gold: tuple[tuple, ...], *, mode: Mode = "set", ordered: bool = False) -> bool:
    if mode == "set":
        return frozenset(pred) == frozenset(gold)
    if ordered:
        return list(pred) == list(gold)
    return Counter(pred) == Counter(gold)
