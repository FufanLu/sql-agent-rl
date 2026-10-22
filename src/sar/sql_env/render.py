"""Turning an execution result into the text the model reads back."""

from __future__ import annotations

from .executor import ExecResult

MAX_CELL = 60


def _cell(v) -> str:
    s = "NULL" if v is None else str(v)
    s = s.replace("\n", " ").replace("|", "\\|")
    return s if len(s) <= MAX_CELL else s[: MAX_CELL - 3] + "..."


def render(res: ExecResult, *, max_rows: int = 50) -> str:
    """A Markdown table of at most ``max_rows`` rows, or the error message.

    The model only sees the first rows: enough to check that columns and
    values look right, while keeping every observation short so a multi-turn
    trajectory fits in the response budget.
    """
    if not res.ok:
        return f"Error ({res.error_kind}): {res.error}"
    if not res.columns:
        return "Query ran, no result set."
    lines = ["| " + " | ".join(_cell(c) for c in res.columns) + " |", "|" + "---|" * len(res.columns)]
    for row in res.rows[:max_rows]:
        lines.append("| " + " | ".join(_cell(v) for v in row) + " |")
    n = len(res.rows)
    if n == 0:
        lines.append("(0 rows)")
    elif n > max_rows or res.truncated:
        more = "+" if res.truncated else ""
        lines.append(f"({n}{more} rows, showing first {min(n, max_rows)})")
    else:
        lines.append(f"({n} rows)")
    return "\n".join(lines)
