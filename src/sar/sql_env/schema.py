"""Schema text for a database the agent has never seen, in the format of its
training prompts (OmniSQL style): CREATE TABLE statements with column
types, primary and foreign keys, and a few example values per column as
comments. Serving uses this to point the agent at any SQLite database.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

MAX_EXAMPLE_CHARS = 40


def _q(name: str) -> str:
    return f'"{name}"'


def _examples(conn: sqlite3.Connection, table: str, col: str, k: int) -> list:
    try:
        rows = conn.execute(
            f"SELECT DISTINCT {_q(col)} FROM {_q(table)} WHERE {_q(col)} IS NOT NULL LIMIT {k}").fetchall()
    except sqlite3.Error:
        return []
    out = []
    for (v,) in rows:
        if isinstance(v, bytes):
            continue
        if isinstance(v, str) and len(v) > MAX_EXAMPLE_CHARS:
            v = v[:MAX_EXAMPLE_CHARS]
        out.append(v)
    return out


def describe(db_path: str | Path, *, examples: int = 2) -> str:
    conn = sqlite3.connect(f"file:{Path(db_path)}?mode=ro&immutable=1", uri=True)
    conn.text_factory = lambda b: b.decode("utf-8", errors="replace")
    try:
        tables = [r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%' ORDER BY rowid")]
        parts = []
        for t in tables:
            cols = conn.execute(f"PRAGMA table_info({_q(t)})").fetchall()  # cid, name, type, notnull, default, pk
            fks = conn.execute(f"PRAGMA foreign_key_list({_q(t)})").fetchall()  # id, seq, table, from, to, ...
            lines = []
            for _cid, name, typ, _nn, _d, _pk in cols:
                ex = _examples(conn, t, name, examples)
                col = f"    {_quote_ident(name)} {typ or 'TEXT'},"
                if ex:
                    col += f" -- example: {ex!r}"
                lines.append(col)
            pk = [c[1] for c in sorted(cols, key=lambda c: c[5]) if c[5]]
            if pk:
                lines.append(f"    PRIMARY KEY ({', '.join(_quote_ident(c) for c in pk)}),")
            for fk in fks:
                lines.append(f"    CONSTRAINT fk_{t}_{fk[3]} FOREIGN KEY ({_quote_ident(fk[3])}) "
                             f"REFERENCES {_quote_ident(fk[2])} ({_quote_ident(fk[4] or fk[3])}),")
            if lines:
                lines[-1] = lines[-1].rstrip(",") if "--" not in lines[-1] else lines[-1].replace(", --", " --", 1)
            parts.append(f"CREATE TABLE {_quote_ident(t)} (\n" + "\n".join(lines) + "\n);")
        return "\n\n".join(parts)
    finally:
        conn.close()


_KEYWORDS = {"order", "group", "select", "from", "where", "table", "index", "key", "value", "year", "role",
             "references", "limit", "date", "time", "user"}


def _quote_ident(name: str) -> str:
    """Backticks only where needed, as in the training schemas (`year`, `value`)."""
    if name.lower() in _KEYWORDS or not name.replace("_", "").isalnum() or name[0].isdigit():
        return "`" + name.replace("`", "``") + "`"
    return name
