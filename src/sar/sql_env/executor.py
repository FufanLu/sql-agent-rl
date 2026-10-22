"""Sandboxed, read-only SQL execution against SQLite databases.

Every SQL string a model writes runs through `execute`: as a tool call during
rollouts (the model looks at the result and revises), and as the reward check
at the end (does the final query return the same rows as the gold query).
Model output is untrusted, so three independent guards apply:

1. The file is opened read-only (`mode=ro`) and `immutable=1`, which also
   skips SQLite's file locking: thousands of concurrent rollouts read the same
   files, and NFS locks would serialize them.
2. An authorizer allows only reads. ATTACH (opening other files), PRAGMA,
   writes and schema changes are rejected before they run.
3. A progress handler aborts a query once it passes its deadline, and at most
   `max_rows + 1` rows are fetched, so a cross join of two big tables can
   neither hang a worker nor exhaust its memory.
"""

from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path

# Authorizer action codes that a read-only query needs. Everything else
# (INSERT, UPDATE, DELETE, CREATE_*, DROP_*, ATTACH, PRAGMA, TRANSACTION, ...)
# is denied.
_ALLOWED_ACTIONS = {
    sqlite3.SQLITE_SELECT,
    sqlite3.SQLITE_READ,
    sqlite3.SQLITE_FUNCTION,
    getattr(sqlite3, "SQLITE_RECURSIVE", 33),  # WITH RECURSIVE
}

# The progress handler runs every this many SQLite VM instructions; small
# enough that a deadline is noticed within a millisecond or so.
_PROGRESS_STEPS = 1000


@dataclass(frozen=True)
class ExecResult:
    ok: bool
    rows: tuple[tuple, ...] = ()
    columns: tuple[str, ...] = ()
    error: str | None = None
    truncated: bool = False  # more than max_rows rows existed
    elapsed_ms: float = 0.0
    error_kind: str | None = field(default=None)  # timeout | denied | syntax | runtime | missing_db


def _authorizer(action: int, *_args) -> int:
    return sqlite3.SQLITE_OK if action in _ALLOWED_ACTIONS else sqlite3.SQLITE_DENY


def _classify(msg: str) -> str:
    m = msg.lower()
    if "interrupted" in m:
        return "timeout"
    if "not authorized" in m or "prohibited" in m or "readonly" in m:
        return "denied"
    if "syntax error" in m or "incomplete input" in m or "one statement at a time" in m:
        return "syntax"
    return "runtime"


def execute(db_path: str | Path, sql: str, *, timeout_s: float = 10.0, max_rows: int = 10_000) -> ExecResult:
    """Runs one SQL statement read-only and returns its rows or the error."""
    t0 = time.perf_counter()

    def done(**kw) -> ExecResult:
        return ExecResult(elapsed_ms=(time.perf_counter() - t0) * 1000, **kw)

    path = Path(db_path)
    if not path.is_file():
        return done(ok=False, error=f"database not found: {path.name}", error_kind="missing_db")
    sql = sql.strip()
    if not sql:
        return done(ok=False, error="empty SQL", error_kind="syntax")

    deadline = t0 + timeout_s
    conn = sqlite3.connect(f"file:{path}?mode=ro&immutable=1", uri=True, check_same_thread=False)
    try:
        conn.set_authorizer(_authorizer)
        conn.set_progress_handler(lambda: 1 if time.perf_counter() > deadline else 0, _PROGRESS_STEPS)
        # Bytes in TEXT columns (a few BIRD databases have invalid UTF-8)
        # would otherwise raise while fetching.
        conn.text_factory = lambda b: b.decode("utf-8", errors="replace")
        cur = conn.execute(sql)
        rows = cur.fetchmany(max_rows + 1)
        columns = tuple(d[0] for d in cur.description or ())
        truncated = len(rows) > max_rows
        return done(ok=True, rows=tuple(rows[:max_rows]), columns=columns, truncated=truncated)
    except sqlite3.Warning as e:  # e.g. more than one statement
        return done(ok=False, error=str(e), error_kind=_classify(str(e)))
    except sqlite3.Error as e:
        return done(ok=False, error=str(e), error_kind=_classify(str(e)))
    except (OverflowError, ValueError, MemoryError) as e:
        return done(ok=False, error=f"{type(e).__name__}: {e}", error_kind="runtime")
    finally:
        conn.close()
