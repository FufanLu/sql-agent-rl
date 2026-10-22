import sqlite3

import pytest

from sar.sql_env.compare import has_top_level_order_by, rows_match
from sar.sql_env.executor import execute
from sar.sql_env.render import render


def test_select(db):
    r = execute(db, "SELECT name FROM singer WHERE country = 'US' ORDER BY name")
    assert r.ok and r.rows == (("Ann",), ("Tim",)) and r.columns == ("name",)


@pytest.mark.parametrize(
    "sql",
    [
        "DELETE FROM singer",
        "UPDATE singer SET age = 0",
        "INSERT INTO singer VALUES (9, 'x', 'y', 1)",
        "DROP TABLE singer",
        "CREATE TABLE t (a)",
        "ATTACH DATABASE '/tmp/x.db' AS x",
        "PRAGMA writable_schema = 1",
        "BEGIN",
    ],
)
def test_writes_and_escapes_are_denied(db, sql):
    r = execute(db, sql)
    assert not r.ok
    # The database is untouched.
    assert sqlite3.connect(db).execute("SELECT count(*) FROM singer").fetchone() == (4,)


def test_multiple_statements_rejected(db):
    r = execute(db, "SELECT 1; DELETE FROM singer")
    assert not r.ok and r.error_kind == "syntax"


def test_timeout(db):
    # A 2000^3 cross join takes minutes; the deadline stops it.
    r = execute(db, "SELECT count(*) FROM big a, big b, big c", timeout_s=0.3)
    assert not r.ok and r.error_kind == "timeout"
    assert r.elapsed_ms < 2000


def test_row_limit(db):
    r = execute(db, "SELECT x FROM big", max_rows=100)
    assert r.ok and len(r.rows) == 100 and r.truncated


def test_errors_are_classified(db, tmp_path):
    assert execute(db, "SELEC name FROM singer").error_kind == "syntax"
    assert execute(db, "SELECT nope FROM singer").error_kind == "runtime"
    assert execute(tmp_path / "missing.sqlite", "SELECT 1").error_kind == "missing_db"
    assert execute(db, "   ").error_kind == "syntax"


def test_recursive_cte_allowed(db):
    r = execute(db, "WITH RECURSIVE c(n) AS (SELECT 1 UNION ALL SELECT n+1 FROM c WHERE n<3) SELECT n FROM c")
    assert r.ok and r.rows == ((1,), (2,), (3,))


def test_rows_match_modes():
    a, b = ((1,), (1,), (2,)), ((2,), (1,))
    assert rows_match(a, b, mode="set")
    assert not rows_match(a, b, mode="strict")
    assert rows_match(((2,), (1,)), ((1,), (2,)), mode="strict")
    assert not rows_match(((2,), (1,)), ((1,), (2,)), mode="strict", ordered=True)


def test_order_by_detection():
    assert has_top_level_order_by("SELECT a FROM t ORDER BY a")
    assert not has_top_level_order_by("SELECT a FROM (SELECT a FROM t ORDER BY a LIMIT 3)")
    assert has_top_level_order_by("WITH x AS (SELECT 1 AS a) SELECT a FROM x ORDER BY a DESC")


def test_render(db):
    out = render(execute(db, "SELECT x FROM big"), max_rows=3)
    assert out.splitlines()[0] == "| x |" and "(2000 rows, showing first 3)" in out
    assert render(execute(db, "SELECT 1 WHERE 0")).endswith("(0 rows)")
    assert render(execute(db, "SELEC")).startswith("Error (syntax)")
