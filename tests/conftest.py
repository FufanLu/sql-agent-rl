import sqlite3
from pathlib import Path

import pytest


def make_db(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE singer (singer_id INTEGER PRIMARY KEY, name TEXT, country TEXT, age INTEGER);
        INSERT INTO singer VALUES (1, 'Joe', 'NL', 52), (2, 'Tim', 'US', 32), (3, 'Ann', 'US', 41),
                                  (4, 'Bo', 'FR', 29);
        CREATE TABLE concert (concert_id INTEGER PRIMARY KEY, singer_id INTEGER, year INTEGER);
        INSERT INTO concert VALUES (1, 1, 2014), (2, 2, 2015), (3, 2, 2015), (4, 3, 2016);
        CREATE TABLE big (x INTEGER);
        WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM c WHERE x < 2000) INSERT INTO big SELECT x FROM c;
        """
    )
    conn.commit()
    conn.close()
    return path


@pytest.fixture()
def db_root(tmp_path: Path) -> Path:
    """A database root laid out like OmniSQL's: <family dir>/<db_id>/<db_id>.sqlite."""
    make_db(tmp_path / "spider" / "database" / "concert_singer" / "concert_singer.sqlite")
    return tmp_path


@pytest.fixture()
def db(db_root: Path) -> Path:
    return db_root / "spider" / "database" / "concert_singer" / "concert_singer.sqlite"
