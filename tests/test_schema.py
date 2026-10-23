import sqlite3

from sar.sql_env.schema import describe


def test_describe_matches_training_style(tmp_path):
    db = tmp_path / "x.sqlite"
    conn = sqlite3.connect(db)
    conn.executescript("""
        CREATE TABLE countries (country_id INTEGER PRIMARY KEY, english_name TEXT, population INTEGER);
        CREATE TABLE country_metrics (metric_id INTEGER PRIMARY KEY, country_id INTEGER, `year` INTEGER, `value` REAL,
            FOREIGN KEY (country_id) REFERENCES countries (country_id));
        INSERT INTO countries VALUES (0, 'Afghanistan', 38928346), (1, 'Åland Islands', 29457), (2, 'X', 1);
        INSERT INTO country_metrics VALUES (0, 0, 2022, 1.5);
    """)
    conn.close()
    text = describe(db)
    assert "CREATE TABLE countries (" in text
    assert "english_name TEXT, -- example: ['Afghanistan', 'Åland Islands']" in text
    assert "`year` INTEGER" in text and "`value` REAL" in text
    assert "PRIMARY KEY (country_id)" in text
    assert "FOREIGN KEY (country_id) REFERENCES countries (country_id)" in text
    # the last line of each table has no trailing comma
    for block in text.split("\n\n"):
        assert not block.splitlines()[-2].endswith(",")
