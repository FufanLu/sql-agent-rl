import pytest
from fastapi.testclient import TestClient

from sar.sql_env.server import app


@pytest.fixture()
def client(db_root, monkeypatch):
    monkeypatch.setenv("SAR_DB_ROOT", str(db_root))
    return TestClient(app)


def test_execute(client):
    r = client.post("/v1/execute", json={"db_family": "spider", "db_id": "concert_singer",
                                         "sql": "SELECT count(*) FROM singer"})
    assert r.status_code == 200 and r.json()["rows"] == [[4]]


def test_execute_denied_write(client):
    r = client.post("/v1/execute", json={"db_family": "spider", "db_id": "concert_singer", "sql": "DROP TABLE singer"})
    assert r.json()["ok"] is False and r.json()["error_kind"] == "denied"


@pytest.mark.parametrize("db_id", ["../../etc/passwd", "a/b", ".."])
def test_path_traversal_rejected(client, db_id):
    r = client.post("/v1/execute", json={"db_family": "spider", "db_id": db_id, "sql": "SELECT 1"})
    assert r.status_code == 400


def test_score(client):
    r = client.post("/v1/score", json={"db_family": "spider", "db_id": "concert_singer",
                                       "pred_sql": "SELECT name FROM singer WHERE age > 40",
                                       "gold_sql": "SELECT name FROM singer WHERE age >= 41"})
    assert r.json()["acc"] == 1.0


def test_metrics(client):
    client.post("/v1/execute", json={"db_family": "spider", "db_id": "concert_singer", "sql": "SELECT 1"})
    assert "sar_sql_exec_total" in client.get("/metrics").text
