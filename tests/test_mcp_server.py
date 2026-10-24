import json
from pathlib import Path

import pytest

pytest.importorskip("mcp")
from mcp.client import Client  # noqa: E402
from test_runner import ScriptedClient  # noqa: E402

from sar.serve import mcp_server  # noqa: E402


@pytest.fixture()
def served(db_root: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(mcp_server, "ROOT", db_root)
    return db_root


async def test_tools_listed(served):
    async with Client(mcp_server.mcp) as client:
        names = {t.name for t in (await client.list_tools()).tools}
    assert names == {"list_databases", "describe_database", "ask"}


async def test_list_and_describe(served):
    assert mcp_server.list_databases() == ["spider/database/concert_singer/concert_singer.sqlite"]
    assert "singer" in mcp_server.describe_database("spider/database/concert_singer/concert_singer.sqlite")


@pytest.mark.parametrize("name", ["../x.sqlite", "/etc/passwd", "spider/database/concert_singer", "missing.sqlite"])
def test_rejects_paths_outside_root(served, name):
    with pytest.raises(ValueError):
        mcp_server.describe_database(name)


async def test_ask_through_mcp(served, monkeypatch):
    """A full MCP round trip: the client calls `ask`, the server runs the agent
    loop against a scripted model and executes the final SQL."""
    scripted = ScriptedClient([
        ("Check the values.", "SELECT DISTINCT country FROM singer", "tool_calls"),
        ("<solution>SELECT name FROM singer WHERE country = 'US'</solution>", None, "stop"),
    ])
    monkeypatch.setattr(mcp_server, "AsyncOpenAI", lambda **_: scripted)
    monkeypatch.setenv("SAR_LLM_BASE_URL", "http://unused/v1")
    async with Client(mcp_server.mcp) as client:
        res = await client.call_tool("ask", {"db": "spider/database/concert_singer/concert_singer.sqlite",
                                             "question": "Names of US singers?"})
    assert not res.is_error
    out = json.loads(res.content[0].text)  # a plain dict return comes back as JSON text
    assert out["final_sql"] == "SELECT name FROM singer WHERE country = 'US'"
    assert out["tool_calls"] == 1 and out["stop_reason"] == "answered"
    assert "Tim" in out["result"] and "Ann" in out["result"]
