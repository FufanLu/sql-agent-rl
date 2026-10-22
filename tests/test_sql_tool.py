"""SqlTool against a minimal stand-in for verl's tool base classes, so the test
runs on a laptop without the GPU stack. On the cluster, test_verl_contract.py
checks the same tool against the real verl."""

import sys
import types

import pytest
from pydantic import BaseModel


class _ToolResponse(BaseModel):
    text: str | None = None
    image: list | None = None
    video: list | None = None


class _BaseTool:
    def __init__(self, config, tool_schema):
        self.config, self.tool_schema = config, tool_schema


@pytest.fixture()
def sql_tool_cls(monkeypatch):
    if "verl" not in sys.modules or not hasattr(sys.modules["verl"], "__version__"):
        base = types.ModuleType("verl.tools.base_tool")
        base.BaseTool = _BaseTool
        schemas = types.ModuleType("verl.tools.schemas")
        schemas.OpenAIFunctionToolSchema = dict
        schemas.ToolResponse = _ToolResponse
        for name, mod in {"verl": types.ModuleType("verl"), "verl.tools": types.ModuleType("verl.tools"),
                          "verl.tools.base_tool": base, "verl.tools.schemas": schemas}.items():
            monkeypatch.setitem(sys.modules, name, mod)
    monkeypatch.delitem(sys.modules, "sar.agent.sql_tool", raising=False)
    from sar.agent.sql_tool import SqlTool

    return SqlTool


async def test_tool_runs_sql_against_the_tasks_db(sql_tool_cls, db_root):
    tool = sql_tool_cls({"db_root": str(db_root), "display_rows": 2}, None)
    iid, _ = await tool.create(create_kwargs={"db_family": "spider", "db_id": "concert_singer"})
    resp, reward, metrics = await tool.execute(iid, {"sql": "SELECT name FROM singer ORDER BY singer_id"})
    assert resp.text.splitlines()[:4] == ["| name |", "|---|", "| Joe |", "| Tim |"]
    assert "(4 rows, showing first 2)" in resp.text
    assert reward == 0.0 and metrics["exec_ok"] == 1.0
    await tool.release(iid)
    assert iid not in tool._instances


async def test_tool_reports_errors_as_text(sql_tool_cls, db_root):
    tool = sql_tool_cls({"db_root": str(db_root)}, None)
    iid, _ = await tool.create(create_kwargs={"db_family": "spider", "db_id": "concert_singer"})
    resp, _, metrics = await tool.execute(iid, {"sql": "SELECT * FROM nope"})
    assert resp.text.startswith("Error (runtime): no such table") and metrics["exec_ok"] == 0.0
    resp, _, _ = await tool.execute(iid, {"sql": ""})
    assert resp.text.startswith("Error (syntax)")
