"""The trained SQL agent as an MCP server, so Claude, an IDE or any MCP client
can hand it a question about a SQLite database.

    SAR_LLM_BASE_URL=http://gateway:8080/v1 SAR_LLM_MODEL=sql-agent \
    SAR_DATABASES=/data/dbs python -m sar.serve.mcp_server

The model runs behind any OpenAI-compatible endpoint (vLLM directly, or the
vllm-sglang-infra gateway in front of it); this process runs the agent loop
and the SQL sandbox locally, next to the databases.

Tools:
  list_databases()                 SQLite files under $SAR_DATABASES
  describe_database(db)            the schema text the agent sees
  ask(db, question, knowledge="")  run the agent; returns the final SQL, its result and the steps taken
"""

from __future__ import annotations

import os
from pathlib import Path

from mcp.server.mcpserver import MCPServer  # mcp 2.x; FastMCP in 1.x
from openai import AsyncOpenAI

from sar.agent.runner import AgentConfig, run_agent
from sar.data.prompts import build_messages
from sar.sql_env.executor import execute
from sar.sql_env.render import render
from sar.sql_env.schema import describe

mcp = MCPServer("sql-agent")
ROOT = Path(os.environ.get("SAR_DATABASES", "."))


def _db(name: str) -> Path:
    path = (ROOT / name).resolve()
    if ROOT.resolve() not in path.parents or path.suffix not in (".sqlite", ".db") or not path.is_file():
        raise ValueError(f"unknown database {name!r}; call list_databases")
    return path


@mcp.tool()
def list_databases() -> list[str]:
    """SQLite databases the agent can query."""
    return sorted(str(p.relative_to(ROOT)) for p in ROOT.rglob("*") if p.suffix in (".sqlite", ".db"))


@mcp.tool()
def describe_database(db: str) -> str:
    """Schema of a database, as the agent sees it."""
    return describe(_db(db))


@mcp.tool()
async def ask(db: str, question: str, knowledge: str = "") -> dict:
    """Answer a question about a database with the trained text-to-SQL agent."""
    path = _db(db)
    client = AsyncOpenAI(base_url=os.environ["SAR_LLM_BASE_URL"], api_key=os.environ.get("SAR_LLM_API_KEY", "EMPTY"))
    cfg = AgentConfig(model=os.environ.get("SAR_LLM_MODEL", "sql-agent"))
    res = await run_agent(client, cfg, build_messages(describe(path), question, knowledge), path)
    out: dict = {"final_sql": res.final_sql, "stop_reason": res.stop_reason, "tool_calls": res.tool_calls,
                 "steps": [m for m in res.messages[2:]]}
    if res.final_sql:
        out["result"] = render(execute(path, res.final_sql, timeout_s=30, max_rows=200), max_rows=50)
    return out


if __name__ == "__main__":
    mcp.run()
