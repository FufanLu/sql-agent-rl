"""The ``execute_sql`` tool the agent calls during rollouts.

verl's ToolAgentLoop drives the conversation: when the model emits a
<tool_call>, the loop parses it, calls ``SqlTool.execute`` and appends the
result as a tool message, and the model continues from there. Which database
a call runs against is per example: the dataset row carries it in
``extra_info.tools_kwargs.execute_sql.create_kwargs`` ({db_family, db_id}),
and verl hands that to ``create`` before each call.

Execution happens either in this process (a thread per call; SQLite releases
the GIL while it runs a query) or, with ``server_url`` set, in the
sql_env.server service, which keeps the databases and the CPU load off the
GPU node's training processes.
"""

from __future__ import annotations

import asyncio
import os
from typing import Any
from uuid import uuid4

import httpx
from verl.tools.base_tool import BaseTool
from verl.tools.schemas import OpenAIFunctionToolSchema, ToolResponse

from sar.sql_env import dbs
from sar.sql_env.executor import ExecResult, execute
from sar.sql_env.render import render


class SqlTool(BaseTool):
    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self.timeout_s = float(config.get("timeout_s", 10))
        self.max_rows = int(config.get("max_rows", 1000))
        self.display_rows = int(config.get("display_rows", 30))
        self.db_root = config.get("db_root") or os.environ.get("SAR_DB_ROOT")
        self.server_url = config.get("server_url") or os.environ.get("SAR_SQL_SERVER")
        self._client = httpx.AsyncClient(timeout=self.timeout_s + 10) if self.server_url else None
        self._instances: dict[str, dict[str, Any]] = {}

    async def create(self, instance_id: str | None = None, create_kwargs: dict | None = None, **kwargs):
        instance_id = instance_id or uuid4().hex
        self._instances[instance_id] = dict(create_kwargs or {})
        return instance_id, ToolResponse()

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs):
        target = self._instances.get(instance_id, {})
        sql = parameters.get("sql")
        if not isinstance(sql, str) or not sql.strip():
            return ToolResponse(text="Error (syntax): the 'sql' argument must be a non-empty string"), 0.0, {}
        if "db_id" not in target:
            return ToolResponse(text="Error: no database is attached to this task"), 0.0, {}
        res = await self._run(target["db_family"], target["db_id"], sql)
        metrics = {"exec_ok": float(res.ok), "exec_ms": res.elapsed_ms}
        return ToolResponse(text=render(res, max_rows=self.display_rows)), 0.0, metrics

    async def _run(self, family: str, db_id: str, sql: str) -> ExecResult:
        if self._client is not None:
            r = await self._client.post(
                f"{self.server_url}/v1/execute",
                json={"db_family": family, "db_id": db_id, "sql": sql,
                      "timeout_s": self.timeout_s, "max_rows": self.max_rows},
            )
            r.raise_for_status()
            body = r.json()
            body["rows"] = tuple(tuple(row) for row in body["rows"])
            body["columns"] = tuple(body["columns"])
            return ExecResult(**body)
        path = dbs.resolve(family, db_id, self.db_root)
        return await asyncio.to_thread(execute, path, sql, timeout_s=self.timeout_s, max_rows=self.max_rows)

    async def release(self, instance_id: str, **kwargs) -> None:
        self._instances.pop(instance_id, None)
