"""Runs the SQL agent against any OpenAI-compatible chat endpoint.

The same loop serves evaluation (vLLM on a GPU node), the deployed agent
(behind the vllm-sglang-infra gateway) and the MCP server, so what is
measured is what gets served. The endpoint must parse tool calls, e.g.
``vllm serve <model> --enable-auto-tool-choice --tool-call-parser hermes``.

The loop mirrors verl's ToolAgentLoop used in training: the model may call
``execute_sql`` up to ``max_tool_calls`` times; a turn without a tool call is
the final answer, expected inside <solution>...</solution>.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI

from sar.data.prompts import DEFAULT_MAX_TOOL_CALLS, TOOL_SCHEMA
from sar.reward.sql_reward import extract_final_sql
from sar.sql_env.executor import execute
from sar.sql_env.render import render


@dataclass
class AgentResult:
    messages: list[dict[str, Any]]
    final_sql: str | None
    stop_reason: str  # answered | format | max_tool_calls | length | error
    final_text: str = ""  # content of the last assistant turn
    tool_calls: int = 0
    completion_tokens: int = 0
    tool_errors: int = 0
    exec_ms: list[float] = field(default_factory=list)


@dataclass
class AgentConfig:
    model: str
    max_tool_calls: int = DEFAULT_MAX_TOOL_CALLS
    temperature: float = 0.0
    top_p: float = 1.0
    max_tokens_per_turn: int = 2048
    tool_timeout_s: float = 10.0
    tool_max_rows: int = 1000
    display_rows: int = 30
    max_tool_response_chars: int = 2048  # = verl multi_turn.max_tool_response_length


def _truncate_middle(text: str, limit: int) -> str:
    # Same as verl's default tool_response_truncate_side=middle.
    if len(text) <= limit:
        return text
    half = limit // 2
    return text[:half] + "...(truncated)..." + text[-half:]


async def run_agent(client: AsyncOpenAI, cfg: AgentConfig, messages: list[dict[str, Any]], db_path: str | Path,
                    *, seed: int | None = None) -> AgentResult:
    msgs = [dict(m) for m in messages]
    res = AgentResult(messages=msgs, final_sql=None, stop_reason="error")
    for _turn in range(cfg.max_tool_calls + 1):
        resp = await client.chat.completions.create(
            model=cfg.model, messages=msgs, tools=[TOOL_SCHEMA], temperature=cfg.temperature, top_p=cfg.top_p,
            max_tokens=cfg.max_tokens_per_turn, seed=seed,
        )
        choice = resp.choices[0]
        res.completion_tokens += resp.usage.completion_tokens if resp.usage else 0
        msg = choice.message
        calls = msg.tool_calls or []
        entry: dict[str, Any] = {"role": "assistant", "content": msg.content or ""}
        if calls:
            entry["tool_calls"] = [{"id": c.id, "type": "function",
                                    "function": {"name": c.function.name, "arguments": c.function.arguments}}
                                   for c in calls[:1]]  # one call per turn, as in training (max_parallel_calls=1)
        msgs.append(entry)
        res.final_text = msg.content or ""
        if choice.finish_reason == "length":
            res.stop_reason = "length"
            return res
        if not calls:
            res.final_sql, why = extract_final_sql(msg.content or "")
            res.stop_reason = "answered" if res.final_sql else "format"
            return res
        if res.tool_calls >= cfg.max_tool_calls:
            res.stop_reason = "max_tool_calls"
            return res
        call = calls[0]
        res.tool_calls += 1
        try:
            sql = json.loads(call.function.arguments).get("sql", "")
        except (json.JSONDecodeError, AttributeError):
            sql = ""
        if call.function.name != "execute_sql" or not isinstance(sql, str) or not sql.strip():
            text = "Error (syntax): call execute_sql with a non-empty 'sql' string"
            res.tool_errors += 1
        else:
            out = await asyncio.to_thread(execute, db_path, sql, timeout_s=cfg.tool_timeout_s,
                                          max_rows=cfg.tool_max_rows)
            res.exec_ms.append(out.elapsed_ms)
            res.tool_errors += 0 if out.ok else 1
            text = render(out, max_rows=cfg.display_rows)
        msgs.append({"role": "tool", "tool_call_id": call.id, "content": _truncate_middle(text, cfg.max_tool_response_chars)})
    res.stop_reason = "max_tool_calls"
    return res
