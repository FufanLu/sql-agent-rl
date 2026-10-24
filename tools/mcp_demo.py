"""The deployment's last hop: an MCP client starts the SQL agent's MCP server
(src/sar/serve/mcp_server.py) over stdio, as Claude Code or an IDE would, and
asks it eval questions. The server runs the agent loop against the model
endpoint in $SAR_LLM_BASE_URL (the gateway) and executes the final SQL; this
script scores each answer against the gold SQL.

    python -P tools/mcp_demo.py <eval parquet> <db root> <out json> [--n 5]
Needs SAR_LLM_BASE_URL, SAR_LLM_MODEL and SAR_LLM_API_KEY for the server.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time

import pandas as pd
from mcp.client import Client
from mcp.client.stdio import StdioServerParameters

from sar.reward.sql_reward import score_sql
from sar.sql_env import dbs

# The stdio transport starts the server with a minimal environment; these are
# what it needs on a compute node (NO_PROXY: or the OpenAI client goes through
# the site proxy, docs/failures.md #12).
PASS_ENV = ["PYTHONPATH", "NO_PROXY", "no_proxy", "SAR_LLM_BASE_URL", "SAR_LLM_MODEL", "SAR_LLM_API_KEY"]


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("db_root")
    ap.add_argument("out")
    ap.add_argument("--n", type=int, default=5)
    args = ap.parse_args()

    rows = pd.read_parquet(args.data).head(args.n).to_dict("records")
    family = rows[0]["extra_info"]["db_family"]
    root = dbs.resolve(family, rows[0]["extra_info"]["db_id"], args.db_root).parent.parent
    env = {k: os.environ[k] for k in PASS_ENV if k in os.environ} | {"SAR_DATABASES": str(root)}
    server = StdioServerParameters(command=sys.executable, args=["-P", "-m", "sar.serve.mcp_server"], env=env)

    report: dict = {"server": "sar.serve.mcp_server over stdio", "endpoint": os.environ.get("SAR_LLM_BASE_URL"),
                    "questions": []}
    async with Client(server, read_timeout_seconds=300) as client:
        report["tools"] = [t.name for t in (await client.list_tools()).tools]
        for row in rows:
            info = row["extra_info"]
            db = f"{info['db_id']}/{info['db_id']}.sqlite"
            t0 = time.time()
            res = await client.call_tool("ask", {"db": db, "question": info["question"]})
            out = json.loads(res.content[0].text) if not res.is_error else {"error": res.content[0].text}
            score = (score_sql(root / db, out["final_sql"], row["reward_model"]["ground_truth"])
                     if out.get("final_sql") else {"acc": 0.0})
            report["questions"].append({
                "db": db, "question": info["question"], "gold": row["reward_model"]["ground_truth"],
                "final_sql": out.get("final_sql"), "tool_calls": out.get("tool_calls"),
                "stop_reason": out.get("stop_reason"), "correct": score["acc"] == 1.0,
                "seconds": round(time.time() - t0, 2), "result": out.get("result"), "steps": out.get("steps"),
                "error": out.get("error"),
            })
            q = report["questions"][-1]
            print(f"{q['correct']!s:5} {q['tool_calls']} calls {q['seconds']:5.1f}s  {info['question'][:70]}", flush=True)
    report["correct"] = sum(q["correct"] for q in report["questions"])
    with open(args.out, "w") as f:
        json.dump(report, f, indent=1, ensure_ascii=False)
    print(f"tools {report['tools']}; correct {report['correct']}/{len(rows)} -> {args.out}")


if __name__ == "__main__":
    asyncio.run(main())
