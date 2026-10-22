"""SQL execution as a service: the sandbox shared by rollout tools, the reward
and evaluation.

    SAR_DB_ROOT=/path/to/omnisql/data uvicorn sar.sql_env.server:app --port 8090 --workers 4

Why a separate service rather than running SQL inside the trainer:
- isolation: a pathological query (huge sort, runaway CTE) burns a worker
  process here, not the Ray actor that holds the policy's GPU memory;
- scaling: rollouts issue thousands of queries per step; this scales with
  more uvicorn workers or replicas behind a load balancer, independent of
  the GPU count;
- one code path: training, eval and the deployed agent all execute SQL
  the same way, so a fix lands everywhere.

Endpoints: POST /v1/execute, POST /v1/score, GET /healthz, GET /metrics (Prometheus).
"""

from __future__ import annotations

import asyncio
import time

from fastapi import FastAPI, HTTPException, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel, Field

from sar.reward.sql_reward import score_sql
from sar.sql_env import dbs
from sar.sql_env.executor import execute

app = FastAPI(title="sar-sql-env")

EXEC_TOTAL = Counter("sar_sql_exec_total", "SQL executions", ["result"])
EXEC_SECONDS = Histogram("sar_sql_exec_seconds", "SQL execution time", buckets=(0.001, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 30))
# A thread per query, bounded: SQLite releases the GIL while executing.
_SEM = asyncio.Semaphore(64)


class ExecuteRequest(BaseModel):
    db_family: str
    db_id: str
    sql: str = Field(max_length=100_000)
    timeout_s: float = Field(10.0, gt=0, le=60)
    max_rows: int = Field(1000, gt=0, le=100_000)


class ScoreRequest(BaseModel):
    db_family: str
    db_id: str
    pred_sql: str = Field(max_length=100_000)
    gold_sql: str = Field(max_length=100_000)


def _path(family: str, db_id: str):
    try:
        return dbs.resolve(family, db_id)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e


@app.post("/v1/execute")
async def execute_endpoint(req: ExecuteRequest) -> dict:
    path = _path(req.db_family, req.db_id)
    t0 = time.perf_counter()
    async with _SEM:
        res = await asyncio.to_thread(execute, path, req.sql, timeout_s=req.timeout_s, max_rows=req.max_rows)
    EXEC_SECONDS.observe(time.perf_counter() - t0)
    EXEC_TOTAL.labels(result="ok" if res.ok else res.error_kind or "error").inc()
    return {"ok": res.ok, "rows": [list(r) for r in res.rows], "columns": list(res.columns), "error": res.error,
            "truncated": res.truncated, "elapsed_ms": res.elapsed_ms, "error_kind": res.error_kind}


@app.post("/v1/score")
async def score_endpoint(req: ScoreRequest) -> dict:
    path = _path(req.db_family, req.db_id)
    async with _SEM:
        return await asyncio.to_thread(score_sql, path, req.pred_sql, req.gold_sql)


@app.get("/healthz")
def healthz() -> dict:
    return {"ok": True, "db_root": str(dbs.db_root())}


@app.get("/metrics")
def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
