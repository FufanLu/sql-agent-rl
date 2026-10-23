"""The reward: did the agent's final SQL return the right rows?

verl calls ``compute_score`` once per finished trajectory with the decoded
response, which for a multi-turn rollout is every assistant turn plus every
tool result in between. The score follows SkyRL-SQL:

    -1  no usable final answer (no single <solution>...</solution> in the last turn,
        or the trajectory ended mid tool call: out of turns or tokens)
     0  a final answer that fails to run or returns different rows
     1  same rows as the gold query (set comparison, as SkyRL-SQL / OmniSQL)

The reward is deliberately sparse and binary. Partial credit (e.g. for
running at all, or for matching some columns) is easy for the policy to
farm without getting more answers right; outcome-only rewards are what the
Arctic-Text2SQL-R1 and SkyRL-SQL results were obtained with.

The returned dict also carries diagnostics (strict match, whether the SQL
ran, the error kind), which verl logs per step next to the score.
"""

from __future__ import annotations

import logging
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

from sar.sql_env import dbs
from sar.sql_env.compare import has_top_level_order_by, rows_match
from sar.sql_env.executor import ExecResult, execute

logger = logging.getLogger(__name__)

SOLUTION_RE = re.compile(r"<solution>(.*?)</solution>", re.S)
FENCE_RE = re.compile(r"^```(?:sql|sqlite)?\s*|\s*```$", re.I)
# How a tool result appears in the decoded text (Qwen2.5 chat template).
TOOL_RESPONSE_END = "</tool_response>"
TOOL_CALL_START = "<tool_call>"

REWARD_TIMEOUT_S = 30.0  # as SkyRL-SQL; BIRD gold queries can take seconds
REWARD_MAX_ROWS = 10_000


def extract_final_sql(response: str) -> tuple[str | None, str]:
    """The SQL in the final answer, and why there is none if there is none."""
    final_turn = response.rsplit(TOOL_RESPONSE_END, 1)[-1]
    found = SOLUTION_RE.findall(final_turn)
    if TOOL_CALL_START in final_turn and not found:
        return None, "unfinished"  # stopped while still calling tools
    if len(found) != 1:
        return None, "format"
    sql = FENCE_RE.sub("", found[0].strip()).strip()
    if not sql:
        return None, "format"
    return sql, ""


_SQL_BLOCK_RE = re.compile(r"```(?:sql|sqlite)?\s*(.*?)```", re.S | re.I)


def extract_lenient_sql(text: str) -> str | None:
    """For diagnosis only, never for reward: the final SQL of an answer that
    ignored the <solution> format, i.e. the last ```sql block. Separates
    "cannot follow the format" from "cannot write the SQL"."""
    sql, _ = extract_final_sql(text)
    if sql:
        return sql
    blocks = [b.strip() for b in _SQL_BLOCK_RE.findall(text) if b.strip()]
    return blocks[-1] if blocks else None


@lru_cache(maxsize=8192)
def _gold_result(db_path: str, gold_sql: str) -> ExecResult:
    # GRPO samples several responses per question, all checked against the
    # same gold query: run it once per process.
    return execute(db_path, gold_sql, timeout_s=REWARD_TIMEOUT_S, max_rows=REWARD_MAX_ROWS)


def score_sql(db_path: str | Path, pred_sql: str, gold_sql: str) -> dict[str, float]:
    gold = _gold_result(str(db_path), gold_sql)
    pred = execute(db_path, pred_sql, timeout_s=REWARD_TIMEOUT_S, max_rows=REWARD_MAX_ROWS)
    # pred_empty / gold_empty expose the empty-result shortcut: when the gold
    # query returns no rows, any query that returns nothing "matches" it
    # (45% of SkyRL-SQL-653's gold queries do, docs/failures.md #6).
    out = {"acc": 0.0, "acc_strict": 0.0, "exec_ok": float(pred.ok), "gold_error": 0.0,
           "pred_empty": float(pred.ok and not pred.rows), "gold_empty": float(gold.ok and not gold.rows)}
    if not gold.ok:
        # A data problem, not the model's: log it and give no signal either way.
        logger.warning("gold SQL failed on %s: %s", db_path, gold.error)
        out["gold_error"] = 1.0
        return out
    if not pred.ok:
        return out
    out["acc"] = float(rows_match(pred.rows, gold.rows, mode="set"))
    out["acc_strict"] = float(rows_match(pred.rows, gold.rows, mode="strict", ordered=has_top_level_order_by(gold_sql)))
    return out


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict[str, Any] | None = None,
    **_: Any,
) -> dict[str, float]:
    """verl's custom reward function entry point (reward.custom_reward_function)."""
    info = extra_info or {}
    # Every key must be present in every result: verl stacks them into arrays.
    out = {"score": -1.0, "acc": 0.0, "acc_strict": 0.0, "exec_ok": 0.0, "format_ok": 0.0,
           "unfinished": 0.0, "gold_error": 0.0, "pred_empty": 0.0, "gold_empty": 0.0}
    sql, why = extract_final_sql(solution_str)
    if sql is None:
        out["unfinished"] = float(why == "unfinished")
        return out
    out["format_ok"] = 1.0
    db_path = dbs.resolve(info["db_family"], info["db_id"])
    out.update(score_sql(db_path, sql, ground_truth))
    out["score"] = out["acc"]
    return out
