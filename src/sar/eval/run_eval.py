"""Execution-accuracy evaluation of the SQL agent on the processed eval sets.

    python -m sar.eval.run_eval --base-url http://127.0.0.1:8000/v1 --model qwen3b \
        --data $SAR_DATA/eval_spider_dev.parquet $SAR_DATA/eval_bird_dev.parquet \
        --out results/qwen3b-base

Writes ``<out>/results.jsonl`` (one line per question and sample) and
``<out>/summary.json``. For each benchmark the summary gives:
  EX          share of questions whose (first) final SQL returns the gold rows (set comparison)
  EX_strict   the same with duplicates, and order when the gold query sorts
  pass@k      with --n > 1 (sampling): chance that at least one of k samples is right,
              by the unbiased estimator of Chen et al. 2021
  answered    share of trajectories that ended with a well-formed <solution>
  avg_tool_calls, stop reasons, tool error rate
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from collections import Counter, defaultdict
from math import comb
from pathlib import Path

import pandas as pd
from openai import AsyncOpenAI

from sar.agent.runner import AgentConfig, run_agent
from sar.reward.sql_reward import score_sql
from sar.sql_env import dbs


def pass_at_k(n: int, c: int, k: int) -> float:
    if n - c < k:
        return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)


def _messages(prompt) -> list[dict]:
    # parquet gives back a numpy array of dicts
    return [{"role": m["role"], "content": m["content"]} for m in prompt]


async def evaluate(args) -> dict:
    client = AsyncOpenAI(base_url=args.base_url, api_key=args.api_key, timeout=600, max_retries=3)
    cfg = AgentConfig(model=args.model, max_tool_calls=args.max_tool_calls, temperature=args.temperature,
                      top_p=args.top_p, max_tokens_per_turn=args.max_tokens_per_turn)
    rows = []
    for path in args.data:
        df = pd.read_parquet(path)
        if args.limit:
            df = df.head(args.limit)
        rows += df.to_dict("records")

    sem = asyncio.Semaphore(args.concurrency)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    fout = (out_dir / "results.jsonl").open("w")
    done = 0
    t0 = time.time()

    async def one(row: dict, sample: int) -> dict:
        nonlocal done
        info = row["extra_info"]
        db_path = dbs.resolve(info["db_family"], info["db_id"], args.db_root)
        async with sem:
            try:
                res = await run_agent(client, cfg, _messages(row["prompt"]), db_path,
                                      seed=None if args.temperature == 0 else args.seed + sample)
            except Exception as e:  # an endpoint error must not kill a 2-hour eval
                rec = {"error": f"{type(e).__name__}: {e}", "stop_reason": "error", "final_sql": None,
                       "tool_calls": 0, "completion_tokens": 0, "tool_errors": 0}
            else:
                rec = {"stop_reason": res.stop_reason, "final_sql": res.final_sql, "tool_calls": res.tool_calls,
                       "completion_tokens": res.completion_tokens, "tool_errors": res.tool_errors}
                if args.save_messages:
                    rec["messages"] = res.messages
        if rec["final_sql"]:
            score = await asyncio.to_thread(score_sql, db_path, rec["final_sql"], row["reward_model"]["ground_truth"])
        else:
            score = {"acc": 0.0, "acc_strict": 0.0, "exec_ok": 0.0, "gold_error": 0.0, "pred_empty": 0.0, "gold_empty": 0.0}
        rec.update(score, benchmark=row["data_source"].removeprefix("sql/"), index=int(info["index"]),
                   db_id=info["db_id"], sample=sample)
        fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
        done += 1
        if done % 200 == 0:
            print(f"{done}/{len(rows) * args.n} done, {time.time() - t0:.0f}s", flush=True)
        return rec

    recs = await asyncio.gather(*(one(r, s) for r in rows for s in range(args.n)))
    fout.close()
    # Endpoint errors are scored 0 so one bad request cannot kill a long run,
    # but many of them mean the numbers measure the plumbing, not the model.
    errors = sum(r["stop_reason"] == "error" for r in recs)
    if errors > args.max_error_rate * len(recs):
        first = next(r["error"] for r in recs if r["stop_reason"] == "error")
        raise SystemExit(f"{errors}/{len(recs)} requests failed (limit {args.max_error_rate:.0%}); first: {first[:500]}")
    summary = summarize(recs, args.n)
    summary["_meta"] = {"errors": errors, "model": args.model, "base_url": args.base_url, "n": args.n, "temperature": args.temperature,
                        "max_tool_calls": args.max_tool_calls, "seconds": round(time.time() - t0, 1),
                        "data": [str(p) for p in args.data]}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def summarize(recs: list[dict], n: int) -> dict:
    by_bench: dict[str, list[dict]] = defaultdict(list)
    for r in recs:
        by_bench[r["benchmark"]].append(r)
    out = {}
    for bench, rs in sorted(by_bench.items()):
        by_q: dict[int, list[dict]] = defaultdict(list)
        for r in rs:
            by_q[r["index"]].append(r)
        first = [sorted(v, key=lambda r: r["sample"])[0] for v in by_q.values()]
        s = {
            "questions": len(by_q),
            "EX": round(sum(r["acc"] for r in first) / len(first), 4),
            "EX_strict": round(sum(r["acc_strict"] for r in first) / len(first), 4),
            "answered": round(sum(r["stop_reason"] == "answered" for r in rs) / len(rs), 4),
            "avg_tool_calls": round(sum(r["tool_calls"] for r in rs) / len(rs), 3),
            "tool_error_rate": round(sum(r["tool_errors"] for r in rs) / max(1, sum(r["tool_calls"] for r in rs)), 4),
            "avg_completion_tokens": round(sum(r["completion_tokens"] for r in rs) / len(rs), 1),
            "gold_errors": sum(r.get("gold_error", 0) > 0 for r in first),
            # EX on questions whose gold result is not empty: immune to the empty-result shortcut
            "EX_nonempty_gold": round(sum(r["acc"] for r in first if not r.get("gold_empty"))
                                      / max(1, sum(not r.get("gold_empty") for r in first)), 4),
            "pred_empty_rate": round(sum(r.get("pred_empty", 0) for r in first) / len(first), 4),
            "stop_reasons": dict(Counter(r["stop_reason"] for r in rs)),
        }
        if n > 1:
            for k in sorted({1, min(5, n), n}):
                s[f"pass@{k}"] = round(sum(pass_at_k(len(v), int(sum(r["acc"] for r in v)), k)
                                           for v in by_q.values()) / len(by_q), 4)
        out[bench] = s
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--api-key", default="EMPTY")
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", nargs="+", type=Path, required=True)
    ap.add_argument("--db-root", default=None, help="OmniSQL data/ dir (default: $SAR_DB_ROOT)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=1, help="samples per question")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-tool-calls", type=int, default=5)
    ap.add_argument("--max-tokens-per-turn", type=int, default=2048)
    ap.add_argument("--concurrency", type=int, default=128)
    ap.add_argument("--limit", type=int, default=0, help="first N questions per file (debugging)")
    ap.add_argument("--save-messages", action="store_true")
    ap.add_argument("--max-error-rate", type=float, default=0.02, help="fail instead of reporting if more requests error")
    args = ap.parse_args()
    summary = asyncio.run(evaluate(args))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
