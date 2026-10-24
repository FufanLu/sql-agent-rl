"""Checks for cluster/deploy_demo.sbatch: the same greedy eval run straight
against vLLM and through the gateway.
  1. agreement: same final SQL and same correctness per question;
  2. metering: one gateway usage record per model request (each assistant
     message in the saved trajectories is one request), all for the tenant;
  3. latency and tokens from the usage records.
    python3 tools/deploy_check.py <direct results dir> <gateway results dir> <usage.jsonl>
"""

from __future__ import annotations

import json
import statistics
import sys
from collections import Counter


def load(d: str) -> dict:
    with open(f"{d}/results.jsonl") as f:
        return {(r["benchmark"], r["index"], r["sample"]): r for r in map(json.loads, f)}


def pct(xs: list[float], p: float) -> float:
    return statistics.quantiles(xs, n=100)[p - 1] if len(xs) > 1 else (xs[0] if xs else float("nan"))


def main() -> None:
    direct, via, usage_path = load(sys.argv[1]), load(sys.argv[2]), sys.argv[3]
    keys = sorted(direct.keys() & via.keys())
    same_sql = sum(direct[k]["final_sql"] == via[k]["final_sql"] for k in keys)
    same_acc = sum(direct[k]["acc"] == via[k]["acc"] for k in keys)
    out: dict = {"questions": len(keys), "missing": len(direct.keys() ^ via.keys()),
                 "same_final_sql": same_sql, "same_correctness": same_acc}
    for name, res in (("direct", direct), ("gateway", via)):
        by = Counter()
        for (bench, _, _), r in res.items():
            by[bench, "n"] += 1
            by[bench, "acc"] += r["acc"]
        out[f"EX_{name}"] = {b: round(by[b, "acc"] / by[b, "n"], 3) for b, k in by if k == "n"}

    requests = sum(sum(m["role"] == "assistant" for m in r.get("messages", [])) for r in via.values())
    with open(usage_path) as f:
        recs = [json.loads(line) for line in f]
    ttft = [r["ttft_ms"] for r in recs if r["status"] == "200"]  # the upstream HTTP code; errors have names
    dur = [r["duration_ms"] for r in recs if r["status"] == "200"]
    out["metering"] = {
        "model_requests_in_trajectories": requests, "usage_records": len(recs),
        "status": dict(Counter(r["status"] for r in recs)), "tenants": dict(Counter(r["tenant"] for r in recs)),
        "prompt_tokens": sum(r["prompt_tokens"] for r in recs), "cached_tokens": sum(r["cached_tokens"] for r in recs),
        "completion_tokens": sum(r["completion_tokens"] for r in recs),
        "ttft_ms_p50_p99": [round(pct(ttft, 50), 1), round(pct(ttft, 99), 1)],
        "duration_ms_p50_p99": [round(pct(dur, 50), 1), round(pct(dur, 99), 1)],
        "retries": sum(r["retries"] for r in recs),
    }
    print(json.dumps(out, indent=1))
    diffs = [k for k in keys if direct[k]["final_sql"] != via[k]["final_sql"]][:5]
    for k in diffs:
        print("differs:", k, "| direct:", direct[k]["final_sql"], "| gateway:", via[k]["final_sql"])


if __name__ == "__main__":
    main()
