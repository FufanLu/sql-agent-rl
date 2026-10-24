"""Breakdown of an eval run (results.jsonl from sar.eval.run_eval): EX by the
number of tool calls, stop reasons, and the tool error rate, per benchmark
and overall. Pure Python (runs on the login node from the code directory).
    python3 tools/eval_breakdown.py <results dir>
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict


def main() -> None:
    d = sys.argv[1]
    with open(f"{d}/results.jsonl") as f:
        rs = [json.loads(line) for line in f]
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in rs:
        groups[r["benchmark"]].append(r)
        groups["ALL"].append(r)
    for name in sorted(groups, key=lambda k: (k == "ALL", k)):
        g = groups[name]
        by_calls: dict[str, list[float]] = defaultdict(list)
        for r in g:
            by_calls[str(min(r["tool_calls"], 3)) if r["tool_calls"] < 3 else "3+"].append(r["acc"])
        calls = sum(r["tool_calls"] for r in g)
        errs = sum(r["tool_errors"] for r in g)
        print(f"== {name}: n {len(g)}, EX {sum(r['acc'] for r in g) / len(g):.3f}, tool error rate {errs / max(1, calls):.3f}")
        print("   EX by tool calls:", {k: (len(v), round(sum(v) / len(v), 3)) for k, v in sorted(by_calls.items())})
        print("   stop reasons:", dict(Counter(r["stop_reason"] for r in g)))


if __name__ == "__main__":
    main()
