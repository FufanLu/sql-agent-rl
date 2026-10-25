"""Two eval runs (results dirs of sar.eval.run_eval) on the same questions:
which one is right more often, and how do they use the tool? First sample of
each question, paired:
  per benchmark  EX of each run, questions right in only one of them with an
                 exact two-sided sign test, and the share of questions where
                 both runs gave the same final SQL (two greedy runs of one model
                 give 88-93% here, so far lower means different models)
  tool calls     per number of execute_sql calls: questions, EX, and how many
                 of them repeat an identical SQL call (needs --save-messages)
  by call count  questions right in only one run, split by whether A called
                 the tool more, fewer or as often as B
    python3 tools/paired_compare.py <results dir A> <results dir B>
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from math import comb


def load(d: str) -> dict:
    first: dict = {}
    with open(f"{d}/results.jsonl") as f:
        for r in map(json.loads, f):
            k = (r["benchmark"], r["index"])
            if k not in first or r["sample"] < first[k]["sample"]:
                first[k] = r
    return first


def sign_test(x: int, y: int) -> float:
    n = x + y
    if n == 0:
        return 1.0
    tail = sum(comb(n, i) for i in range(min(x, y) + 1)) / 2**n
    return min(1.0, 2 * tail)


def called_sql(r: dict) -> list[str] | None:
    """The SQL of each tool call, or None when the messages were not saved."""
    if not r.get("messages"):
        return None
    out = []
    for m in r["messages"]:
        for tc in m.get("tool_calls") or []:
            try:
                out.append(json.loads(tc["function"]["arguments"])["sql"].strip())
            except (ValueError, KeyError, TypeError, AttributeError):
                out.append("<unparsed>")
    return out


def main() -> None:
    a, b = load(sys.argv[1]), load(sys.argv[2])
    keys = sorted(a.keys() & b.keys())
    by_bench: dict[str, list] = defaultdict(list)
    for k in keys:
        by_bench[k[0]].append(k)
    by_bench["ALL"] = keys
    print(f"A = {sys.argv[1]}\nB = {sys.argv[2]}\n{len(keys)} questions in both")
    print(f"{'benchmark':17} {'n':>5} | {'EX A':>6} {'EX B':>6} | only A  only B  p      | same SQL")
    for bench, ks in sorted(by_bench.items(), key=lambda kv: (kv[0] == "ALL", kv[0])):
        only_a = sum(a[k]["acc"] > b[k]["acc"] for k in ks)
        only_b = sum(b[k]["acc"] > a[k]["acc"] for k in ks)
        same = sum((a[k].get("final_sql") or "") == (b[k].get("final_sql") or "") for k in ks)
        ex_a = sum(a[k]["acc"] for k in ks) / len(ks)
        ex_b = sum(b[k]["acc"] for k in ks) / len(ks)
        print(
            f"{bench:17} {len(ks):5d} | {ex_a:.4f} {ex_b:.4f} | {only_a:6d} {only_b:7d}  {sign_test(only_a, only_b):.3f}  |"
            f" {same / len(ks):.3f}"
        )

    print("\ntool calls: questions, EX, questions repeating an identical SQL call")
    for name, run in (("A", a), ("B", b)):
        n: Counter = Counter()
        right: Counter = Counter()
        rep: Counter = Counter()
        for k in keys:
            c = run[k]["tool_calls"]
            n[c] += 1
            right[c] += run[k]["acc"]
            sqls = called_sql(run[k])
            rep[c] += sqls is not None and len(sqls) != len(set(sqls))
        cells = [f"{c}: {n[c]} ({right[c] / n[c]:.3f}, rep {rep[c]})" for c in sorted(n)]
        print(f"  {name} " + "  ".join(cells))

    print("\nright in only one run, by who called the tool more")
    groups = {
        "A more": [k for k in keys if a[k]["tool_calls"] > b[k]["tool_calls"]],
        "A fewer": [k for k in keys if a[k]["tool_calls"] < b[k]["tool_calls"]],
        "same": [k for k in keys if a[k]["tool_calls"] == b[k]["tool_calls"]],
    }
    for label, ks in groups.items():
        only_a = sum(a[k]["acc"] > b[k]["acc"] for k in ks)
        only_b = sum(b[k]["acc"] > a[k]["acc"] for k in ks)
        print(
            f"  {label:8} {len(ks):5d} questions: only A {only_a}, only B {only_b}, p = {sign_test(only_a, only_b):.4f}"
        )


if __name__ == "__main__":
    main()
