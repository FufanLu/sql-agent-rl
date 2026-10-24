"""Two eval runs (results dirs of sar.eval.run_eval) on the same questions: did
one of them learn to return empty results where the gold result is not empty?
Per benchmark, first sample of each question:
  EX            plain execution accuracy
  EX_ne         EX on non-empty-gold questions (immune to the empty shortcut)
  empty_ne      share of non-empty-gold questions answered with an empty
                result: always wrong, the shortcut showing where it does not pay
  empty_eg      share of empty-gold questions answered with an empty result
and, paired over the non-empty-gold questions, how many only A or only B
answered empty, with an exact two-sided sign test on those discordant pairs.
A question counts as empty-gold if either run saw an empty gold result (the
gold flag is 0 when a sample has no final SQL).
    python3 tools/empty_compare.py <results dir A> <results dir B>
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
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


def main() -> None:
    a, b = load(sys.argv[1]), load(sys.argv[2])
    keys = sorted(a.keys() & b.keys())
    by_bench: dict[str, list] = defaultdict(list)
    for k in keys:
        by_bench[k[0]].append(k)
    by_bench["ALL"] = keys
    print(
        f"A = {sys.argv[1]}\nB = {sys.argv[2]}\n{len(keys)} questions in both, {len(a.keys() ^ b.keys())} in one only"
    )
    print(
        f"{'benchmark':17} {'n':>5} {'n_eg':>4} | {'EX A':>6} {'EX B':>6} | {'EX_ne A':>7} {'EX_ne B':>7} |"
        f" {'empty_ne A':>10} {'empty_ne B':>10} | {'empty_eg A':>10} {'empty_eg B':>10} | only A  only B  p"
    )
    for bench, ks in sorted(by_bench.items(), key=lambda kv: (kv[0] == "ALL", kv[0])):
        eg = [k for k in ks if a[k].get("gold_empty") or b[k].get("gold_empty")]
        ne = [k for k in ks if not (a[k].get("gold_empty") or b[k].get("gold_empty"))]

        def share(run: dict, sel: list, key: str) -> str:
            return f"{sum(run[k].get(key, 0) for k in sel) / len(sel):.4f}" if sel else "-"

        only_a = sum(bool(a[k].get("pred_empty")) and not b[k].get("pred_empty") for k in ne)
        only_b = sum(bool(b[k].get("pred_empty")) and not a[k].get("pred_empty") for k in ne)
        print(
            f"{bench:17} {len(ks):5d} {len(eg):4d} | {share(a, ks, 'acc'):>6} {share(b, ks, 'acc'):>6} |"
            f" {share(a, ne, 'acc'):>7} {share(b, ne, 'acc'):>7} |"
            f" {share(a, ne, 'pred_empty'):>10} {share(b, ne, 'pred_empty'):>10} |"
            f" {share(a, eg, 'pred_empty'):>10} {share(b, eg, 'pred_empty'):>10} |"
            f" {only_a:6d} {only_b:7d}  {sign_test(only_a, only_b):.3f}"
        )


if __name__ == "__main__":
    main()
