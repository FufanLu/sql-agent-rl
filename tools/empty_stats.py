"""Does the policy learn to return empty results? Per training step, from the
reward's own trajectory log ($SAR_REWARD_LOG_DIR/rewards_<pid>.jsonl, one
line per scored trajectory):
  gold_empty  share of samples whose gold query returns no rows
  pred_empty  share of samples whose final SQL returns no rows, split by
              whether the gold is empty (there it scores) or not (there it
              is always wrong: a sign of over-strict filters)
  acc         accuracy on empty-gold and on non-empty-gold samples
The reward computes gold_empty only for samples with a final SQL (format
failures carry 0), so questions are classified by their gold result as seen
in any formatted sample of that question; never-formatted questions count as
"unknown".
    python3 tools/empty_stats.py <reward_log dir> [--every 5]
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("log_dir")
    ap.add_argument("--every", type=int, default=5, help="print every Nth step (and the first and last)")
    args = ap.parse_args()

    by_step: dict[int, list[dict]] = defaultdict(list)
    for f in sorted(Path(args.log_dir).glob("rewards_*.jsonl")):
        for line in f.open():
            r = json.loads(line)
            if r["split"] == "train":
                by_step[r["policy_step"]].append(r)

    # A question's gold result does not change within a run (date-dependent
    # gold SQL is evaluated on the same day), so one formatted sample decides it.
    gold_empty_q: dict[int, bool] = {}
    for rows in by_step.values():
        for r in rows:
            if r["format_ok"] and not r["gold_error"]:
                gold_empty_q[r["index"]] = bool(r["gold_empty"])

    def share(rows: list[dict], key: str) -> str:
        return f"{sum(r[key] for r in rows) / len(rows):.3f}" if rows else "  -  "

    steps = sorted(by_step)
    unknown = len({r["index"] for rows in by_step.values() for r in rows} - gold_empty_q.keys())
    print(f"questions: {len(gold_empty_q)} classified ({sum(gold_empty_q.values())} with empty gold), {unknown} never formatted")
    print("policy_step k = the k+1-th training step (verl counts from 1)")
    print("step     n  score  gold_empty* pred_empty | empty gold: n  pred_empty   acc | non-empty gold: n  pred_empty   acc")
    for s in steps:
        if s not in (steps[0], steps[-1]) and s % args.every:
            continue
        rows = by_step[s]
        eg = [r for r in rows if gold_empty_q.get(r["index"]) is True]
        ng = [r for r in rows if gold_empty_q.get(r["index"]) is False]
        score = sum(r["score"] for r in rows) / len(rows)
        print(f"{s:4d} {len(rows):5d} {score:6.3f}  {share(rows, 'gold_empty'):>10}  {share(rows, 'pred_empty'):>10} |"
              f" {len(eg):14d}  {share(eg, 'pred_empty'):>10} {share(eg, 'acc'):>5} |"
              f" {len(ng):17d}  {share(ng, 'pred_empty'):>10} {share(ng, 'acc'):>5}")
    print("* gold_empty column: the per-sample flag, 0 for format failures")


if __name__ == "__main__":
    main()
