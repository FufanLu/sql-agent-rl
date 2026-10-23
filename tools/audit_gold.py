"""Audits gold queries: which return no rows, and which depend on the current
date ('now', CURRENT_DATE, ...) so that their answer drifts over time.

An empty gold result is a free reward: any query that returns nothing
matches it. A date-dependent gold query was correct when the dataset was
made and may be empty or different today.
    python -P tools/audit_gold.py <processed dir> [split ...]
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

from sar.sql_env import dbs
from sar.sql_env.executor import execute

DATE_RE = re.compile(r"'now'|current_date|current_timestamp|current_time|julianday\(\s*\)", re.I)


def main() -> None:
    root = Path(sys.argv[1])
    splits = sys.argv[2:] or ["train", "val"]
    report = {}
    for split in splits:
        df = pd.read_parquet(root / f"{split}.parquet")
        c: Counter = Counter()
        empty_ids = []
        for row in df.itertuples():
            info, gold = row.extra_info, row.reward_model["ground_truth"]
            res = execute(dbs.resolve(info["db_family"], info["db_id"]), gold, timeout_s=30, max_rows=1)
            dated = bool(DATE_RE.search(gold))
            empty = res.ok and len(res.rows) == 0
            c["total"] += 1
            c["date_dependent"] += dated
            c["empty"] += empty
            c["empty_and_date_dependent"] += empty and dated
            c[f"family_{info['db_family']}_empty"] += empty
            if empty:
                empty_ids.append({"index": int(info["index"]), "db_id": info["db_id"], "date_dependent": dated})
        report[split] = {"counts": dict(c), "empty": empty_ids}
        print(split, dict(c), flush=True)
    (root / "gold_audit.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
