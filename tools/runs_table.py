"""Every cluster job with its outcome: joins the start records the jobs write
($SAR_ROOT/runs.tsv, see cluster/common.sh) with Slurm's accounting.
Run on the login node; paste new rows into docs/runs.md with results.
    python3 tools/runs_table.py [--since 2026-10-22]
"""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", default="2026-10-22")
    ap.add_argument("--root", default=os.environ.get("SAR_ROOT", f"/scratch/{os.environ.get('USER')}/sar"))
    args = ap.parse_args()

    starts = {}
    tsv = Path(args.root) / "runs.tsv"
    if tsv.exists():
        for line in tsv.read_text().splitlines():
            parts = line.split("\t")
            if len(parts) >= 6:
                starts[parts[0]] = {"code": parts[4], "vars": parts[5].strip()}

    out = subprocess.run(
        ["sacct", "-X", "-n", "-P", "-S", args.since, "-o", "JobID,JobName%30,Partition,State,Start,Elapsed,NodeList"],
        capture_output=True, text=True, check=True).stdout
    print("| Job | Name | Partition | State | Start | Elapsed | Node | Code | What |")
    print("|---|---|---|---|---|---|---|---|---|")
    for line in out.splitlines():
        jid, name, part, state, start, elapsed, node = line.split("|")
        if not name.startswith("sar-"):
            continue
        s = starts.get(jid, {})
        print(f"| {jid} | {name} | {part} | {state} | {start.replace('T', ' ')} | {elapsed} | {node} "
              f"| {s.get('code', '')} | {s.get('vars', '')} |")


if __name__ == "__main__":
    main()
