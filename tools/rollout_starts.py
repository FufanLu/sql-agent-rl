"""How do sampled responses start, per training step? Counts the first 40
characters of each rollout in <ckpt>/rollouts/<step>.jsonl by pattern, with
the mean score of each group, and the share of each step's 128 question
groups that carry a learning signal (mixed scores).
    python3 tools/rollout_starts.py <rollouts dir> [step ...]
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

PATTERNS = [  # first match wins
    ("<tool_call> token", re.compile(r"^\s*<tool_call>")),
    ("pieces of <tool_call> as text", re.compile(r"^\s*(<\s*)?(tool|_call|>)")),
    ("<solution> directly", re.compile(r"^\s*<solution>")),
    ("prose first", re.compile(r"^\s*[A-Za-z]")),
]


def kind(text: str) -> str:
    for name, pat in PATTERNS:
        if pat.search(text):
            return name
    return "other"


def main() -> None:
    d = Path(sys.argv[1])
    steps = [int(s) for s in sys.argv[2:]] or sorted(int(f.stem) for f in d.glob("*.jsonl"))
    for step in steps:
        recs = [json.loads(line) for line in (d / f"{step}.jsonl").open()]
        by_kind: dict[str, list[float]] = defaultdict(list)
        groups: dict[str, list[float]] = defaultdict(list)
        for r in recs:
            by_kind[kind(r["output"])].append(r["score"])
            groups[r["input"]].append(r["score"])
        mixed = sum(len(set(v)) > 1 for v in groups.values())
        summary = {k: (len(v), round(sum(v) / len(v), 3)) for k, v in sorted(by_kind.items(), key=lambda x: -len(x[1]))}
        print(f"step {step}: n {len(recs)}, groups with signal {mixed}/{len(groups)}; start -> (count, mean score): {summary}")
        if step == steps[0]:
            examples = Counter(r["output"][:40] for r in recs if kind(r["output"]) != "<tool_call> token")
            print("   most common non-<tool_call> starts:", [s for s, _ in examples.most_common(5)])


if __name__ == "__main__":
    main()
