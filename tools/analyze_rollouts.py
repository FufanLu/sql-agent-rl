"""Per-step diagnostics from verl's rollout dumps (<ckpt>/rollouts/<step>.jsonl).

verl 0.9.1 drops our extra reward fields for training batches, and runs
started before the reward kept its own log (sar.reward.sql_reward._log)
have only the dumps. Each dump record has the prompt, the response and the
gold SQL; the question text locates the training row (and its database), and
the final SQL is scored again.
    python -P tools/analyze_rollouts.py <rollouts dir> <train parquet> [--out per_step.json]
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

from sar.reward.sql_reward import compute_score

# The dumped prompt is decoded without special tokens, so it ends with the
# remains of the assistant header: "...Question:\n<question>\nassistant\n".
QUESTION_RE = re.compile(r"Question:\n(.*?)\n(?:assistant\n?)?\Z", re.S)


def question_of(prompt: str) -> str | None:
    m = QUESTION_RE.search(prompt)
    return m.group(1).strip() if m else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("rollouts", type=Path)
    ap.add_argument("train", type=Path, nargs="+")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    by_question: dict[str, dict] = {}
    for p in args.train:
        for row in pd.read_parquet(p).itertuples():
            by_question[row.extra_info["question"].strip()] = row.extra_info

    steps = {}
    for f in sorted(args.rollouts.glob("*.jsonl"), key=lambda f: int(f.stem)):
        recs = [json.loads(line) for line in f.open()]
        agg = {k: 0.0 for k in ("n", "unmatched", "score", "acc", "format_ok", "pred_empty", "gold_empty", "tool_calls",
                                "multi_turn")}
        for r in recs:
            info = by_question.get(question_of(r["input"]) or "")
            agg["n"] += 1
            if info is None:
                agg["unmatched"] += 1
                continue
            s = compute_score("", r["output"], r["gts"], {**info, "split": "train"})
            calls = r["output"].count("<tool_call>")
            for k in ("score", "acc", "format_ok", "pred_empty", "gold_empty"):
                agg[k] += s[k]
            agg["tool_calls"] += calls
            agg["multi_turn"] += float(calls > 0)
        n = agg["n"] - agg["unmatched"]
        steps[int(f.stem)] = {"n": int(agg["n"]), "unmatched": int(agg["unmatched"]),
                              **{k: round(agg[k] / max(1, n), 4) for k in agg if k not in ("n", "unmatched")}}
        print(f.stem, steps[int(f.stem)], flush=True)
    if args.out:
        args.out.write_text(json.dumps(steps, indent=2))


if __name__ == "__main__":
    main()
