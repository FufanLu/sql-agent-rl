"""SFT data by rejection sampling: keep a teacher's trajectories that got the
answer right, in LLaMA-Factory's sharegpt format.

    # 1. sample the teacher on the training questions, keeping full conversations
    FILES=train.parquet EVAL_ARGS="--n 4 --temperature 0.7 --save-messages" \
        MODEL_NAME=Qwen2.5-Coder-32B-Instruct TAG=teacher32b-train sbatch cluster/eval.sbatch
    # 2. convert
    python -m sar.data.sft --results $SAR_ROOT/results/teacher32b-train/results.jsonl \
        --out $SAR_ROOT/data/processed/sft --per-question 2

Selection, per question: only trajectories whose final SQL matches the gold
rows under the *strict* comparison (a stricter bar for imitation than for
reward), that made at least one tool call (the student should learn to look
before answering), keeping the ``per-question`` shortest. Questions where the
teacher never succeeds are left to RL.

LLaMA-Factory's sharegpt format has no slot for text the assistant writes
before a tool call, so that text is dropped from function_call turns; the
rendered tokens otherwise match the model's own chat template exactly
(tools/check_sft_template.py).
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from sar.data.prompts import TOOL_SCHEMA


def to_sharegpt(messages: list[dict]) -> dict:
    system = messages[0]["content"] if messages[0]["role"] == "system" else ""
    conv = []
    for m in messages[1:] if system else messages:
        if m["role"] == "user":
            conv.append({"from": "human", "value": m["content"]})
        elif m["role"] == "tool":
            conv.append({"from": "observation", "value": m["content"]})
        elif m.get("tool_calls"):
            fn = m["tool_calls"][0]["function"]
            args = fn["arguments"]
            args = json.loads(args) if isinstance(args, str) else args
            conv.append({"from": "function_call", "value": json.dumps({"name": fn["name"], "arguments": args},
                                                                       ensure_ascii=False)})
        else:
            conv.append({"from": "gpt", "value": m["content"]})
    return {"conversations": conv, "system": system, "tools": json.dumps([TOOL_SCHEMA], ensure_ascii=False)}


# Chat-template control tokens that leaked into a reply as text (the 32B
# teacher sometimes writes "<|im_start|>" before a tool call).
_TEMPLATE_TOKENS = ("<|im_start|>", "<|im_end|>", "<|endoftext|>")


def _clean(messages: list[dict]) -> bool:
    """No leaked tokens in the assistant text that the export keeps: turns
    with a tool call keep only the call's arguments (see to_sharegpt)."""
    kept = [m.get("content") or "" for m in messages if m["role"] == "assistant" and not m.get("tool_calls")]
    return not any(t in text for text in kept for t in _TEMPLATE_TOKENS)


def select(records: list[dict], per_question: int) -> list[dict]:
    by_q: dict[tuple, list[dict]] = defaultdict(list)
    for r in records:
        if (r.get("acc_strict") == 1.0 and r.get("tool_calls", 0) >= 1 and r.get("messages")
                and _clean(r["messages"])):
            by_q[(r["benchmark"], r["index"])].append(r)
    out = []
    for rs in by_q.values():
        rs.sort(key=lambda r: (r["completion_tokens"], r["sample"]))
        seen: set[str] = set()
        for r in rs:
            # Greedy-ish teachers often repeat the same trajectory; a duplicate
            # would only double that example's weight in SFT.
            key = json.dumps(r["messages"], sort_keys=True, ensure_ascii=False)
            if key in seen:
                continue
            seen.add(key)
            out.append(r)
            if len(seen) == per_question:
                break
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--per-question", type=int, default=2)
    args = ap.parse_args()
    recs = [json.loads(line) for line in args.results.open()]
    chosen = select(recs, args.per_question)
    args.out.mkdir(parents=True, exist_ok=True)
    rows = [to_sharegpt(r["messages"]) for r in chosen]
    (args.out / "sql_agent_sft.json").write_text(json.dumps(rows, ensure_ascii=False))
    info = {"sql_agent_sft": {"file_name": "sql_agent_sft.json", "formatting": "sharegpt",
                              "columns": {"messages": "conversations", "system": "system", "tools": "tools"}}}
    (args.out / "dataset_info.json").write_text(json.dumps(info, indent=2))
    questions = len({(r["benchmark"], r["index"]) for r in recs})
    solved = len({(r["benchmark"], r["index"]) for r in chosen})
    print(f"{len(recs)} trajectories over {questions} questions; kept {len(rows)} from {solved} questions")


if __name__ == "__main__":
    main()
