"""Does LLaMA-Factory render a multi-turn tool-calling conversation into the
same tokens as the model's own chat template (what verl and vLLM use)?

If SFT trains on one rendering and RL / serving use another, the model is
fine-tuned on a format it never sees again. Run on a CPU node with the
cluster env:
    python -P tools/check_sft_template.py /path/to/Qwen2.5-Coder-3B-Instruct
Prints the first differing position (if any) for: the HF template, and
LLaMA-Factory's "qwen" template with assistant text before a tool call
(a) dropped and (b) kept as plain content.
"""

from __future__ import annotations

import json
import sys

from transformers import AutoTokenizer

from sar.data.prompts import TOOL_SCHEMA, build_messages


def conversation():
    msgs = build_messages("CREATE TABLE singer (name TEXT, country TEXT);", "Names of US singers?")
    call = {"name": "execute_sql", "arguments": {"sql": "SELECT DISTINCT country FROM singer"}}
    msgs += [
        {"role": "assistant", "content": "", "tool_calls": [{"type": "function", "function": call}]},
        {"role": "tool", "content": "| country |\n|---|\n| US |\n(1 rows)"},
        {"role": "assistant", "content": "<solution>SELECT name FROM singer WHERE country = 'US'</solution>"},
    ]
    return msgs, call


def hf_ids(tok, msgs):
    return tok.apply_chat_template(msgs, tools=[TOOL_SCHEMA], tokenize=True, return_dict=False)


def llamafactory_ids(tok, msgs, call):
    from llamafactory.data.template import TEMPLATES

    tpl = TEMPLATES["qwen"]
    lf_msgs = []
    for m in msgs[1:]:
        if m["role"] == "user":
            lf_msgs.append({"role": "user", "content": m["content"]})
        elif m["role"] == "tool":
            lf_msgs.append({"role": "observation", "content": m["content"]})
        elif m.get("tool_calls"):
            lf_msgs.append({"role": "function", "content": json.dumps(call, ensure_ascii=False)})
        else:
            lf_msgs.append({"role": "assistant", "content": m["content"]})
    pairs = tpl.encode_multiturn(tok, lf_msgs, system=msgs[0]["content"], tools=json.dumps([TOOL_SCHEMA]))
    ids = []
    for p, r in pairs:
        ids += p + r
    return ids


def first_diff(a, b):
    for i, (x, y) in enumerate(zip(a, b, strict=False)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def main() -> None:
    tok = AutoTokenizer.from_pretrained(sys.argv[1])
    msgs, call = conversation()
    ref = hf_ids(tok, msgs)
    lf = llamafactory_ids(tok, msgs, call)
    i = first_diff(ref, lf)
    print(f"HF template: {len(ref)} tokens; LLaMA-Factory qwen: {len(lf)} tokens")
    if i is None:
        print("IDENTICAL")
        return
    print(f"first difference at token {i}")
    print("HF :", repr(tok.decode(ref[max(0, i - 30): i + 60])))
    print("LF :", repr(tok.decode(lf[max(0, i - 30): i + 60])))


if __name__ == "__main__":
    main()
