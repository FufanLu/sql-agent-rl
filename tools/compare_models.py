"""Two model directories on the same prompt: does each tokenizer render it to
the same token ids, and what does each model generate (greedy, few tokens)?
Separates "the weights changed" from "the tokenizer/template changed".
    python -P tools/compare_models.py <dir A> <dir B> [--swap]
--swap also runs each model with the other's tokenizer.
"""

from __future__ import annotations

import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from sar.data.prompts import TOOL_SCHEMA, build_messages

MSGS = build_messages("CREATE TABLE singer (singer_id INTEGER, name TEXT, country TEXT);",
                      "What are the names of singers from the US?")


def render(tok):
    return tok.apply_chat_template(MSGS, tools=[TOOL_SCHEMA], tokenize=True, add_generation_prompt=True,
                                   return_dict=False)


def generate(model_dir, tok, label):
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.bfloat16)
    ids = torch.tensor([render(tok)])
    with torch.no_grad():
        logits = model(ids).logits[0, -1].float()
        top = torch.topk(torch.softmax(logits, -1), 5)
        out = model.generate(ids, max_new_tokens=24, do_sample=False)[0, ids.shape[1]:]
    print(f"[{label}] top-5 first token:", [(tok.decode([int(i)]), round(float(p), 3)) for p, i in zip(top.values, top.indices, strict=True)])
    print(f"[{label}] greedy 24 tokens:", repr(tok.decode(out)), "ids", out[:6].tolist())


def main() -> None:
    a, b = sys.argv[1], sys.argv[2]
    ta, tb = AutoTokenizer.from_pretrained(a), AutoTokenizer.from_pretrained(b)
    ia, ib = render(ta), render(tb)
    print("prompt tokens:", len(ia), len(ib), "identical:", ia == ib)
    if ia != ib:
        i = next((k for k, (x, y) in enumerate(zip(ia, ib, strict=False)) if x != y), min(len(ia), len(ib)))
        print("first difference at", i, repr(ta.decode(ia[max(0, i - 10):i + 10])), "vs", repr(tb.decode(ib[max(0, i - 10):i + 10])))
    print("eos:", ta.eos_token, ta.eos_token_id, "|", tb.eos_token, tb.eos_token_id)
    generate(a, ta, "A weights, A tokenizer")
    generate(b, tb, "B weights, B tokenizer")
    if "--swap" in sys.argv:
        generate(b, ta, "B weights, A tokenizer")


if __name__ == "__main__":
    main()
