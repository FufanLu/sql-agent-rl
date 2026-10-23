"""Gradient probe for the SFT data without any training framework: plain
PyTorch forward/backward of the assistant-token loss on the first SFT
examples, reporting the gradient norm and non-finite parameter gradients.

    python -P tools/grad_probe.py <model dir> <sft json> [--n 8] [--dtype bf16|fp32] [--ckpt]

Used to locate where the SFT run's inf gradients come from (docs/failures.md
#24): if plain PyTorch gets finite gradients, the problem is in the trainer
layer; if not, in the model, its dtype, or the data.
"""

from __future__ import annotations

import argparse
import json
import math

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from sar.data.prompts import TOOL_SCHEMA


def to_messages(row: dict) -> list[dict]:
    msgs = [{"role": "system", "content": row["system"]}]
    for t in row["conversations"]:
        if t["from"] == "human":
            msgs.append({"role": "user", "content": t["value"]})
        elif t["from"] == "observation":
            msgs.append({"role": "tool", "content": t["value"]})
        elif t["from"] == "function_call":
            call = json.loads(t["value"])
            msgs.append({"role": "assistant", "content": "",
                         "tool_calls": [{"type": "function", "function": call}]})
        else:
            msgs.append({"role": "assistant", "content": t["value"]})
    return msgs


def encode(tok, msgs: list[dict]) -> tuple[list[int], list[int]]:
    """Token ids and labels (-100 except on assistant turns), from the
    difference between successive prefixes of the conversation."""
    ids: list[int] = []
    labels: list[int] = []
    for i in range(1, len(msgs) + 1):
        prefix = tok.apply_chat_template(msgs[:i], tools=[TOOL_SCHEMA], tokenize=True, return_dict=False)
        new = prefix[len(ids):]
        assert prefix[: len(ids)] == ids, "prefix tokenization is not monotone"
        is_assistant = msgs[i - 1]["role"] == "assistant"
        ids += new
        labels += new if is_assistant else [-100] * len(new)
    return ids, labels


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("data")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp32"])
    ap.add_argument("--ckpt", action="store_true", help="gradient checkpointing")
    args = ap.parse_args()

    import transformers
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype, attn_implementation="sdpa").cuda()
    model.train()
    if args.ckpt:
        model.gradient_checkpointing_enable()
    with open(args.data) as f:
        rows = json.load(f)[: args.n]
    print(f"transformers {transformers.__version__}, dtype {args.dtype}, ckpt {args.ckpt}, examples {len(rows)}")
    for j, row in enumerate(rows):
        model.zero_grad(set_to_none=True)
        ids, labels = encode(tok, to_messages(row))
        x = torch.tensor([ids], device="cuda")
        y = torch.tensor([labels], device="cuda")
        out = model(input_ids=x, labels=y)
        loss = out.loss
        logits_ok = bool(torch.isfinite(out.logits).all())
        loss.backward()
        sq, bad = 0.0, []
        for name, p in model.named_parameters():
            if p.grad is None:
                continue
            g = p.grad.float()
            if not torch.isfinite(g).all():
                bad.append(name)
            else:
                sq += float(g.pow(2).sum())
        print(f"ex {j}: tokens {len(ids)} labeled {sum(t != -100 for t in labels)} loss {float(loss):.4f} "
              f"logits finite {logits_ok} grad_norm {math.sqrt(sq):.3f} non-finite grads {len(bad)} {bad[:3]}",
              flush=True)


if __name__ == "__main__":
    main()
