"""Stage 1 of a two-stage fine-tune for tokens the model never learned
(docs/failures.md #29): train only the embedding rows of <tool_call> and
</tool_call> (tied input/output in Qwen2.5-Coder-3B) on the SFT data, at a
high learning rate, with every other parameter frozen; stage 2 is the usual
full SFT from the result.

Why: the mean-of-pieces initialization (tools/init_tool_tokens.py) made the
token readable, but its row stays close to the average of '<', 'tool',
'_call', '>', so anything that raises its probability raises theirs too, and
sampling often spells it out in pieces (11337674). Only the row itself can
separate them, and full SFT / RL move it by ~1e-5 / ~1e-6 per step.

    python -P tools/train_tool_tokens.py <model dir> <sft json> <out dir> [--lr 1e-3] [--epochs 2]
Reports p(<tool_call>) at the answer start of held-out examples before and
after each epoch, and checks that no other weight changed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent))
from grad_probe import encode, to_messages  # noqa: E402

from sar.data.prompts import TOOL_SCHEMA  # noqa: E402

TOKENS = ["<tool_call>", "</tool_call>"]


def p_tool_call(model, tok, rows, tid) -> float:
    """Mean probability of <tool_call> as the first answer token."""
    ps = []
    with torch.no_grad():
        for row in rows:
            msgs = to_messages(row)
            prompt = tok.apply_chat_template(msgs[:2], tools=[TOOL_SCHEMA], tokenize=True, add_generation_prompt=True,
                                             return_dict=False)
            logits = model(torch.tensor([prompt], device="cuda")).logits[0, -1].float()
            ps.append(float(torch.softmax(logits, -1)[tid]))
    return sum(ps) / len(ps)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("data")
    ap.add_argument("out")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--heldout", type=int, default=20)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32, attn_implementation="sdpa").cuda()
    emb = model.get_input_embeddings().weight
    assert model.get_output_embeddings().weight.data_ptr() == emb.data_ptr(), "expects tied embeddings (Coder-3B)"
    ids = [tok.convert_tokens_to_ids(t) for t in TOKENS]
    for p in model.parameters():
        p.requires_grad_(False)
    emb.requires_grad_(True)
    mask = torch.zeros(emb.shape[0], 1, device="cuda")
    mask[ids] = 1.0
    emb.register_hook(lambda g: g * mask)  # only the two rows learn
    frozen_before = emb.detach().clone()
    opt = torch.optim.Adam([emb], lr=args.lr, weight_decay=0.0)

    with open(args.data) as f:
        rows = json.load(f)
    heldout, train = rows[: args.heldout], rows[args.heldout:]
    model.eval()  # no dropout in Qwen2 anyway; keeps generation-style numerics
    print(f"p(<tool_call>) at answer start, held-out {len(heldout)}: before {p_tool_call(model, tok, heldout, ids[0]):.3f}",
          flush=True)
    step = 0
    for epoch in range(args.epochs):
        total, n = 0.0, 0
        for i, row in enumerate(train):
            x, y = encode(tok, to_messages(row))
            loss = model(input_ids=torch.tensor([x], device="cuda"), labels=torch.tensor([y], device="cuda")).loss
            (loss / args.accum).backward()
            total, n = total + float(loss), n + 1
            if (i + 1) % args.accum == 0 or i + 1 == len(train):
                opt.step()
                opt.zero_grad(set_to_none=True)
                step += 1
        p = p_tool_call(model, tok, heldout, ids[0])
        print(f"epoch {epoch + 1}: {step} updates, mean train loss {total / n:.4f}, held-out p(<tool_call>) {p:.3f}, "
              f"row norms {[round(float(emb[i].norm()), 4) for i in ids]}", flush=True)

    changed = (emb.detach() != frozen_before).any(dim=1).nonzero().flatten().tolist()
    assert sorted(changed) == sorted(ids), f"rows other than {ids} changed: {changed[:10]}"
    model.to(torch.bfloat16).save_pretrained(args.out)
    tok.save_pretrained(args.out)
    print("saved", args.out, "; changed rows:", changed)


if __name__ == "__main__":
    main()
