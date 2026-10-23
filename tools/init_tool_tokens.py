"""Gives Qwen2.5-Coder's untrained tool-call tokens a usable embedding before
SFT (docs/failures.md #29): each of <tool_call> and </tool_call> gets the
mean embedding of the plain-text tokens of its own spelling ("<", "tool",
"_call", ">"), the usual initialization for a new token. For models with an
untied output layer (Coder 7B/32B) the lm_head row is set the same way.
    python -P tools/init_tool_tokens.py <base model dir> <out dir>
"""

from __future__ import annotations

import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

TOKENS = ["<tool_call>", "</tool_call>"]


def main() -> None:
    src, dst = sys.argv[1], sys.argv[2]
    tok = AutoTokenizer.from_pretrained(src)
    model = AutoModelForCausalLM.from_pretrained(src, dtype=torch.bfloat16)
    emb = model.get_input_embeddings().weight
    head = model.get_output_embeddings().weight
    tied = head.data_ptr() == emb.data_ptr()
    with torch.no_grad():
        for t in TOKENS:
            tid = tok.convert_tokens_to_ids(t)
            # Tokenize the spelling as plain text: without its last character
            # the string no longer matches the added token, then append ">".
            pieces = (tok(t[:-1], add_special_tokens=False).input_ids
                      + tok(t[-1], add_special_tokens=False).input_ids)
            assert tid not in pieces and len(pieces) > 1, (t, pieces)
            before = float(emb[tid].float().norm())
            emb[tid] = emb[pieces].float().mean(0).to(emb.dtype)
            if not tied:
                head[tid] = head[pieces].float().mean(0).to(head.dtype)
            print(f"{t}: id {tid}, pieces {[tok.decode([p]) for p in pieces]}, "
                  f"embedding norm {before:.4f} -> {float(emb[tid].float().norm()):.4f}, tied={tied}")
    model.save_pretrained(dst)
    tok.save_pretrained(dst)
    print("saved", dst)


if __name__ == "__main__":
    main()
