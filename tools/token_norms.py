"""Embedding norms of the tool-calling special tokens vs ordinary tokens.
Hypothesis (docs/failures.md #29): Qwen2.5-Coder's <tool_call> embedding is
(nearly) untrained, so with tied embeddings no hidden state can make it
likely, and SFT pushing towards it wrecks the first answer position.
    python -P tools/token_norms.py <model dir> [<model dir> ...]
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from safetensors import safe_open
from transformers import AutoTokenizer

TOKENS = ["<tool_call>", "</tool_call>", "<|im_start|>", "<|im_end|>", "<|endoftext|>", "<tool_response>"]


def embedding(d: Path) -> torch.Tensor:
    for f in sorted(d.glob("*.safetensors")):
        with safe_open(str(f), "pt") as h:
            if "model.embed_tokens.weight" in h.keys():  # noqa: SIM118 (a safetensors handle, not a dict)
                return h.get_tensor("model.embed_tokens.weight").float()
    raise SystemExit(f"no embed_tokens in {d}")


for arg in sys.argv[1:]:
    d = Path(arg)
    tok = AutoTokenizer.from_pretrained(d)
    emb = embedding(d)
    norms = emb.norm(dim=-1)
    regular = norms[: len(tok) - len(tok.get_added_vocab())]
    q = torch.quantile(regular, torch.tensor([0.01, 0.5, 0.99]))
    print(f"== {d.name}: vocab rows {emb.shape[0]}, regular-token norm p1/p50/p99 = "
          f"{q[0]:.3f} / {q[1]:.3f} / {q[2]:.3f}")
    for t in TOKENS:
        i = tok.convert_tokens_to_ids(t)
        print(f"   {t:<16} id {i:<7} norm {float(norms[i]):.4f}")
