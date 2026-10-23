"""Is the saved lm_head the tied input embedding (as Qwen2.5-Coder-3B
expects), and what does each config say about tying?
    python -P tools/check_lm_head.py <model dir> [<model dir> ...]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from safetensors import safe_open


def find(d: Path, name: str):
    for f in sorted(d.glob("*.safetensors")):
        with safe_open(str(f), "pt") as h:
            if name in h.keys():  # noqa: SIM118 (a safetensors handle, not a dict)
                return h.get_tensor(name).float()
    return None


for arg in sys.argv[1:]:
    d = Path(arg)
    cfg = json.loads((d / "config.json").read_text())
    tie = {k: v for k, v in cfg.items() if "tie" in k}
    emb, head = find(d, "model.embed_tokens.weight"), find(d, "lm_head.weight")
    print(f"== {d.name}: tie settings {tie}")
    print(f"   embed_tokens: {None if emb is None else (tuple(emb.shape), round(float(emb.norm()), 2))}")
    if head is None:
        print("   lm_head: not stored")
    else:
        same = emb is not None and head.shape == emb.shape and bool((head - emb).abs().max() < 1e-6)
        print(f"   lm_head: {tuple(head.shape)} norm {float(head.norm()):.2f}, abs max {float(head.abs().max()):.3g}, "
              f"zeros {float((head == 0).float().mean()):.3f}, equal to embed_tokens: {same}")
