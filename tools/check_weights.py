"""Sanity check of a fine-tuned model directory against its base model:
non-finite values per tensor, how far the weights moved, and whether the
saved tokenizer / chat template / generation config differ from the base.
    python -P tools/check_weights.py <fine-tuned dir> <base dir>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from safetensors import safe_open


def tensors(d: Path) -> dict[str, tuple[Path, str]]:
    out = {}
    for f in sorted(d.glob("*.safetensors")):
        with safe_open(str(f), "pt") as h:
            for k in h.keys():  # noqa: SIM118 (a safetensors handle, not a dict)
                out[k] = (f, k)
    return out


def load(ref: tuple[Path, str]) -> torch.Tensor:
    with safe_open(str(ref[0]), "pt") as h:
        return h.get_tensor(ref[1]).float()


def main() -> None:
    ft, base = Path(sys.argv[1]), Path(sys.argv[2])
    a, b = tensors(ft), tensors(base)
    print(f"tensors: fine-tuned {len(a)}, base {len(b)}; only in fine-tuned: {sorted(set(a) - set(b))[:5]}; "
          f"only in base: {sorted(set(b) - set(a))[:5]}")
    bad, moved = [], []
    for k in sorted(a):
        t = load(a[k])
        n_bad = int((~torch.isfinite(t)).sum())
        if n_bad:
            bad.append((k, n_bad, t.numel()))
        if k in b:
            w = load(b[k])
            d = (t - w).abs()
            rel = float((t - w).norm() / (w.norm() + 1e-12))
            moved.append((float(d.mean()), k, rel, float(d.max())))
    print(f"tensors with non-finite values: {len(bad)}")
    for k, n, total in bad[:10]:
        print(f"  {k}: {n}/{total}")
    moved.sort(reverse=True)
    print("largest mean |change| vs base:", [(k, f"{m:.2e}") for m, k, _, _ in moved[:5]])
    print("smallest mean |change| vs base:", [(k, f"{m:.2e}") for m, k, _, _ in moved[-3:]])
    # A few entries can change enormously while the mean stays tiny.
    by_rel = sorted(moved, key=lambda x: -x[2])
    print("largest relative change ||dW||/||W||:", [(k, f"{r:.2e}", f"max {mx:.2e}") for _, k, r, mx in by_rel[:8]])
    by_max = sorted(moved, key=lambda x: -x[3])
    print("largest single-entry change:", [(k, f"{mx:.2e}") for _, k, _, mx in by_max[:8]])
    for name in ("tokenizer_config.json", "generation_config.json", "config.json"):
        fa, fb = ft / name, base / name
        if not fa.exists():
            print(f"{name}: missing in fine-tuned")
            continue
        ja, jb = json.loads(fa.read_text()), json.loads(fb.read_text())
        diff = {k: (jb.get(k), ja.get(k)) for k in set(ja) | set(jb)
                if ja.get(k) != jb.get(k) and k not in ("transformers_version",)}
        for k, (old, new) in diff.items():
            print(f"{name} [{k}]: base={str(old)[:160]!r} -> fine-tuned={str(new)[:160]!r}")
    jinja_a, jinja_b = ft / "chat_template.jinja", base / "chat_template.jinja"
    if jinja_a.exists() or jinja_b.exists():
        same = jinja_a.exists() and jinja_b.exists() and jinja_a.read_text() == jinja_b.read_text()
        print(f"chat_template.jinja: fine-tuned exists={jinja_a.exists()} base exists={jinja_b.exists()} identical={same}")


if __name__ == "__main__":
    main()
