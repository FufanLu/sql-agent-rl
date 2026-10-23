"""Why do SFT models that fit their training data (low loss) end every answer
at once (docs/failures.md #29)? For SFT example 0:
  1. are the generation-prompt tokens a prefix of the full training sequence?
  2. per-token probability of the true next token at the first answer tokens;
  3. top-5 at the answer start, from a full-sequence pass and a prompt-only pass.
    python -P tools/first_token_probe.py <sft json> <model dir> [<model dir> ...]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent))
from grad_probe import encode, to_messages  # noqa: E402

from sar.data.prompts import TOOL_SCHEMA  # noqa: E402


def main() -> None:
    data, models = sys.argv[1], sys.argv[2:]
    with open(data) as f:
        row = json.load(f)[0]
    for d in models:
        tok = AutoTokenizer.from_pretrained(d)
        model = AutoModelForCausalLM.from_pretrained(d, dtype=torch.float32).eval()
        msgs = to_messages(row)
        full, labels = encode(tok, msgs)
        prompt = tok.apply_chat_template(msgs[:2], tools=[TOOL_SCHEMA], tokenize=True, add_generation_prompt=True,
                                         return_dict=False)
        n = len(prompt)
        print(f"== {Path(d).name}")
        print("   prompt is a prefix of the training sequence:", full[:n] == prompt, f"(prompt {n} tokens)")
        if full[:n] != prompt:
            i = next(k for k, (a, b) in enumerate(zip(full, prompt, strict=False)) if a != b)
            print("   first difference at", i, "train:", [tok.decode([t]) for t in full[i - 3:i + 3]],
                  "prompt:", [tok.decode([t]) for t in prompt[i - 3:i + 3]])
        print("   tokens around the answer start:", [(t, tok.decode([t])) for t in full[n - 4:n + 4]])
        with torch.no_grad():
            logp = torch.log_softmax(model(torch.tensor([full])).logits[0].float(), -1)
            logp_prompt = torch.log_softmax(model(torch.tensor([prompt])).logits[0, -1].float(), -1)
        print("   p(true next token) for the first answer tokens (full-sequence pass):")
        for pos in range(n - 1, n + 7):
            nxt = full[pos + 1]
            print(f"     after {tok.decode([full[pos]])!r:>16} -> {tok.decode([nxt])!r:<16} p={float(logp[pos, nxt].exp()):.3f}"
                  f"  labeled={labels[pos + 1] != -100}")
        top_full = torch.topk(logp[n - 1].exp(), 5)
        top_prompt = torch.topk(logp_prompt.exp(), 5)
        print("   top-5 at answer start, full pass:  ", [(tok.decode([int(i)]), round(float(p), 3)) for p, i in zip(top_full.values, top_full.indices, strict=True)])
        print("   top-5 at answer start, prompt pass:", [(tok.decode([int(i)]), round(float(p), 3)) for p, i in zip(top_prompt.values, top_prompt.indices, strict=True)])


if __name__ == "__main__":
    main()
