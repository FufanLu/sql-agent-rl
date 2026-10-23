"""Does a saved model still fit the SFT data it was trained on? Loss on the
first SFT examples (assistant tokens only, as tools/grad_probe.py encodes
them) for several model directories, plus the first tokens each generates
for example 0's prompt.
    python -P tools/sft_loss.py <sft json> <model dir> [<model dir> ...]
A fine-tuned model whose training loss ended near 0.3 should score well below
the base model here; if not, the weights on disk are not the trained weights.
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
        rows = json.load(f)[:8]
    for d in models:
        tok = AutoTokenizer.from_pretrained(d)
        model = AutoModelForCausalLM.from_pretrained(d, dtype=torch.float32)
        model.eval()
        losses = []
        with torch.no_grad():
            for row in rows:
                ids, labels = encode(tok, to_messages(row))
                out = model(input_ids=torch.tensor([ids]), labels=torch.tensor([labels]))
                losses.append(round(float(out.loss), 3))
            msgs = to_messages(rows[0])
            prompt = tok.apply_chat_template(msgs[:2], tools=[TOOL_SCHEMA], tokenize=True,
                                             add_generation_prompt=True, return_dict=False)
            x = torch.tensor([prompt])
            probs = torch.softmax(model(x).logits[0, -1].float(), -1)
            top = torch.topk(probs, 3)
            gen = model.generate(x, max_new_tokens=20, do_sample=False)[0, len(prompt):]
        cfg = json.loads((Path(d) / "config.json").read_text())
        print(f"== {Path(d).name}: tie={cfg.get('tie_word_embeddings')} rope={cfg.get('rope_parameters') or cfg.get('rope_theta')}")
        print(f"   loss on 8 SFT examples: {losses}  mean {sum(losses) / len(losses):.3f}")
        print(f"   first token top-3: {[(tok.decode([int(i)]), round(float(p), 3)) for p, i in zip(top.values, top.indices, strict=True)]}")
        print(f"   greedy 20 tokens: {tok.decode(gen)!r}", flush=True)


if __name__ == "__main__":
    main()
