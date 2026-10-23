"""Builds every example of the SFT parquet through verl's MultiTurnSFTDataset
(the class the verl SFT trainer uses) on a CPU node, before spending GPU time:
its per-turn tokenization must equal rendering the whole conversation (the
dataset's own sanity check raises otherwise), every example must fit, and the
loss mask must cover exactly the assistant turns.
    python -P tools/check_verl_sft_data.py <model dir> <sft_messages.parquet> [max_length]
"""

from __future__ import annotations

import sys

from omegaconf import OmegaConf
from transformers import AutoTokenizer
from verl.utils.dataset.multiturn_sft_dataset import MultiTurnSFTDataset


def main() -> None:
    model, parquet = sys.argv[1], sys.argv[2]
    max_length = int(sys.argv[3]) if len(sys.argv) > 3 else 16384
    tok = AutoTokenizer.from_pretrained(model)
    cfg = OmegaConf.create({"messages_key": "messages", "tools_key": "tools", "max_length": max_length,
                            "truncation": "error", "pad_mode": "no_padding", "ignore_input_ids_mismatch": False})
    ds = MultiTurnSFTDataset(parquet_files=[parquet], tokenizer=tok, config=cfg)
    errors, lengths, labeled = [], [], []
    for i in range(len(ds)):
        try:
            ex = ds[i]
        except Exception as e:  # the sanity check raises AssertionError on a mismatch
            errors.append((i, f"{type(e).__name__}: {str(e)[:120]}"))
            continue
        ids, mask = ex["input_ids"], ex["loss_mask"]
        lengths.append(int(ids.shape[0]))
        labeled.append(int(mask.sum()))
        if i == 0:
            print("example 0 keys:", sorted(ex.keys()))
            print("example 0 tokens", int(ids.shape[0]), "loss-masked", int(mask.sum()))
            print("example 0 loss-masked text:", repr(tok.decode(ids[mask.bool()])))
    print(f"examples {len(ds)}, errors {len(errors)}, max length {max(lengths, default=0)}, "
          f"labeled tokens min {min(labeled, default=0)} max {max(labeled, default=0)}")
    for e in errors[:5]:
        print("error", e)


if __name__ == "__main__":
    main()
