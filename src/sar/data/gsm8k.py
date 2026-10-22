"""GSM8K in verl's format, for the stack smoke test only (cluster/smoke.sbatch).

    python -m sar.data.gsm8k --src $SAR_ROOT/data/hf/gsm8k --out $SAR_ROOT/data/processed/gsm8k

data_source "openai/gsm8k" routes the reward to verl's built-in GSM8K scorer,
which checks the number after "####".
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

INSTRUCTION = 'Let\'s think step by step and output the final answer after "####".'


def convert(df: pd.DataFrame, split: str) -> pd.DataFrame:
    rows = []
    for i, r in enumerate(df.itertuples()):
        answer = r.answer.split("####")[-1].strip().replace(",", "")
        rows.append({
            "data_source": "openai/gsm8k",
            "prompt": [{"role": "user", "content": f"{r.question} {INSTRUCTION}"}],
            "ability": "math",
            "reward_model": {"style": "rule", "ground_truth": answer},
            "extra_info": {"split": split, "index": i},
        })
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for split in ("train", "test"):
        src = next((args.src / "main").glob(f"{split}-*.parquet"))
        convert(pd.read_parquet(src), split).to_parquet(args.out / f"{split}.parquet", index=False)
        print(split, "ok")


if __name__ == "__main__":
    main()
