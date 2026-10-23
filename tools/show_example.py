"""Prints one training example exactly as the model sees it, and its gold
result on the real database. Used for the handout's worked example.
    python -P tools/show_example.py <model dir> <processed train.parquet> <db_id>
"""

from __future__ import annotations

import sys

import pandas as pd
from transformers import AutoTokenizer

from sar.data.prompts import TOOL_SCHEMA
from sar.sql_env import dbs
from sar.sql_env.executor import execute
from sar.sql_env.render import render


def main() -> None:
    model, parquet, db_id = sys.argv[1:4]
    df = pd.read_parquet(parquet)
    row = df[df.extra_info.map(lambda e: e["db_id"]) == db_id].iloc[0]
    msgs = [{"role": m["role"], "content": m["content"]} for m in row.prompt]
    tok = AutoTokenizer.from_pretrained(model)
    text = tok.apply_chat_template(msgs, tools=[TOOL_SCHEMA], tokenize=False, add_generation_prompt=True)
    ids = tok.apply_chat_template(msgs, tools=[TOOL_SCHEMA], tokenize=True, add_generation_prompt=True, return_dict=False)
    print("=== rendered prompt:", len(ids), "tokens ===")
    print(text)
    info = row.extra_info
    gold = row.reward_model["ground_truth"]
    print("=== gold SQL ===")
    print(gold)
    res = execute(dbs.resolve(info["db_family"], info["db_id"]), gold, timeout_s=30, max_rows=10000)
    print("=== gold result ===")
    print(render(res, max_rows=10), f"elapsed {res.elapsed_ms:.1f} ms")
    print("=== prompt token lengths over the training set ===")
    lens = [len(tok.apply_chat_template([{"role": m["role"], "content": m["content"]} for m in p], tools=[TOOL_SCHEMA],
                                        tokenize=True, add_generation_prompt=True, return_dict=False)) for p in df.prompt]
    s = pd.Series(lens)
    print(s.describe(percentiles=[0.5, 0.9, 0.99]).round(0).to_string())


if __name__ == "__main__":
    main()
