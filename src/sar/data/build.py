"""Builds the versioned training and evaluation parquet files verl reads.

    python -m sar.data.build --data-root $SAR_ROOT/data --out $SAR_ROOT/data/processed

Inputs (cluster/download_data.sbatch puts them there):
  hf/SkyRL-SQL-653-data/train.parquet   training questions (540 SynSQL + 113 Spider train)
  hf/SkyRL-SQL-eval/*.json              eval questions with schema and external knowledge
  omnisql/data/...                      the original benchmark files (for db_id) and databases

Outputs, one row per question in verl's RL dataset format:
  train.parquet, val.parquet (a held-out slice of train), eval_<benchmark>.parquet,
  and manifest.json with counts and SHA-256 of every file, so each training run
  can record exactly which data it saw.

Checks that fail the build instead of producing silently wrong data:
  - every database file exists;
  - eval files are aligned to the original benchmark by index, and the gold
    SQL must agree on every row;
  - every gold query runs (a few known-bad ones are counted and dropped);
  - no eval question (same db_id and question text) appears in train.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from pydantic import BaseModel

from sar.data.prompts import DEFAULT_MAX_TOOL_CALLS, build_messages
from sar.sql_env import dbs
from sar.sql_env.executor import execute

# eval file in SkyRL-SQL-eval -> (db family, original file under omnisql/data, its gold SQL key)
EVAL_SETS = {
    "spider_dev": ("dev_spider.json", "spider", "spider/dev.json", "query"),
    "spider_test": ("test_spider.json", "spider_test", "spider/test.json", "query"),
    "spider_dk": ("dev_spider_dk.json", "spider_dk", "Spider-DK/Spider-DK.json", "query"),
    "spider_syn": ("dev_spider_syn.json", "spider", "Spider-Syn/dev.json", "query"),
    "spider_realistic": ("dev_spider_realistic.json", "spider", "spider-realistic/spider-realistic.json", "query"),
    "bird_dev": ("dev_bird.json", "bird_dev", "bird/dev_20240627/dev.json", "SQL"),
}
TRAIN_FAMILY = {"synsql": "synsql", "spider": "spider"}


class Example(BaseModel):
    benchmark: str
    split: str
    index: int
    db_family: str
    db_id: str
    question: str
    schema_text: str
    external_knowledge: str
    gold_sql: str

    def to_verl_row(self, max_tool_calls: int) -> dict:
        target = {"db_family": self.db_family, "db_id": self.db_id}
        return {
            "data_source": f"sql/{self.benchmark}",
            "prompt": build_messages(self.schema_text, self.question, self.external_knowledge,
                                     max_tool_calls=max_tool_calls),
            "ability": "text2sql",
            "agent_name": "tool_agent",
            "reward_model": {"style": "rule", "ground_truth": self.gold_sql},
            "extra_info": {
                "split": self.split, "index": self.index, "question": self.question, **target,
                "tools_kwargs": {"execute_sql": {"create_kwargs": target}},
            },
        }


# SkyRL-SQL-653 stores the filled-in fields in the user message as
# "{db_details}:<schema>;\n {external_knowledge}: <ek>;\n {question}: <q>".
_SKYRL_USER = re.compile(r"\{db_details\}:(?P<schema>.*);\n \{external_knowledge\}:(?P<ek>.*?);\n \{question\}:(?P<q>.*)\Z", re.S)


def load_train(data_root: Path) -> list[Example]:
    df = pd.read_parquet(data_root / "hf/SkyRL-SQL-653-data/train.parquet")
    out = []
    for i, row in enumerate(df.itertuples()):
        user = next(m["content"] for m in row.prompt if m["role"] == "user")
        m = _SKYRL_USER.match(user)
        if not m:
            raise ValueError(f"train row {i}: unexpected user message layout")
        out.append(Example(benchmark="skyrl_sql_653", split="train", index=i, db_family=TRAIN_FAMILY[row.data],
                           db_id=row.db_id, question=m["q"].strip(), schema_text=m["schema"].strip(),
                           external_knowledge=m["ek"].strip(), gold_sql=row.reward_model["ground_truth"]))
    return out


def _norm_sql(s: str) -> str:
    return " ".join(s.split()).rstrip(";").lower()


def load_eval(data_root: Path, name: str) -> list[Example]:
    fname, family, orig_path, gold_key = EVAL_SETS[name]
    rows = json.loads((data_root / "hf/SkyRL-SQL-eval" / fname).read_text())
    orig = json.loads((data_root / "omnisql/data" / orig_path).read_text())
    if len(rows) != len(orig):
        raise ValueError(f"{name}: {len(rows)} rows but {len(orig)} in {orig_path}")
    out = []
    for i, (r, o) in enumerate(zip(rows, orig, strict=True)):
        if _norm_sql(r["output_seq"]) != _norm_sql(o[gold_key]):
            raise ValueError(f"{name} row {i}: gold SQL differs from {orig_path}; alignment by index is wrong")
        out.append(Example(benchmark=name, split="eval", index=i, db_family=family, db_id=o["db_id"],
                           question=r["question"], schema_text=r["schema"],
                           external_knowledge=r.get("external_knowledge") or "", gold_sql=r["output_seq"]))
    return out


@dataclass
class Report:
    kept: int = 0
    missing_db: int = 0
    gold_failed: int = 0


def validate(examples: list[Example], db_root: Path, timeout_s: float) -> tuple[list[Example], Report]:
    rep, kept = Report(), []
    for ex in examples:
        path = dbs.resolve(ex.db_family, ex.db_id, db_root)
        if not path.is_file():
            rep.missing_db += 1
            continue
        if not execute(path, ex.gold_sql, timeout_s=timeout_s, max_rows=1).ok:
            rep.gold_failed += 1
            continue
        kept.append(ex)
    rep.kept = len(kept)
    return kept, rep


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--val-size", type=int, default=32)
    ap.add_argument("--max-tool-calls", type=int, default=DEFAULT_MAX_TOOL_CALLS)
    ap.add_argument("--gold-timeout", type=float, default=30.0)
    ap.add_argument("--eval-sets", default=",".join(EVAL_SETS))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    db_root = args.data_root / "omnisql/data"
    args.out.mkdir(parents=True, exist_ok=True)
    manifest: dict = {"max_tool_calls": args.max_tool_calls, "files": {}, "reports": {}}

    evals = {name: load_eval(args.data_root, name) for name in args.eval_sets.split(",")}
    eval_keys = {(e.db_id, e.question.strip().lower()) for exs in evals.values() for e in exs}

    train = load_train(args.data_root)
    overlap = [e for e in train if (e.db_id, e.question.strip().lower()) in eval_keys]
    if overlap:
        raise ValueError(f"{len(overlap)} training questions also appear in eval sets, e.g. {overlap[0].question!r}")
    train, rep = validate(train, db_root, args.gold_timeout)
    manifest["reports"]["train"] = rep.__dict__
    random.Random(args.seed).shuffle(train)
    splits = {"val": train[: args.val_size], "train": train[args.val_size :]}
    for ex in splits["val"]:
        ex.split = "val"

    for name, exs in evals.items():
        exs, rep = validate(exs, db_root, args.gold_timeout)
        manifest["reports"][name] = rep.__dict__
        splits[f"eval_{name}"] = exs

    # What verl evaluates every few training steps: the held-out training
    # questions plus a fixed sample of Spider dev and BIRD dev. The full eval
    # sets run separately (sar.eval), on saved checkpoints.
    rng = random.Random(args.seed)
    monitor = list(splits["val"])
    for name, k in (("spider_dev", 300), ("bird_dev", 200)):
        if f"eval_{name}" in splits:
            monitor += rng.sample(splits[f"eval_{name}"], min(k, len(splits[f"eval_{name}"])))
    splits["val_monitor"] = monitor

    for split, exs in splits.items():
        path = args.out / f"{split}.parquet"
        pd.DataFrame([e.to_verl_row(args.max_tool_calls) for e in exs]).to_parquet(path, index=False)
        manifest["files"][path.name] = {"rows": len(exs), "sha256": sha256(path)}
        print(f"{path.name}: {len(exs)} rows  {manifest['reports'].get(split.removeprefix('eval_'), '')}")
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
