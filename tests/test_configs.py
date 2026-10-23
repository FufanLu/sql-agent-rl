from pathlib import Path

import yaml

from sar.data.prompts import TOOL_SCHEMA, build_messages

ROOT = Path(__file__).resolve().parents[1]


def test_tool_yaml_matches_prompt_schema():
    cfg = yaml.safe_load((ROOT / "configs/tools/sql.yaml").read_text())
    (tool,) = cfg["tools"]
    assert tool["class_name"] == "sar.agent.sql_tool.SqlTool"
    assert tool["tool_schema"] == TOOL_SCHEMA


def test_grpo_runs_compose_base():
    for path in (ROOT / "configs/grpo").glob("*.yaml"):
        cfg = yaml.safe_load(path.read_text())
        if path.name == "base.yaml":
            assert cfg["defaults"][0] == "ppo_trainer"
            assert "hydra" not in cfg  # searchpath is only allowed in the primary (run) config
        else:
            assert "base" in cfg["defaults"], path.name
            assert cfg["hydra"]["searchpath"] == ["pkg://verl.trainer.config"], path.name


def test_messages_fill_every_field():
    msgs = build_messages("CREATE TABLE t (a INT);", "How many rows?", "None", max_tool_calls=3)
    assert [m["role"] for m in msgs] == ["system", "user"]
    assert "at most 3 times" in msgs[0]["content"]
    assert "(none)" in msgs[1]["content"] and "{" not in msgs[1]["content"]
