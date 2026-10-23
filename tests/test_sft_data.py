import json

from sar.data.sft import select, to_sharegpt

MSGS = [
    {"role": "system", "content": "sys"},
    {"role": "user", "content": "question"},
    {"role": "assistant", "content": "look first",
     "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "execute_sql", "arguments": '{"sql": "SELECT 1"}'}}]},
    {"role": "tool", "tool_call_id": "c1", "content": "| 1 |"},
    {"role": "assistant", "content": "<solution>SELECT 1</solution>"},
]


def test_sharegpt_roles_alternate():
    row = to_sharegpt(MSGS)
    assert row["system"] == "sys"
    assert [t["from"] for t in row["conversations"]] == ["human", "function_call", "observation", "gpt"]
    assert json.loads(row["conversations"][1]["value"]) == {"name": "execute_sql", "arguments": {"sql": "SELECT 1"}}
    assert json.loads(row["tools"])[0]["function"]["name"] == "execute_sql"


def rec(index, sample, strict, tools, tokens):
    return {"benchmark": "b", "index": index, "sample": sample, "acc_strict": strict, "tool_calls": tools,
            "completion_tokens": tokens, "messages": MSGS}


def test_select_keeps_shortest_correct_with_tool_use():
    recs = [rec(0, 0, 1.0, 1, 300), rec(0, 1, 1.0, 2, 100), rec(0, 2, 1.0, 1, 200),
            rec(0, 3, 0.0, 1, 50),       # wrong
            rec(1, 0, 1.0, 0, 10)]       # right but never looked at the data
    out = select(recs, per_question=2)
    assert [(r["index"], r["sample"]) for r in out] == [(0, 1), (0, 2)]
