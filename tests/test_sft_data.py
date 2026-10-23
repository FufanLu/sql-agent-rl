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
    # each sample gets its own final answer, so records are distinct conversations
    msgs = [dict(m) for m in MSGS]
    msgs[4] = {**msgs[4], "content": f"<solution>SELECT {sample}</solution>"}
    return {"benchmark": "b", "index": index, "sample": sample, "acc_strict": strict, "tool_calls": tools,
            "completion_tokens": tokens, "messages": msgs}


def test_select_keeps_shortest_correct_with_tool_use():
    recs = [rec(0, 0, 1.0, 1, 300), rec(0, 1, 1.0, 2, 100), rec(0, 2, 1.0, 1, 200),
            rec(0, 3, 0.0, 1, 50),       # wrong
            rec(1, 0, 1.0, 0, 10)]       # right but never looked at the data
    out = select(recs, per_question=2)
    assert [(r["index"], r["sample"]) for r in out] == [(0, 1), (0, 2)]


def test_select_drops_leaked_template_tokens_only_where_text_is_kept():
    in_call_turn = [dict(m) for m in MSGS]
    in_call_turn[2] = {**in_call_turn[2], "content": "<|im_start|>"}  # dropped by the export anyway
    in_answer = [dict(m) for m in MSGS]
    in_answer[4] = {**in_answer[4], "content": "<|im_start|><solution>SELECT 1</solution>"}
    a, b = rec(0, 0, 1.0, 1, 10), rec(1, 0, 1.0, 1, 10)
    a["messages"], b["messages"] = in_call_turn, in_answer
    assert [r["index"] for r in select([a, b], per_question=2)] == [0]


def test_select_skips_duplicate_trajectories():
    # two samples with the same conversation count once
    recs = [rec(0, 0, 1.0, 1, 100), rec(0, 1, 1.0, 1, 100), rec(0, 2, 1.0, 1, 120)]
    recs[1]["messages"] = recs[0]["messages"]  # sample 1 repeats sample 0 exactly
    out = select(recs, per_question=2)
    assert [r["sample"] for r in out] == [0, 2]


def test_openai_export_keeps_calls_as_dicts():
    from sar.data.sft import to_openai
    out = to_openai(MSGS)
    assert [m["role"] for m in out] == ["system", "user", "assistant", "tool", "assistant"]
    call = out[2]["tool_calls"][0]["function"]
    assert call == {"name": "execute_sql", "arguments": {"sql": "SELECT 1"}} and out[2]["content"] == ""
    assert out[4]["content"] == "<solution>SELECT 1</solution>"
