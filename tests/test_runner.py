import json
from types import SimpleNamespace as NS

import pytest

from sar.agent.runner import AgentConfig, run_agent
from sar.data.prompts import build_messages
from sar.eval.run_eval import pass_at_k, summarize


class ScriptedClient:
    """Stands in for AsyncOpenAI: returns the scripted assistant turns in order
    and records the messages it was sent."""

    def __init__(self, turns):
        self.turns, self.sent = list(turns), []
        self.chat = NS(completions=NS(create=self._create))

    async def _create(self, **kw):
        self.sent.append([dict(m) for m in kw["messages"]])
        content, call, finish = self.turns.pop(0)
        tool_calls = None
        if call is not None:
            tool_calls = [NS(id=f"call{len(self.sent)}", function=NS(name="execute_sql", arguments=json.dumps({"sql": call})))]
        msg = NS(content=content, tool_calls=tool_calls)
        return NS(choices=[NS(message=msg, finish_reason=finish)], usage=NS(completion_tokens=10))


MSGS = build_messages("CREATE TABLE singer (...)", "Names of US singers?")


async def test_tool_then_answer(db):
    client = ScriptedClient([
        ("Check the values.", "SELECT DISTINCT country FROM singer", "tool_calls"),
        ("<solution>SELECT name FROM singer WHERE country = 'US'</solution>", None, "stop"),
    ])
    res = await run_agent(client, AgentConfig(model="m"), MSGS, db)
    assert res.stop_reason == "answered" and res.tool_calls == 1
    assert res.final_sql == "SELECT name FROM singer WHERE country = 'US'"
    # The second request carries the tool result back to the model.
    tool_msg = client.sent[1][-1]
    assert tool_msg["role"] == "tool" and "| US |" in tool_msg["content"]


async def test_tool_call_budget(db):
    client = ScriptedClient([("", "SELECT 1", "tool_calls")] * 3)
    res = await run_agent(client, AgentConfig(model="m", max_tool_calls=2), MSGS, db)
    assert res.stop_reason == "max_tool_calls" and res.tool_calls == 2 and res.final_sql is None


async def test_bad_final_format(db):
    res = await run_agent(ScriptedClient([("SELECT name FROM singer", None, "stop")]), AgentConfig(model="m"), MSGS, db)
    assert res.stop_reason == "format"


async def test_length_stop(db):
    res = await run_agent(ScriptedClient([("<solution>SELECT", None, "length")]), AgentConfig(model="m"), MSGS, db)
    assert res.stop_reason == "length"


def test_pass_at_k():
    assert pass_at_k(5, 0, 1) == 0.0
    assert pass_at_k(5, 5, 1) == 1.0
    assert pass_at_k(5, 1, 1) == pytest.approx(0.2)
    assert pass_at_k(5, 1, 5) == 1.0
    assert pass_at_k(4, 2, 2) == pytest.approx(1 - 1 / 6)


def test_summarize_uses_first_sample_for_ex():
    def rec(i, s, acc):
        return {"benchmark": "b", "index": i, "sample": s, "acc": acc, "acc_strict": acc, "stop_reason": "answered",
                "tool_calls": 1, "tool_errors": 0, "completion_tokens": 5}
    recs = [rec(0, 0, 1.0), rec(0, 1, 0.0), rec(1, 0, 0.0), rec(1, 1, 1.0)]
    s = summarize(recs, n=2)["b"]
    assert s["EX"] == 0.5 and s["pass@2"] == 1.0 and s["pass@1"] == 0.5
