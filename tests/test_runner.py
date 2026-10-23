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


@pytest.mark.parametrize("content", [
    'Let me check.\n{"name": "execute_sql", "arguments": {"sql": "SELECT DISTINCT country FROM singer"}}',
    '<tool>\n{"name": "execute_sql", "arguments": {"sql": "SELECT DISTINCT country FROM singer"}}\n</tool>',
    'Let me check.\n```json\n{"name": "execute_sql", "arguments": {"sql": "SELECT DISTINCT country FROM singer"}}\n```',
])
async def test_lenient_parser_recovers_unwrapped_calls(db, content):
    client = ScriptedClient([
        (content, None, "stop"),
        ("<solution>SELECT name FROM singer WHERE country = 'US'</solution>", None, "stop"),
    ])
    res = await run_agent(client, AgentConfig(model="m", lenient_tool_calls=True), MSGS, db)
    assert res.loose_tool_calls == 1 and res.tool_calls == 1 and res.stop_reason == "answered"
    call_turn = res.messages[2]
    assert call_turn["tool_calls"][0]["function"]["name"] == "execute_sql"
    assert json.loads(call_turn["tool_calls"][0]["function"]["arguments"]) == {"sql": "SELECT DISTINCT country FROM singer"}
    assert "{" not in call_turn["content"] and "<tool>" not in call_turn["content"]  # stored without the raw JSON
    assert "| US |" in res.messages[3]["content"]


async def test_lenient_parser_off_by_default(db):
    content = '{"name": "execute_sql", "arguments": {"sql": "SELECT 1"}}'
    res = await run_agent(ScriptedClient([(content, None, "stop")]), AgentConfig(model="m"), MSGS, db)
    assert res.stop_reason == "format" and res.tool_calls == 0


def test_lenient_parser_ignores_final_answers():
    from sar.agent.runner import parse_loose_tool_call
    assert parse_loose_tool_call('<solution>SELECT 1</solution> {"name": "execute_sql", "arguments": {"sql": "x"}}') is None
    assert parse_loose_tool_call('{"name": "execute_sql", "arguments": "SELECT 1"}') is None
