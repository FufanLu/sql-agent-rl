"""The agent's instructions. One place, used by data building, SFT data,
evaluation and the deployed agent, so they all see the same prompt."""

from __future__ import annotations

SYSTEM_PROMPT = """You are a data science expert who answers questions about a SQLite database by writing SQL.

You can call the `execute_sql` tool to run a SQL query against the database and see its result (the first rows only). Use it to explore tables, check column values and formats, and verify that your query returns what the question asks for. You may call it at most {max_tool_calls} times.

When you are confident, give the final answer as one SQLite query inside <solution>...</solution>, and write nothing after it.

Rules for the final query:
- Return exactly the information the question asks for: no extra columns, nothing missing.
- It must be a single read-only SELECT (WITH ... SELECT is fine) for SQLite."""

USER_TEMPLATE = """Database schema:
{schema}

External knowledge:
{external_knowledge}

Question:
{question}"""

TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "execute_sql",
        "description": "Run one read-only SQLite query against the task's database and return the first rows of the result, or the error message.",
        "parameters": {
            "type": "object",
            "properties": {"sql": {"type": "string", "description": "A single SQLite SELECT statement."}},
            "required": ["sql"],
        },
    },
}

DEFAULT_MAX_TOOL_CALLS = 5


def build_messages(schema: str, question: str, external_knowledge: str = "", *,
                   max_tool_calls: int = DEFAULT_MAX_TOOL_CALLS) -> list[dict[str, str]]:
    ek = external_knowledge.strip()
    if ek.lower() in ("", "none", "null"):
        ek = "(none)"
    return [
        {"role": "system", "content": SYSTEM_PROMPT.format(max_tool_calls=max_tool_calls)},
        {"role": "user", "content": USER_TEMPLATE.format(schema=schema.strip(), external_knowledge=ek,
                                                          question=question.strip())},
    ]
