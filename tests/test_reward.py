import pytest

from sar.reward.sql_reward import compute_score, extract_final_sql

GOLD = "SELECT name FROM singer WHERE country = 'US'"
INFO = {"db_family": "spider", "db_id": "concert_singer"}


@pytest.fixture(autouse=True)
def _root(db_root, monkeypatch):
    monkeypatch.setenv("SAR_DB_ROOT", str(db_root))


def trajectory(*final: str) -> str:
    """A decoded two-turn rollout as verl passes it: tool call, tool result, final turn."""
    return (
        "<think>look at the table</think>\n<tool_call>\n"
        '{"name": "execute_sql", "arguments": {"sql": "SELECT * FROM singer LIMIT 3"}}\n</tool_call>\n'
        "user\n<tool_response>\n| singer_id | name |\n|---|---|\n| 1 | Joe |\n</tool_response>\nassistant\n"
        + "".join(final)
    )


def test_correct_answer_scores_1():
    r = compute_score("spider", trajectory("<solution>SELECT name FROM singer WHERE country='US'</solution>"), GOLD, INFO)
    assert r["score"] == 1.0 and r["acc"] == 1.0 and r["format_ok"] == 1.0


def test_equivalent_sql_written_differently_scores_1():
    sql = "SELECT s.name FROM singer AS s WHERE s.country IN ('US') ORDER BY s.age"
    assert compute_score("spider", trajectory(f"<solution>{sql}</solution>"), GOLD, INFO)["score"] == 1.0


def test_wrong_rows_score_0():
    r = compute_score("spider", trajectory("<solution>SELECT name FROM singer</solution>"), GOLD, INFO)
    assert r["score"] == 0.0 and r["exec_ok"] == 1.0


def test_sql_error_scores_0():
    r = compute_score("spider", trajectory("<solution>SELECT nam FROM singer</solution>"), GOLD, INFO)
    assert r["score"] == 0.0 and r["exec_ok"] == 0.0


@pytest.mark.parametrize(
    "final",
    [
        "The answer is SELECT name FROM singer WHERE country = 'US'",  # no tags
        "<solution>SELECT 1</solution><solution>SELECT 2</solution>",  # two answers
        "<solution>   </solution>",
    ],
)
def test_format_failures_score_minus_1(final):
    assert compute_score("spider", trajectory(final), GOLD, INFO)["score"] == -1.0


def test_only_the_final_turn_counts():
    # A <solution> seen in an earlier turn is not the final answer.
    text = "<solution>SELECT name FROM singer WHERE country='US'</solution>" + trajectory("I am not sure.")
    assert compute_score("spider", text, GOLD, INFO)["score"] == -1.0


def test_unfinished_trajectory():
    text = trajectory('<tool_call>\n{"name": "execute_sql", "arguments": {"sql": "SELECT 1"}}\n</tool_call>')
    r = compute_score("spider", text, GOLD, INFO)
    assert r["score"] == -1.0 and r["unfinished"] == 1.0


def test_write_attempt_cannot_score():
    r = compute_score("spider", trajectory("<solution>DELETE FROM singer</solution>"), GOLD, INFO)
    assert r["score"] == 0.0 and r["exec_ok"] == 0.0


def test_strict_mode_reported_separately():
    # Missing DISTINCT: same set, different multiset.
    gold = "SELECT DISTINCT year FROM concert"
    r = compute_score("spider", trajectory("<solution>SELECT year FROM concert</solution>"), gold, INFO)
    assert r["acc"] == 1.0 and r["acc_strict"] == 0.0


def test_markdown_fence_is_stripped():
    sql, _ = extract_final_sql("<solution>```sql\nSELECT 1\n```</solution>")
    assert sql == "SELECT 1"


def test_result_keys_are_constant():
    a = compute_score("spider", trajectory("no answer"), GOLD, INFO)
    b = compute_score("spider", trajectory(f"<solution>{GOLD}</solution>"), GOLD, INFO)
    assert a.keys() == b.keys()


def test_empty_result_shortcut_is_visible():
    # Gold returns no rows; a query that returns nothing "matches" it.
    gold = "SELECT name FROM singer WHERE age > 100"
    r = compute_score("spider", trajectory("<solution>SELECT name FROM singer WHERE 1 = 0</solution>"), gold, INFO)
    assert r["score"] == 1.0 and r["gold_empty"] == 1.0 and r["pred_empty"] == 1.0
