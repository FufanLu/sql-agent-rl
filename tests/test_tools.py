import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location("analyze_rollouts", Path(__file__).parents[1] / "tools/analyze_rollouts.py")
analyze_rollouts = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analyze_rollouts)


def test_question_from_dumped_prompt():
    # tail of a real dump (qwen3b_mt rollouts/5.jsonl)
    prompt = ("    PRIMARY KEY (County_Id)\n);\n\nExternal knowledge:\n(none)\n\nQuestion:\n"
              "For each county, find the name of the county and the number of delegates from that county.\nassistant\n")
    assert analyze_rollouts.question_of(prompt) == (
        "For each county, find the name of the county and the number of delegates from that county.")
