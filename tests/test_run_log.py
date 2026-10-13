import json

import pytest

from agent_eval.agent.loop import AgentConfig, RunResult, ToolCall
from agent_eval.env.tasks import Task
from agent_eval.env.tools import Answer
from agent_eval.experiment.run_log import append_record, load_records, run_record, summarize
from agent_eval.solver import Solution

CONFIG = AgentConfig("claude-haiku-5-5", "medium")

CALL = ToolCall(1, "check_schedule", {"order_ids": ["o2", "o1"]}, '{"on_time": true}', False)


@pytest.fixture
def task(city) -> Task:
    """o1, o2 and o3 of the test city. o2 o1 o3 is on time and finishes at 90, like the answer."""
    del city.orders["o4"]
    return Task("task-three", "hard", city, Solution(["o1", "o2", "o3"], 90))


def finished_run(answer: Answer | None, outcome: str = "answered") -> RunResult:
    """A run of two turns and one tool call, with 3000 input and 400 output tokens."""
    return RunResult("task-three", outcome, answer, 2, [CALL], 3000, 400, 6.5)


# run_record


def test_record_holds_the_run_and_more(task):
    record = run_record(task, CONFIG, finished_run(Answer(True, ["o2", "o1", "o3"])))
    assert record.pop("cost_usd") == pytest.approx(0.0005)
    assert record == {
        "task_id": "task-three",
        "outcome": "answered",
        "answer": {"solvable": True, "order_ids": ["o2", "o1", "o3"]},
        "turns": 2,
        "tool_calls": [
            {
                "turn": 1,
                "name": "check_schedule",
                "tool_input": {"order_ids": ["o2", "o1"]},
                "output": '{"on_time": true}',
                "is_error": False,
            }
        ],
        "input_tokens": 3000,
        "output_tokens": 400,
        "seconds": 6.5,
        "difficulty": "hard",
        "model": "claude-haiku-5-5",
        "effort": "medium",
        "verdict": "correct",
    }


def test_record_grades_the_answer(task):
    record = run_record(task, CONFIG, finished_run(Answer(True, ["o1", "o3", "o2"])))
    assert record["verdict"] == "not_on_time"


def test_record_of_a_run_without_an_answer(task):
    record = run_record(task, CONFIG, finished_run(None, outcome="turn_limit"))
    assert record["outcome"] == "turn_limit"
    assert record["answer"] is None
    assert record["verdict"] == "no_answer"


def test_record_prices_the_model_of_the_run(task):
    config = AgentConfig("claude-sonnet-5-5", "low")
    record = run_record(task, config, finished_run(Answer(False, [])))
    assert record["model"] == "claude-sonnet-5-5"
    assert record["effort"] == "low"
    assert record["cost_usd"] == pytest.approx(0.01)


def test_record_comes_back_from_json_unchanged(task):
    record = run_record(task, CONFIG, finished_run(Answer(True, ["o2", "o1", "o3"])))
    assert json.loads(json.dumps(record)) == record


# append_record and load_records


def test_append_creates_the_log(tmp_path):
    path = tmp_path / "log.jsonl"
    append_record(path, {"task_id": "task-001", "verdict": "correct"})
    assert path.read_text(encoding="utf-8") == '{"task_id": "task-001", "verdict": "correct"}\n'


def test_append_keeps_what_is_already_there(tmp_path):
    path = tmp_path / "log.jsonl"
    append_record(path, {"task_id": "task-001"})
    append_record(path, {"task_id": "task-002"})
    lines = path.read_text(encoding="utf-8").splitlines()
    assert lines == ['{"task_id": "task-001"}', '{"task_id": "task-002"}']


def test_load_records_in_order(tmp_path):
    path = tmp_path / "log.jsonl"
    records = [{"task_id": f"task-00{i}", "turns": i} for i in range(3)]
    for record in records:
        append_record(path, record)
    assert load_records(path) == records


def test_a_record_survives_the_log(task, tmp_path):
    path = tmp_path / "log.jsonl"
    record = run_record(task, CONFIG, finished_run(Answer(True, ["o2", "o1", "o3"])))
    append_record(path, record)
    assert load_records(path) == [record]


# summarize


def a_record(difficulty: str, verdict: str, cost: float) -> dict:
    """The fields of a log line that summarize reads."""
    return {"difficulty": difficulty, "verdict": verdict, "cost_usd": cost}


def test_summarize_by_difficulty_and_over_all_runs():
    records = [
        a_record("easy", "correct", 0.125),
        a_record("hard", "correct", 0.5),
        a_record("hard", "not_earliest", 0.25),
        a_record("hard", "no_answer", 0.75),
    ]
    assert summarize(records) == {
        "easy": {"runs": 1, "correct": 1, "cost_usd": 0.125},
        "hard": {"runs": 3, "correct": 1, "cost_usd": 1.5},
        "all": {"runs": 4, "correct": 2, "cost_usd": 1.625},
    }


def test_summarize_when_nothing_is_correct():
    records = [a_record("medium", "missed_solution", 0.5), a_record("medium", "no_answer", 0.25)]
    assert summarize(records) == {
        "medium": {"runs": 2, "correct": 0, "cost_usd": 0.75},
        "all": {"runs": 2, "correct": 0, "cost_usd": 0.75},
    }


def test_summarize_no_records():
    assert summarize([]) == {}
