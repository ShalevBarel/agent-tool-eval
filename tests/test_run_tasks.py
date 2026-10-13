import copy
from pathlib import Path

import pytest

from agent_eval.env.tasks import Task, load_tasks, save_tasks
from agent_eval.experiment import run_tasks
from agent_eval.experiment.run_log import load_records
from agent_eval.solver import Solution
from fake_client import FakeClient, response, text, tool_use

SAVED_TASKS = Path(__file__).resolve().parent.parent / "data" / "tasks.jsonl"


@pytest.fixture
def tasks_file(city, tmp_path) -> Path:
    """Two tasks in the test city: task-000 has a solution, and task-001 has none."""
    no_solution = Task("task-001", "hard", copy.deepcopy(city), None)
    del city.orders["o4"]
    solvable = Task("task-000", "medium", city, Solution(["o1", "o2", "o3"], 90))
    path = tmp_path / "tasks.jsonl"
    save_tasks([no_solution, solvable], path)
    return path


def submit(solvable: bool, order_ids: list[str]):
    """A response that submits an answer."""
    return response(tool_use("submit_answer", {"solvable": solvable, "order_ids": order_ids}))


def use_fake_client(responses, monkeypatch) -> FakeClient:
    """Make the script use a fake client with these responses instead of the real API."""
    client = FakeClient(responses)
    monkeypatch.setattr(run_tasks.anthropic, "Anthropic", lambda **options: client)
    return client


# choose_tasks


def test_all_tasks_sorted_by_id():
    tasks = load_tasks(SAVED_TASKS)
    chosen = run_tasks.choose_tasks(list(reversed(tasks)), None, 0)
    assert [task.task_id for task in chosen] == [task.task_id for task in tasks]


def test_the_same_seed_gives_the_same_sample():
    tasks = load_tasks(SAVED_TASKS)
    first = [task.task_id for task in run_tasks.choose_tasks(tasks, 20, 0)]
    again = [task.task_id for task in run_tasks.choose_tasks(tasks, 20, 0)]
    assert first == again
    assert len(set(first)) == 20
    assert first == sorted(first)


def test_another_seed_gives_another_sample():
    tasks = load_tasks(SAVED_TASKS)
    seed_0 = [task.task_id for task in run_tasks.choose_tasks(tasks, 20, 0)]
    seed_1 = [task.task_id for task in run_tasks.choose_tasks(tasks, 20, 1)]
    assert seed_0 != seed_1


# main


def test_logs_every_run_in_task_order(tasks_file, tmp_path, monkeypatch, capsys):
    use_fake_client([submit(True, ["o2", "o1", "o3"]), submit(False, [])], monkeypatch)
    out = tmp_path / "runs" / "log.jsonl"
    run_tasks.main(["--tasks-file", str(tasks_file), "--out", str(out)])

    records = load_records(out)
    assert [record["task_id"] for record in records] == ["task-000", "task-001"]
    assert [record["verdict"] for record in records] == ["correct", "correct"]
    assert {record["model"] for record in records} == {"claude-haiku-5-5"}

    printed = capsys.readouterr().out
    assert "  1/2  task-000  medium  correct" in printed
    assert "  2/2  task-001  hard    correct" in printed
    assert "\nall          2        2   100%" in printed
    assert "\ncorrect          2\n" in printed


def test_summary_shows_wrong_answers(tasks_file, tmp_path, monkeypatch, capsys):
    stop = response(text("I can't tell."), stop_reason="end_turn")
    use_fake_client([submit(False, []), stop], monkeypatch)
    run_tasks.main(["--tasks-file", str(tasks_file), "--out", str(tmp_path / "log.jsonl")])
    printed = capsys.readouterr().out
    assert "\nall          2        0     0%" in printed
    assert "\nno_answer        1\n" in printed
    assert "\nmissed_solution  1\n" in printed


def test_model_and_effort_reach_the_log(tasks_file, tmp_path, monkeypatch):
    client = use_fake_client([submit(True, ["o1", "o2", "o3"]), submit(False, [])], monkeypatch)
    out = tmp_path / "log.jsonl"
    arguments = ["--model", "claude-sonnet-5-5", "--effort", "high"]
    run_tasks.main(["--tasks-file", str(tasks_file), "--out", str(out), *arguments])
    assert client.messages.requests[0]["output_config"] == {"effort": "high"}
    assert {(record["model"], record["effort"]) for record in load_records(out)} == {
        ("claude-sonnet-5-5", "high")
    }


def test_refuses_to_add_to_an_existing_log(tasks_file, tmp_path, monkeypatch):
    client = use_fake_client([], monkeypatch)
    out = tmp_path / "log.jsonl"
    out.write_text("", encoding="utf-8")
    with pytest.raises(SystemExit):
        run_tasks.main(["--tasks-file", str(tasks_file), "--out", str(out)])
    assert client.messages.requests == []


def test_refuses_a_sample_larger_than_the_tasks(tasks_file, tmp_path, monkeypatch):
    use_fake_client([], monkeypatch)
    out = tmp_path / "log.jsonl"
    with pytest.raises(SystemExit):
        run_tasks.main(["--tasks-file", str(tasks_file), "--out", str(out), "--sample", "3"])
    assert not out.exists()
