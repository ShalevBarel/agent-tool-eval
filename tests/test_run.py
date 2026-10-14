import copy
from pathlib import Path

import pytest

from agent_eval.agent import run
from agent_eval.env.tasks import Task, save_tasks
from agent_eval.solver import Solution
from fake_client import FakeClient, response, text, tool_use

CONFIGS = Path(__file__).resolve().parent.parent / "configs"


@pytest.fixture
def tasks_file(city, tmp_path) -> Path:
    """Two tasks in the test city, without the unreachable o4.

    In task-two, o1 then o2 is on time and ends at 42, and o2 then o1 ends
    at 49. In task-three, o1 o2 o3 ends at 90, and o1 o3 o2 is late.
    """
    del city.orders["o4"]
    three = Task("task-three", "hard", copy.deepcopy(city), Solution(["o1", "o2", "o3"], 90))
    del city.orders["o3"]
    two = Task("task-two", "easy", city, Solution(["o1", "o2"], 42))
    path = tmp_path / "tasks.jsonl"
    save_tasks([two, three], path)
    return path


def use_fake_client(responses, monkeypatch):
    """Make the script use a fake client with these responses instead of the real API."""
    monkeypatch.setattr(run.anthropic, "Anthropic", lambda: FakeClient(responses))


def submit(order_ids: list[str], **tokens):
    call = tool_use("submit_answer", {"solvable": True, "order_ids": order_ids}, "toolu_b")
    return response(call, **tokens)


def test_shows_the_calls_and_both_answers(tasks_file, monkeypatch, capsys):
    use_fake_client(
        [
            response(tool_use("check_schedule", {"order_ids": ["o1"]}, "toolu_a")),
            submit(["o1", "o2"], cache_write_tokens=50, cache_read_tokens=1500),
        ],
        monkeypatch,
    )
    run.main(["--task", "task-two", "--tasks-file", str(tasks_file)])
    printed = capsys.readouterr().out
    assert printed.startswith("task-two, easy, orders o1 o2\n")
    assert 'turn  1  check_schedule {"order_ids": ["o1"]}' in printed
    assert "outcome  answered, after 2 turns" in printed
    assert "tokens   200 in, 20 out, 50 written to the cache, 1500 read from it\n" in printed
    assert "agent    o1 o2, last delivery at minute 42\n" in printed
    assert "exact    o1 o2, last delivery at minute 42\n" in printed
    assert "verdict  correct\n" in printed
    assert "cost     $0.0001\n" in printed


def test_shows_when_the_agent_finishes_later_than_the_exact_answer(tasks_file, monkeypatch, capsys):
    use_fake_client([submit(["o2", "o1"])], monkeypatch)
    run.main(["--task", "task-two", "--tasks-file", str(tasks_file)])
    printed = capsys.readouterr().out
    assert "agent    o2 o1, last delivery at minute 49\n" in printed
    assert "exact    o1 o2, last delivery at minute 42\n" in printed
    assert "verdict  not_earliest\n" in printed


def test_shows_why_an_answer_is_not_on_time(tasks_file, monkeypatch, capsys):
    use_fake_client([submit(["o1", "o3", "o2"])], monkeypatch)
    run.main(["--task", "task-three", "--tasks-file", str(tasks_file)])
    printed = capsys.readouterr().out
    late = "order o2 arrives at minute 167, after its window closes at 45"
    assert f"agent    o1 o3 o2, not on time: {late}\n" in printed
    assert "verdict  not_on_time\n" in printed


def test_shows_a_run_without_an_answer(tasks_file, monkeypatch, capsys):
    use_fake_client([response(text("Hmm."), stop_reason="end_turn")], monkeypatch)
    run.main(["--task", "task-two", "--tasks-file", str(tasks_file)])
    printed = capsys.readouterr().out
    assert "outcome  end_turn, after 1 turns" in printed
    assert "agent    no answer" in printed
    assert "verdict  no_answer\n" in printed


def test_uses_the_config_and_prices_its_model(tasks_file, monkeypatch, capsys):
    answer = submit(["o1", "o2"], input_tokens=20_000, output_tokens=3_000)
    use_fake_client([answer], monkeypatch)
    config = str(CONFIGS / "sonnet-low.toml")
    run.main(["--task", "task-two", "--tasks-file", str(tasks_file), "--config", config])
    assert "cost     $0.0700\n" in capsys.readouterr().out


def test_an_unknown_task(tasks_file, monkeypatch):
    use_fake_client([], monkeypatch)
    with pytest.raises(SystemExit):
        run.main(["--task", "task-nine", "--tasks-file", str(tasks_file)])
