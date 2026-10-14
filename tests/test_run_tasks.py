import copy
from pathlib import Path

import pytest

from agent_eval.agent.loop import AgentConfig
from agent_eval.env.tasks import Task, load_tasks
from agent_eval.experiment.run_log import append_record, load_records
from agent_eval.experiment.run_tasks import (
    check_log_matches,
    choose_tasks,
    cost_without_cache,
    run_tasks,
    show_summary,
)
from agent_eval.solver import Solution
from fake_client import FakeClient, response, text, tool_use

SAVED_TASKS = Path(__file__).resolve().parent.parent / "data" / "tasks.jsonl"

CONFIG = AgentConfig("claude-haiku-5-5", "medium")


@pytest.fixture
def tasks(city) -> list[Task]:
    """Two tasks in the test city: task-000 has a solution, and task-001 has none."""
    no_solution = Task("task-001", "hard", copy.deepcopy(city), None)
    del city.orders["o4"]
    solvable = Task("task-000", "medium", city, Solution(["o1", "o2", "o3"], 90))
    return [solvable, no_solution]


def submit(solvable: bool, order_ids: list[str], **tokens):
    """A response that submits an answer."""
    call = tool_use("submit_answer", {"solvable": solvable, "order_ids": order_ids})
    return response(call, **tokens)


def answers(count: int) -> list:
    """Right answers to the two tasks, for count runs of each, alternating between them."""
    return [submit(True, ["o2", "o1", "o3"]), submit(False, [])] * count


# choose_tasks


def test_all_tasks_sorted_by_id():
    saved = load_tasks(SAVED_TASKS)
    chosen = choose_tasks(list(reversed(saved)), None, 0)
    assert [task.task_id for task in chosen] == [task.task_id for task in saved]


def test_the_same_seed_gives_the_same_sample():
    saved = load_tasks(SAVED_TASKS)
    first = [task.task_id for task in choose_tasks(saved, 20, 0)]
    again = [task.task_id for task in choose_tasks(saved, 20, 0)]
    assert first == again
    assert len(set(first)) == 20
    assert first == sorted(first)


def test_another_seed_gives_another_sample():
    saved = load_tasks(SAVED_TASKS)
    seed_0 = [task.task_id for task in choose_tasks(saved, 20, 0)]
    seed_1 = [task.task_id for task in choose_tasks(saved, 20, 1)]
    assert seed_0 != seed_1


# run_tasks


def test_logs_every_run(tasks, tmp_path, capsys):
    log = tmp_path / "runs" / "log.jsonl"
    records = run_tasks(FakeClient(answers(1)), tasks, CONFIG, 1, 1.0, log, workers=1)
    assert load_records(log) == records
    assert [(record["task_id"], record["repeat"]) for record in records] == [
        ("task-000", 0),
        ("task-001", 0),
    ]
    assert [record["verdict"] for record in records] == ["correct", "correct"]
    printed = capsys.readouterr().out
    assert "2 runs to do, 0 already in" in printed
    assert "   1/2  task-000  repeat 0  medium  correct" in printed
    assert "   2/2  task-001  repeat 0  hard    correct" in printed


def test_repeats_come_after_every_first_run(tasks, tmp_path):
    log = tmp_path / "log.jsonl"
    records = run_tasks(FakeClient(answers(3)), tasks, CONFIG, 3, 1.0, log, workers=1)
    assert [(record["task_id"], record["repeat"]) for record in records] == [
        ("task-000", 0),
        ("task-001", 0),
        ("task-000", 1),
        ("task-001", 1),
        ("task-000", 2),
        ("task-001", 2),
    ]


def test_picks_up_where_the_log_left_off(tasks, tmp_path, capsys):
    log = tmp_path / "log.jsonl"
    run_tasks(FakeClient(answers(1)[:1]), tasks[:1], CONFIG, 1, 1.0, log, workers=1)
    client = FakeClient(answers(2)[1:])
    records = run_tasks(client, tasks, CONFIG, 2, 1.0, log, workers=1)
    assert len(client.messages.requests) == 3
    assert [(record["task_id"], record["repeat"]) for record in records] == [
        ("task-000", 0),
        ("task-001", 0),
        ("task-000", 1),
        ("task-001", 1),
    ]
    assert "3 runs to do, 1 already in" in capsys.readouterr().out


def test_nothing_to_do_costs_nothing(tasks, tmp_path):
    log = tmp_path / "log.jsonl"
    run_tasks(FakeClient(answers(1)), tasks, CONFIG, 1, 1.0, log, workers=1)
    client = FakeClient([])
    run_tasks(client, tasks, CONFIG, 1, 1.0, log, workers=1)
    assert client.messages.requests == []


def test_runs_in_parallel_batches(tasks, tmp_path):
    log = tmp_path / "log.jsonl"
    records = run_tasks(FakeClient(answers(2)), tasks, CONFIG, 2, 1.0, log, workers=3)
    assert sorted((record["task_id"], record["repeat"]) for record in records) == [
        ("task-000", 0),
        ("task-000", 1),
        ("task-001", 0),
        ("task-001", 1),
    ]


def test_stops_at_the_budget(tasks, tmp_path, capsys):
    """Each run costs 1000 * 0.50 / 1e6 = $0.0005 in output tokens, and 100 input tokens."""
    log = tmp_path / "log.jsonl"
    costly = [submit(True, ["o2", "o1", "o3"], output_tokens=1000), submit(False, [])] * 2
    client = FakeClient(costly)
    records = run_tasks(client, tasks, CONFIG, 2, 0.0004, log, workers=1)
    assert len(records) == 1
    assert "stopped: the log's runs cost $0.0005, the budget is $0.00" in capsys.readouterr().out


def test_the_budget_counts_earlier_runs(tasks, tmp_path):
    log = tmp_path / "log.jsonl"
    append_record(log, {"task_id": "task-009", "repeat": 0, "cost_usd": 5.0, **config_fields()})
    client = FakeClient(answers(1))
    records = run_tasks(client, tasks, CONFIG, 1, 5.0, log, workers=1)
    assert client.messages.requests == []
    assert len(records) == 1


def test_an_error_logs_the_rest_of_the_batch_first(tasks, tmp_path):
    class Overloaded(Exception):
        pass

    class FailsOnTaskOne(FakeClient):
        def __init__(self):
            super().__init__([])
            self.messages.create = self.create

        def create(self, **request):
            if "o4" in request["messages"][0]["content"]:
                raise Overloaded("still overloaded")
            return submit(True, ["o2", "o1", "o3"])

    log = tmp_path / "log.jsonl"
    with pytest.raises(Overloaded):
        run_tasks(FailsOnTaskOne(), list(reversed(tasks)), CONFIG, 1, 1.0, log, workers=2)
    assert [record["task_id"] for record in load_records(log)] == ["task-000"]


def config_fields(config: AgentConfig = CONFIG) -> dict:
    return {"model": config.model, "effort": config.effort, "tool_specs": config.tool_specs}


def test_refuses_a_log_of_another_configuration(tasks, tmp_path):
    log = tmp_path / "log.jsonl"
    other = AgentConfig("claude-sonnet-5-5", "low")
    append_record(
        log, {"task_id": "task-000", "repeat": 0, "cost_usd": 0.1, **config_fields(other)}
    )
    client = FakeClient(answers(1))
    with pytest.raises(ValueError, match="Use another log file"):
        run_tasks(client, tasks, CONFIG, 1, 1.0, log, workers=1)
    assert client.messages.requests == []


# check_log_matches


def test_a_log_of_this_configuration_matches():
    check_log_matches([config_fields(), config_fields()], CONFIG)


def test_other_tool_descriptions_do_not_match():
    improved = AgentConfig("claude-haiku-5-5", "medium", tool_specs="improved")
    with pytest.raises(ValueError):
        check_log_matches([config_fields(improved)], CONFIG)


# cost_without_cache and show_summary


def a_record(difficulty: str, verdict: str, outcome: str = "answered") -> dict:
    """A haiku run that read 9000 tokens from the cache and wrote 1000 to it."""
    return {
        "difficulty": difficulty,
        "verdict": verdict,
        "outcome": outcome,
        "model": "claude-haiku-5-5",
        "input_tokens": 2000,
        "output_tokens": 1000,
        "cache_write_tokens": 1000,
        "cache_read_tokens": 9000,
        "cost_usd": 0.000915,
    }


def test_cost_without_cache_reads_every_input_token_at_full_price():
    assert cost_without_cache(a_record("easy", "correct")) == pytest.approx(0.0017)


def test_summary(capsys):
    records = [
        a_record("easy", "correct"),
        a_record("hard", "correct"),
        a_record("hard", "not_earliest"),
        a_record("hard", "no_answer", outcome="max_tokens"),
    ]
    show_summary(records)
    printed = capsys.readouterr().out
    assert "\neasy         1        1   100%" in printed
    assert "\nhard         3        1    33%" in printed
    assert "\nall          4        2    50%" in printed
    assert "\nnot_earliest     1\n" in printed
    assert "\nended by     answered 3, max_tokens 1\n" in printed
    assert "\nper run      $0.0009 on average\n" in printed
    assert "\ncache saved  $0.0031, 46% of the cost without it\n" in printed


def test_summary_of_an_empty_log(capsys):
    show_summary([])
    assert capsys.readouterr().out == "          runs  correct  share      cost  per correct\n"


def test_summary_without_cache_tokens(capsys):
    record = a_record("easy", "correct")
    record.update(cache_write_tokens=0, cache_read_tokens=0, input_tokens=0, output_tokens=0)
    record["cost_usd"] = 0.0
    show_summary([record])
    assert "cache saved" not in capsys.readouterr().out


def test_the_summary_reads_what_run_tasks_logged(tasks, tmp_path, capsys):
    log = tmp_path / "log.jsonl"
    stop = response(text("I can't tell."), stop_reason="end_turn")
    run_tasks(FakeClient([submit(False, []), stop]), tasks, CONFIG, 1, 1.0, log, workers=1)
    show_summary(load_records(log))
    printed = capsys.readouterr().out
    assert "\nall          2        0     0%" in printed
    assert "\nno_answer        1\n" in printed
    assert "\nmissed_solution  1\n" in printed
