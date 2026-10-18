import pandas as pd
import pytest

from agent_eval.analysis.tables import load_runs, task_totals
from agent_eval.experiment.run_log import append_record

COLUMNS = ["config", "task_id", "difficulty", "repeat", "verdict", "cost_usd"]


@pytest.fixture
def runs() -> pd.DataFrame:
    """Two configurations, two tasks and two repeats, in no particular order."""
    rows = [
        ("sonnet-low", "task-001", "hard", 0, "not_earliest", 0.03),
        ("haiku-high", "task-001", "hard", 0, "correct", 0.002),
        ("haiku-high", "task-000", "easy", 0, "correct", 0.001),
        ("sonnet-low", "task-000", "easy", 1, "correct", 0.01),
        ("haiku-high", "task-001", "hard", 1, "no_answer", 0.004),
        ("sonnet-low", "task-000", "easy", 0, "correct", 0.02),
        ("haiku-high", "task-000", "easy", 1, "correct", 0.001),
        ("sonnet-low", "task-001", "hard", 1, "correct", 0.04),
    ]
    return pd.DataFrame(rows, columns=COLUMNS)


def test_load_runs_puts_every_log_in_one_table(tmp_path):
    for config, repeats in [("haiku-high", 2), ("sonnet-low", 1)]:
        for repeat in range(repeats):
            record = {"task_id": "task-000", "repeat": repeat, "verdict": "correct"}
            append_record(tmp_path / f"{config}.jsonl", record)

    runs = load_runs([tmp_path / "haiku-high.jsonl", tmp_path / "sonnet-low.jsonl"])

    assert runs["config"].tolist() == ["haiku-high", "haiku-high", "sonnet-low"]
    assert runs["repeat"].tolist() == [0, 1, 0]
    assert runs.index.tolist() == [0, 1, 2]


def test_task_totals_columns(runs):
    totals = task_totals(runs)
    assert totals.columns.tolist() == [
        "config",
        "task_id",
        "difficulty",
        "runs",
        "correct",
        "cost_usd",
    ]


def test_task_totals_has_a_row_per_config_and_task_in_order(runs):
    totals = task_totals(runs)
    assert list(zip(totals["config"], totals["task_id"])) == [
        ("haiku-high", "task-000"),
        ("haiku-high", "task-001"),
        ("sonnet-low", "task-000"),
        ("sonnet-low", "task-001"),
    ]
    assert totals.index.tolist() == [0, 1, 2, 3]


def test_task_totals_keeps_the_difficulty(runs):
    assert task_totals(runs)["difficulty"].tolist() == ["easy", "hard", "easy", "hard"]


def test_task_totals_counts_runs_and_correct_answers(runs):
    totals = task_totals(runs)
    assert totals["runs"].tolist() == [2, 2, 2, 2]
    assert totals["correct"].tolist() == [2, 1, 2, 1]


def test_task_totals_adds_up_the_cost(runs):
    assert task_totals(runs)["cost_usd"].tolist() == pytest.approx([0.002, 0.006, 0.03, 0.07])


def test_task_totals_counts_a_task_with_more_runs(runs):
    extra = pd.DataFrame([("haiku-high", "task-000", "easy", 2, "not_earliest", 0.003)])
    totals = task_totals(pd.concat([runs, extra.set_axis(COLUMNS, axis=1)], ignore_index=True))
    assert totals["runs"].tolist() == [3, 2, 2, 2]
    assert totals["correct"].tolist() == [2, 1, 2, 1]
