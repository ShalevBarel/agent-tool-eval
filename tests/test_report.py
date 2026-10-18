import pandas as pd
import pytest

from agent_eval.analysis.report import estimates, report

DIFFICULTIES = {"task-000": "easy", "task-001": "easy", "task-002": "hard", "task-003": "hard"}

# 1 is a correct run, 0 a wrong one: "good" is right in 7 of 8 runs, "bad" in 3.
OUTCOMES = {
    "good": {"task-000": [1, 1], "task-001": [1, 1], "task-002": [1, 0], "task-003": [1, 1]},
    "bad": {"task-000": [1, 1], "task-001": [1, 0], "task-002": [0, 0], "task-003": [0, 0]},
}


@pytest.fixture
def runs() -> pd.DataFrame:
    """Two configurations on two easy and two hard tasks, two repeats each, at $0.01 a run."""
    rows = []
    for config, by_task in OUTCOMES.items():
        for task_id, outcomes in by_task.items():
            for repeat, outcome in enumerate(outcomes):
                verdict = "correct" if outcome else "not_earliest"
                rows.append((config, task_id, DIFFICULTIES[task_id], repeat, verdict, 0.01))
    columns = ["config", "task_id", "difficulty", "repeat", "verdict", "cost_usd"]
    return pd.DataFrame(rows, columns=columns)


def test_estimates_success_rate_and_cost_per_correct_answer(runs):
    results = estimates(runs, n_resamples=200)
    assert results["good"]["success"].value == pytest.approx(7 / 8)
    assert results["bad"]["success"].value == pytest.approx(3 / 8)
    assert results["good"]["cost_per_correct"].value == pytest.approx(0.08 / 7)
    assert results["bad"]["cost_per_correct"].value == pytest.approx(0.08 / 3)


def test_estimates_intervals_hold_the_values(runs):
    for result in estimates(runs, n_resamples=200).values():
        for estimate in result.values():
            assert estimate.low <= estimate.value <= estimate.high


def test_estimates_put_the_best_first(runs):
    assert list(estimates(runs, n_resamples=200)) == ["good", "bad"]


def test_estimates_resample_within_each_difficulty(runs):
    # Every resample has two easy tasks, always right for "good", and two hard
    # ones, so it never gets below 6 of 8.
    assert estimates(runs, n_resamples=200)["good"]["success"].low >= 6 / 8


def test_estimates_need_the_same_tasks_for_every_configuration(runs):
    fewer = runs[~((runs["config"] == "bad") & (runs["task_id"] == "task-003"))]
    with pytest.raises(ValueError, match="ran different tasks"):
        estimates(fewer, n_resamples=10)


def test_report_lists_every_configuration_best_first(runs):
    lines = report(runs, n_resamples=200).splitlines()
    assert lines[0].split()[:3] == ["config", "runs", "success"]
    assert lines[1].split()[:3] == ["good", "8", "87.5%"]
    assert lines[2].split()[:3] == ["bad", "8", "37.5%"]


def test_report_shows_success_by_difficulty(runs):
    lines = report(runs, n_resamples=200).splitlines()
    start = lines.index(next(line for line in lines if line.startswith("success by difficulty")))
    assert lines[start].split()[-2:] == ["easy", "hard"]
    assert lines[start + 1].split() == ["good", "100%", "75%"]
    assert lines[start + 2].split() == ["bad", "75%", "0%"]


def test_report_repeats_with_the_same_seed(runs):
    assert report(runs, n_resamples=200, seed=1) == report(runs, n_resamples=200, seed=1)
