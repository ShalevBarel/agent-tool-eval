import numpy as np
import pandas as pd
import pytest

from agent_eval.analysis.compare import holm, tasks_needed
from agent_eval.analysis.report import (
    COMPARISONS,
    comparisons,
    estimates,
    estimates_by_difficulty,
    report,
)

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


def test_estimates_by_difficulty_success_rates(runs):
    results = estimates_by_difficulty(runs, n_resamples=200)
    assert results["good"]["easy"].value == pytest.approx(4 / 4)
    assert results["good"]["hard"].value == pytest.approx(3 / 4)
    assert results["bad"]["easy"].value == pytest.approx(3 / 4)
    assert results["bad"]["hard"].value == pytest.approx(0 / 4)


def test_estimates_by_difficulty_intervals_hold_the_values(runs):
    for by_level in estimates_by_difficulty(runs, n_resamples=200).values():
        for estimate in by_level.values():
            assert estimate.low <= estimate.value <= estimate.high


def test_estimates_by_difficulty_resample_only_the_tasks_of_the_difficulty(runs):
    # "good" is always right on the easy tasks, so every resample of them is too.
    easy = estimates_by_difficulty(runs, n_resamples=200)["good"]["easy"]
    assert (easy.low, easy.high) == (1, 1)


def test_estimates_by_difficulty_order(runs):
    results = estimates_by_difficulty(runs, n_resamples=200)
    assert list(results) == ["bad", "good"]
    assert list(results["good"]) == ["easy", "hard"]


def test_estimates_by_difficulty_need_the_same_tasks_for_every_configuration(runs):
    fewer = runs[~((runs["config"] == "bad") & (runs["task_id"] == "task-003"))]
    with pytest.raises(ValueError, match="ran different tasks"):
        estimates_by_difficulty(fewer, n_resamples=10)


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


def test_comparisons_success_difference_and_cost_ratio(runs):
    [comparison] = comparisons(runs, [("good", "bad")], n_resamples=200)
    assert (comparison.a, comparison.b) == ("good", "bad")
    assert comparison.success_difference.value == pytest.approx(7 / 8 - 3 / 8)
    assert comparison.cost_ratio.value == pytest.approx((0.08 / 7) / (0.08 / 3))


def test_comparisons_intervals_hold_the_values(runs):
    [comparison] = comparisons(runs, [("good", "bad")], n_resamples=200)
    for estimate in (comparison.success_difference, comparison.cost_ratio):
        assert estimate.low <= estimate.value <= estimate.high


def test_comparisons_test_the_per_task_differences(runs):
    # Per task, "good" minus "bad" is 0, 1/2, 1/2 and 1: three positive
    # differences, so p is 2/8, as in the sign-flip test's own tests.
    [comparison] = comparisons(runs, [("good", "bad")], n_resamples=200)
    assert comparison.p_value == pytest.approx(0.25, abs=0.03)
    assert comparison.holm_p_value == comparison.p_value
    assert comparison.tasks_needed == tasks_needed(np.std([0, 0.5, 0.5, 1], ddof=1), 0.05)


def test_comparisons_adjust_for_every_comparison(runs):
    found = comparisons(runs, [("good", "bad"), ("bad", "good")], n_resamples=200)
    assert [comparison.a for comparison in found] == ["good", "bad"]
    assert found[1].success_difference.value == pytest.approx(-0.5)
    p_values = [comparison.p_value for comparison in found]
    assert [comparison.holm_p_value for comparison in found] == holm(p_values)


def test_comparisons_leave_out_configurations_not_in_the_runs(runs):
    found = comparisons(runs, [("good", "missing"), ("good", "bad")], n_resamples=200)
    assert [(comparison.a, comparison.b) for comparison in found] == [("good", "bad")]
    assert comparisons(runs, n_resamples=10) == []


def test_comparisons_planned_by_default():
    assert COMPARISONS[0] == ("sonnet-low", "haiku-high")
    assert len(COMPARISONS) == len(set(COMPARISONS)) == 5


def test_report_shows_the_comparisons(runs):
    lines = report(runs, n_resamples=200, pairs=[("good", "bad")]).splitlines()
    row = next(line for line in lines if line.startswith("good vs bad"))
    assert row.split()[3:5] == ["+50.0", "pts"]
    cost_row = [line for line in lines if line.startswith("good vs bad")][1]
    assert cost_row.split()[3] == "0.43x"


def test_report_without_comparisons(runs):
    assert "a vs b" not in report(runs, n_resamples=200)
