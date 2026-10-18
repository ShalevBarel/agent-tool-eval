"""Do the intervals keep their promise? Simulated runs whose true success rate is known.

A 95% interval should hold the true rate in about 95% of experiments. The
simulated experiments look like the real one: 50 tasks per difficulty, 3
repeats, and the runs of one task agree about 80% of the time.
"""

from statistics import mean

import numpy as np

from agent_eval.analysis.bootstrap import ratio_estimate, resample_indices
from agent_eval.analysis.simulate import simulate_correct, task_probabilities

MEANS = {"easy": 0.95, "medium": 0.85, "hard": 0.6}
CONCENTRATION = 1.0
TASKS_PER_DIFFICULTY = 50
REPEATS = 3
EXPERIMENTS = 500
RESAMPLES = 500


def coverage(unit: str, seed: int = 0) -> float:
    """Return the share of simulated experiments whose interval holds the true success rate.

    unit is what the bootstrap resamples within each difficulty: "task"
    (all the runs of a task go together) or "run" (each run on its own).
    """
    rng = np.random.default_rng(seed)
    truth = mean(MEANS.values())
    difficulties = np.repeat(list(MEANS), TASKS_PER_DIFFICULTY)
    hits = 0
    for _ in range(EXPERIMENTS):
        probabilities = np.concatenate(
            [
                task_probabilities(task_mean, CONCENTRATION, TASKS_PER_DIFFICULTY, rng)
                for task_mean in MEANS.values()
            ]
        )
        correct = simulate_correct(probabilities, REPEATS, rng)
        if unit == "task":
            numerators = correct.sum(axis=1)
            denominators = np.full(len(correct), REPEATS)
            groups = difficulties
        else:
            numerators = correct.ravel()
            denominators = np.ones(correct.size)
            groups = np.repeat(difficulties, REPEATS)
        indices = resample_indices(groups, RESAMPLES, rng)
        estimate = ratio_estimate(numerators, denominators, indices)
        hits += estimate.low <= truth <= estimate.high
    return hits / EXPERIMENTS


def test_resampling_tasks_holds_the_truth_about_95_percent_of_the_time():
    assert 0.90 <= coverage("task") <= 0.98


def test_resampling_runs_holds_it_less_often():
    assert coverage("run") < 0.88
