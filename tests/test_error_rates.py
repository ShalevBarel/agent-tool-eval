"""Does the paired test keep its promises? Simulated comparisons whose truth is known.

With no real difference, a test at level 0.05 should find one in at most
about 5% of experiments. With the difference that tasks_needed was asked
about, and that many tasks, it should find it in about 80%.
"""

import numpy as np

from agent_eval.analysis.compare import sign_flip_test, tasks_needed
from agent_eval.analysis.simulate import simulate_correct, task_probabilities

ALPHA = 0.05
EXPERIMENTS = 400
FLIPS = 1000


def false_positive_rate(seed: int = 0) -> float:
    """Return how often the test finds a difference between two equally good configurations.

    The simulated tasks look like the real ones: 50 per difficulty, and 3
    repeats that mostly agree. Both configurations have the same chance on
    every task.
    """
    rng = np.random.default_rng(seed)
    rejections = 0
    for _ in range(EXPERIMENTS):
        probabilities = np.concatenate(
            [task_probabilities(mean, 1.0, 50, rng) for mean in (0.95, 0.85, 0.6)]
        )
        first = simulate_correct(probabilities, 3, rng).mean(axis=1)
        second = simulate_correct(probabilities, 3, rng).mean(axis=1)
        rejections += sign_flip_test(first - second, FLIPS, rng) <= ALPHA
    return rejections / EXPERIMENTS


def power_at_tasks_needed(seed: int = 0) -> float:
    """Return how often the test finds a true difference of 0.07 with tasks_needed tasks.

    On each task, a beats b with chance 0.12 and loses with chance 0.05,
    which needs 265 tasks for power 0.8.
    """
    rng = np.random.default_rng(seed)
    chances = [0.12, 0.83, 0.05]
    values = np.array([1.0, 0.0, -1.0])
    difference = values @ chances
    sd = np.sqrt(values**2 @ chances - difference**2)
    n_tasks = tasks_needed(sd, difference, ALPHA, 0.8)
    rejections = 0
    for _ in range(EXPERIMENTS):
        differences = rng.choice(values, size=n_tasks, p=chances)
        rejections += sign_flip_test(differences, FLIPS, rng) <= ALPHA
    return rejections / EXPERIMENTS


def test_no_real_difference_is_found_about_5_percent_of_the_time():
    # Ties make the test a little cautious, so the share is usually a bit under 5%.
    assert 0.01 <= false_positive_rate() <= 0.08


def test_tasks_needed_gives_the_test_about_80_percent_power():
    assert 0.70 <= power_at_tasks_needed() <= 0.88
