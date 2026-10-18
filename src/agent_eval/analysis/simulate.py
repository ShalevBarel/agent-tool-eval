"""Simulated run outcomes whose true success rate is known, to check the intervals."""

import numpy as np


def task_probabilities(
    mean: float, concentration: float, size: int, rng: np.random.Generator
) -> np.ndarray:
    """Return random chances of success, one per task, from a beta distribution.

    The beta distribution has the given mean. A low concentration spreads
    the tasks out toward chances near 0 and near 1, so tasks differ a lot,
    and the runs of one task tend to agree.
    """
    return rng.beta(mean * concentration, (1 - mean) * concentration, size=size)


def simulate_correct(
    probabilities: np.ndarray, repeats: int, rng: np.random.Generator
) -> np.ndarray:
    """Return simulated outcomes, 1 for a correct run and 0 otherwise.

    The result has one row per task and one column per repeat. Each run of
    task i is correct with chance probabilities[i], independently of the
    other runs.
    """
    return (rng.random((len(probabilities), repeats)) < probabilities[:, np.newaxis]).astype(int)
