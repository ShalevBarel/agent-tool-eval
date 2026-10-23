"""Bootstrap confidence intervals that resample tasks, not runs."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Estimate:
    """A point estimate and its confidence interval.

    Attributes:
        value: The estimate on the data itself.
        low: The lower end of the interval.
        high: The upper end of the interval.
    """

    value: float
    low: float
    high: float


def resample_indices(groups: np.ndarray, n_resamples: int, rng: np.random.Generator) -> np.ndarray:
    """Return bootstrap resamples of positions, drawn with replacement within each group.

    groups holds one label per position, such as each task's difficulty.
    Every resample keeps the size of every group: a group with 50 positions
    gets 50 draws, all from its own positions.

    Returns:
        An integer array with one row per resample and one column per
        position, ordered group by group.
    """
    columns = []
    for group in np.unique(groups):
        members = np.flatnonzero(groups == group)
        columns.append(rng.choice(members, size=(n_resamples, len(members))))
    return np.concatenate(columns, axis=1)


def resampled_ratios(
    numerators: np.ndarray, denominators: np.ndarray, indices: np.ndarray
) -> np.ndarray:
    """Return the ratio of the sums in every resample, one per row of indices.

    Each row of indices names the positions of one resample, and its ratio
    is the sum of their numerators over the sum of their denominators.
    """
    raise NotImplementedError


def percentile_estimate(value: float, resampled: np.ndarray, level: float = 0.95) -> Estimate:
    """Return value with a percentile interval: the middle level of its resampled values.

    resampled holds the same statistic computed on every resample, such as
    one success rate per resample. At level 0.95 the interval runs from
    their 2.5th percentile to their 97.5th.
    """
    raise NotImplementedError


def ratio_estimate(
    numerators: np.ndarray, denominators: np.ndarray, indices: np.ndarray, level: float = 0.95
) -> Estimate:
    """Return the ratio of two sums, with a percentile bootstrap interval.

    The value is sum(numerators) / sum(denominators). Each row of indices
    is one resample: the ratio of the sums over the positions it names.
    The interval holds the middle level of these ratios, such as the middle
    95%. With correct answers over runs, this is the success rate; with cost
    over correct answers, the cost per correct answer.
    """
    value = numerators.sum() / denominators.sum()
    ratios = numerators[indices].sum(axis=1) / denominators[indices].sum(axis=1)
    tail = 100 * (1 - level) / 2
    low, high = np.percentile(ratios, [tail, 100 - tail])
    return Estimate(float(value), float(low), float(high))
