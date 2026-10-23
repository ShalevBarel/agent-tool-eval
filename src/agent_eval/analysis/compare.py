"""Paired comparisons of two configurations: a test, its correction for many tests, and power."""

import math
from statistics import NormalDist

import numpy as np

# Means of thirds are not exact in floating point, so two sign flips with the
# same mean can differ in the last digits. Means this close count as equal.
TIE_TOLERANCE = 1e-9


def sign_flip_test(differences: np.ndarray, n_flips: int, rng: np.random.Generator) -> float:
    """Return the two-sided p-value of a paired sign-flip permutation test.

    differences holds one difference per task between two configurations,
    such as a's share of correct runs minus b's. If the two are equally
    good, every difference is as likely to be negative as positive. The
    test flips the sign of each difference at random n_flips times and
    returns the share of flips whose mean is at least as far from 0 as the
    observed mean, counting the observed differences as one more flip:
    (1 + extreme flips) / (1 + n_flips).
    """
    observed = abs(differences.mean())
    signs = rng.choice(np.array([-1, 1]), size=(n_flips, len(differences)))
    flipped = np.abs((signs * differences).mean(axis=1))
    extreme = np.sum(flipped >= observed - TIE_TOLERANCE)
    return float((1 + extreme) / (1 + n_flips))


def mcnemar_exact(only_a: int, only_b: int) -> float:
    """Return the two-sided p-value of McNemar's exact test.

    With one run per task, only_a counts the tasks where only configuration
    a is correct, and only_b those where only b is. If the two are equally
    good, each of these tasks is equally likely to go either way, so only_a
    is binomial with n = only_a + only_b and p = 1/2. The p-value is twice
    the chance of a count at most min(only_a, only_b), and at most 1.
    """
    n = only_a + only_b
    tail = sum(math.comb(n, k) for k in range(min(only_a, only_b) + 1)) / 2**n
    return min(1.0, 2 * tail)


def holm(p_values: list[float]) -> list[float]:
    """Return the p-values adjusted by Holm's method, in the order given.

    The i-th smallest of m p-values is multiplied by m - i + 1 (counting i
    from 1), then raised to the largest adjusted value before it, so the
    order stays the same, and capped at 1. Rejecting every hypothesis whose
    adjusted p-value is at most alpha keeps the chance of any false
    rejection at most alpha.
    """
    m = len(p_values)
    order = sorted(range(m), key=lambda i: p_values[i])
    adjusted = [0.0] * m
    largest = 0.0
    for rank, i in enumerate(order):
        largest = max(largest, (m - rank) * p_values[i])
        adjusted[i] = min(1.0, largest)
    return adjusted


def tasks_needed(sd: float, difference: float, alpha: float = 0.05, power: float = 0.8) -> int:
    """Return how many tasks a paired comparison needs to detect a difference.

    sd is the standard deviation of the per-task differences. The answer
    uses the normal approximation for a two-sided test at level alpha:
    ((z of 1 - alpha/2 + z of power) * sd / difference) squared, rounded up.
    """
    z = NormalDist().inv_cdf
    return math.ceil(((z(1 - alpha / 2) + z(power)) * sd / difference) ** 2)
