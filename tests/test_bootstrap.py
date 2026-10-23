import numpy as np
import pytest

from agent_eval.analysis.bootstrap import (
    Estimate,
    percentile_estimate,
    ratio_estimate,
    resample_indices,
    resampled_ratios,
)

GROUPS = np.array(["hard", "easy", "hard", "easy", "medium", "hard"])


def test_resample_indices_has_a_row_per_resample_and_a_column_per_position():
    indices = resample_indices(GROUPS, 7, np.random.default_rng(0))
    assert indices.shape == (7, 6)
    assert np.issubdtype(indices.dtype, np.integer)


def test_resample_indices_keeps_the_size_of_every_group():
    indices = resample_indices(GROUPS, 200, np.random.default_rng(0))
    for row in indices:
        assert sorted(GROUPS[row]) == sorted(GROUPS)


def test_resample_indices_draws_with_replacement_from_every_position():
    indices = resample_indices(GROUPS, 200, np.random.default_rng(0))
    assert any(len(set(row)) < len(row) for row in indices)
    assert set(indices.ravel()) == set(range(len(GROUPS)))


def test_resample_indices_draws_the_positions_of_a_group_equally_often():
    groups = np.zeros(4, dtype=int)
    indices = resample_indices(groups, 10_000, np.random.default_rng(0))
    shares = np.bincount(indices.ravel(), minlength=4) / indices.size
    assert shares == pytest.approx([0.25, 0.25, 0.25, 0.25], abs=0.01)


def test_resample_indices_repeats_with_the_same_seed():
    first = resample_indices(GROUPS, 50, np.random.default_rng(3))
    second = resample_indices(GROUPS, 50, np.random.default_rng(3))
    assert np.array_equal(first, second)


def test_ratio_estimate_value_is_the_ratio_of_the_sums():
    correct = np.array([3, 2, 0])
    runs = np.array([3, 3, 3])
    estimate = ratio_estimate(correct, runs, np.array([[0, 1, 2]]))
    assert estimate.value == pytest.approx(5 / 9)


def test_ratio_estimate_works_for_cost_per_correct_answer():
    cost = np.array([0.03, 0.01, 0.02])
    correct = np.array([3, 1, 0])
    estimate = ratio_estimate(cost, correct, np.array([[0, 1, 2]]))
    assert estimate.value == pytest.approx(0.015)


def test_ratio_estimate_interval_from_hand_made_resamples():
    correct = np.array([3, 2, 0])
    runs = np.array([3, 3, 3])
    # The resamples' ratios are 1, 5/9, 0, 2/3 and 2/3.
    indices = np.array([[0, 0, 0], [0, 1, 2], [2, 2, 2], [1, 1, 1], [0, 0, 2]])
    estimate = ratio_estimate(correct, runs, indices, level=0.5)
    assert estimate.low == pytest.approx(5 / 9)
    assert estimate.high == pytest.approx(2 / 3)


def test_ratio_estimate_default_level_is_95_percent():
    values = np.arange(101.0)
    # Each resample is a single position, so the ratios are 0, 1, ..., 100.
    indices = np.arange(101).reshape(101, 1)
    estimate = ratio_estimate(values, np.ones(101), indices)
    assert estimate.value == pytest.approx(50)
    assert (estimate.low, estimate.high) == pytest.approx((2.5, 97.5))


def test_ratio_estimate_wider_interval_at_a_higher_level():
    rng = np.random.default_rng(0)
    correct = rng.integers(0, 4, size=30)
    runs = np.full(30, 3)
    indices = resample_indices(np.zeros(30, dtype=int), 1000, rng)
    narrow = ratio_estimate(correct, runs, indices, level=0.5)
    wide = ratio_estimate(correct, runs, indices, level=0.99)
    assert wide.low < narrow.low <= narrow.value <= narrow.high < wide.high


def test_resampled_ratios_one_ratio_per_resample():
    correct = np.array([3, 2, 0])
    runs = np.array([3, 3, 3])
    indices = np.array([[0, 0, 0], [0, 1, 2], [2, 2, 2], [1, 1, 1], [0, 0, 2]])
    ratios = resampled_ratios(correct, runs, indices)
    assert ratios == pytest.approx([1, 5 / 9, 0, 2 / 3, 2 / 3])


def test_resampled_ratios_of_cost_over_correct_answers():
    cost = np.array([0.03, 0.01, 0.02])
    correct = np.array([3, 1, 0])
    ratios = resampled_ratios(cost, correct, np.array([[0, 1, 2], [1, 2, 2]]))
    assert ratios == pytest.approx([0.06 / 4, 0.05 / 1])


def test_percentile_estimate_keeps_the_value():
    estimate = percentile_estimate(np.float64(0.25), np.array([0.1, 0.2, 0.3]))
    assert estimate == Estimate(0.25, pytest.approx(0.105), pytest.approx(0.295))
    assert type(estimate.value) is float
    assert type(estimate.low) is float and type(estimate.high) is float


def test_percentile_estimate_takes_the_middle_of_the_resampled_values():
    resampled = np.arange(101.0)
    estimate = percentile_estimate(50.0, resampled)
    assert (estimate.low, estimate.high) == pytest.approx((2.5, 97.5))
    narrow = percentile_estimate(50.0, resampled, level=0.5)
    assert (narrow.low, narrow.high) == pytest.approx((25, 75))


def test_percentile_estimate_of_a_paired_difference():
    # Two configurations on the same resamples: the interval is of the differences.
    first = np.array([0.9, 0.8, 1.0, 0.7, 0.9])
    second = np.array([0.6, 0.6, 0.7, 0.6, 0.5])
    estimate = percentile_estimate(0.25, first - second, level=0.5)
    assert (estimate.low, estimate.high) == pytest.approx((0.2, 0.3))
