import math

import numpy as np
import pytest

from agent_eval.analysis.compare import holm, mcnemar_exact, sign_flip_test, tasks_needed


def test_sign_flip_test_returns_a_float_between_0_and_1():
    differences = np.array([1.0, -1 / 3, 2 / 3, 0.0, 1 / 3])
    p_value = sign_flip_test(differences, 1000, np.random.default_rng(0))
    assert isinstance(p_value, float)
    assert 0 < p_value <= 1


def test_sign_flip_test_counts_the_observed_differences_as_one_more_flip():
    # 20 positive differences: a random flip is almost never as extreme, so the
    # p-value is (1 + 0) / (1 + 99).
    p_value = sign_flip_test(np.ones(20), 99, np.random.default_rng(0))
    assert p_value == pytest.approx(0.01)


def test_sign_flip_test_matches_the_exact_p_value():
    # 3 positive differences and one 0: of the 8 sign patterns of the three, only
    # all-positive and all-negative are as far from 0, so p = 2/8.
    differences = np.array([1.0, 1.0, 1.0, 0.0])
    p_value = sign_flip_test(differences, 20_000, np.random.default_rng(0))
    assert p_value == pytest.approx(0.25, abs=0.02)


def test_sign_flip_test_of_ten_positive_differences():
    # Only 2 of the 1024 sign patterns are as far from 0.
    p_value = sign_flip_test(np.full(10, 2 / 3), 100_000, np.random.default_rng(0))
    assert p_value == pytest.approx(2 / 1024, abs=0.0005)


def test_sign_flip_test_is_1_when_nothing_differs():
    assert sign_flip_test(np.zeros(30), 500, np.random.default_rng(0)) == 1.0


def test_sign_flip_test_is_1_when_the_differences_cancel_out():
    assert sign_flip_test(np.array([1.0, -1.0, 1.0, -1.0]), 500, np.random.default_rng(0)) == 1.0


def test_sign_flip_test_counts_flips_with_the_same_mean_despite_rounding():
    # The mean is 0, but in floating point it comes out about -2e-17, and some
    # flips come out exactly 0. Every flip is at least as far from 0.
    differences = np.array([2, 3, -2, -1, -3, 1, -1, 1, -1, 2, 1, -2]) / 3
    assert differences.mean() != 0
    assert sign_flip_test(differences, 2000, np.random.default_rng(0)) == 1.0


def test_sign_flip_test_is_two_sided():
    differences = np.array([1.0, 2 / 3, 1.0, -1 / 3, 1 / 3, 0.0, 1.0])
    forward = sign_flip_test(differences, 1000, np.random.default_rng(5))
    backward = sign_flip_test(-differences, 1000, np.random.default_rng(5))
    assert forward == backward


def test_sign_flip_test_repeats_with_the_same_seed():
    differences = np.array([1.0, 2 / 3, -1 / 3, 1 / 3, 0.0, 1.0])
    first = sign_flip_test(differences, 300, np.random.default_rng(2))
    second = sign_flip_test(differences, 300, np.random.default_rng(2))
    assert first == second


def test_sign_flip_test_with_one_run_per_task_is_mcnemars_test():
    # 13 tasks where only a is right, 10 where only b is, and 127 where both
    # agree: the differences are 1, -1 and 0.
    differences = np.array([1.0] * 13 + [-1.0] * 10 + [0.0] * 127)
    p_value = sign_flip_test(differences, 20_000, np.random.default_rng(0))
    assert p_value == pytest.approx(mcnemar_exact(13, 10), abs=0.01)


def test_mcnemar_exact_hand_computed():
    # One task for a and 9 for b: P(count <= 1) = (1 + 10) / 1024, doubled.
    assert mcnemar_exact(1, 9) == pytest.approx(22 / 1024)
    assert mcnemar_exact(0, 5) == pytest.approx(2 / 32)


def test_mcnemar_exact_is_symmetric():
    assert mcnemar_exact(9, 1) == mcnemar_exact(1, 9)
    assert mcnemar_exact(13, 10) == mcnemar_exact(10, 13)


def test_mcnemar_exact_is_at_most_1():
    assert mcnemar_exact(5, 5) == 1.0
    assert mcnemar_exact(0, 0) == 1.0


def test_mcnemar_exact_of_a_close_comparison():
    expected = 2 * sum(math.comb(23, k) for k in range(11)) / 2**23
    assert mcnemar_exact(13, 10) == pytest.approx(expected)
    assert mcnemar_exact(13, 10) == pytest.approx(0.6776, abs=0.0001)


def test_holm_hand_computed():
    # Sorted: 0.005 * 4, 0.01 * 3, 0.03 * 2, then 0.04 * 1 is raised to 0.06.
    adjusted = holm([0.01, 0.04, 0.03, 0.005])
    assert adjusted == pytest.approx([0.03, 0.06, 0.06, 0.02])


def test_holm_of_one_p_value_changes_nothing():
    assert holm([0.03]) == pytest.approx([0.03])


def test_holm_of_no_p_values():
    assert holm([]) == []


def test_holm_is_at_most_1():
    assert holm([0.5, 0.6]) == pytest.approx([1.0, 1.0])


def test_holm_of_equal_p_values():
    assert holm([0.01, 0.01]) == pytest.approx([0.02, 0.02])


def test_holm_keeps_the_order_and_sits_between_raw_and_bonferroni():
    p_values = list(np.random.default_rng(0).random(8) / 10)
    adjusted = holm(p_values)
    order = np.argsort(p_values)
    assert np.all(np.diff(np.array(adjusted)[order]) >= 0)
    for p_value, holm_p_value in zip(p_values, adjusted):
        assert p_value <= holm_p_value <= min(1.0, len(p_values) * p_value)


def test_tasks_needed_textbook_values():
    # ((1.960 + 0.842) * sd / difference) ** 2 = 7.85 * (sd / difference) ** 2.
    assert tasks_needed(1.0, 1.0) == 8
    assert tasks_needed(1.0, 0.5) == 32
    assert tasks_needed(0.5, 0.1) == 197


def test_tasks_needed_returns_an_int():
    assert isinstance(tasks_needed(0.3, 0.05), int)


def test_tasks_needed_grows_with_the_square_of_the_noise():
    assert tasks_needed(0.6, 0.05) == pytest.approx(4 * tasks_needed(0.3, 0.05), abs=4)


def test_tasks_needed_more_for_a_stricter_level_or_more_power():
    assert tasks_needed(0.3, 0.05, alpha=0.01) > tasks_needed(0.3, 0.05)
    assert tasks_needed(0.3, 0.05, power=0.9) > tasks_needed(0.3, 0.05)
