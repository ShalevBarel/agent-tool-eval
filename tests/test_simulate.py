import numpy as np
import pytest

from agent_eval.analysis.simulate import simulate_correct, task_probabilities


def test_task_probabilities_are_chances():
    probabilities = task_probabilities(0.8, 1.0, 1000, np.random.default_rng(0))
    assert probabilities.shape == (1000,)
    assert ((probabilities >= 0) & (probabilities <= 1)).all()


def test_task_probabilities_have_the_mean():
    probabilities = task_probabilities(0.8, 1.0, 100_000, np.random.default_rng(0))
    assert probabilities.mean() == pytest.approx(0.8, abs=0.005)


def test_low_concentration_spreads_the_tasks_out():
    rng = np.random.default_rng(0)
    spread = task_probabilities(0.5, 0.5, 100_000, rng)
    close = task_probabilities(0.5, 50, 100_000, rng)
    # A beta distribution with mean m and concentration c has variance m(1 - m) / (c + 1).
    assert spread.var() == pytest.approx(0.25 / 1.5, rel=0.02)
    assert close.var() == pytest.approx(0.25 / 51, rel=0.02)


def test_simulate_correct_has_a_row_per_task_and_a_column_per_repeat():
    correct = simulate_correct(np.array([0.2, 0.5, 0.9, 0.7]), 3, np.random.default_rng(0))
    assert correct.shape == (4, 3)
    assert set(np.unique(correct)) <= {0, 1}


def test_simulate_correct_with_certain_tasks():
    correct = simulate_correct(np.array([0.0, 1.0]), 5, np.random.default_rng(0))
    assert correct.tolist() == [[0, 0, 0, 0, 0], [1, 1, 1, 1, 1]]


def test_simulate_correct_follows_each_task_s_chance():
    probabilities = np.array([0.1, 0.5, 0.9])
    correct = simulate_correct(probabilities, 100_000, np.random.default_rng(0))
    assert correct.mean(axis=1) == pytest.approx(probabilities, abs=0.01)
