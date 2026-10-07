"""Cross-check the two exact solvers on thousands of random tasks.

The solvers search in different ways, so a bug in one is very unlikely to be
matched by the same wrong answer from the other. Each seed builds a different
random task, so a failing test names the seed that reproduces it.
"""

import random

import pytest

from agent_eval.env.tools import check_schedule
from agent_eval.solver import brute_force, held_karp
from random_worlds import random_task

SEEDS = range(2000)


@pytest.mark.parametrize("seed", SEEDS)
def test_solvers_agree(seed):
    world, order_ids = random_task(random.Random(seed))
    expected = brute_force.solve(world, order_ids)
    actual = held_karp.solve(world, order_ids)
    if expected is None:
        assert actual is None
        return

    assert actual is not None
    assert actual.finish_minute == expected.finish_minute
    assert sorted(actual.order_ids) == sorted(order_ids)
    report = check_schedule(world, actual.order_ids)
    assert report["on_time"]
    assert report["finish_minute"] == actual.finish_minute


def test_random_tasks_are_a_fair_mix():
    """Agreement proves little if almost every task has the same answer."""
    solvable = 0
    for seed in SEEDS:
        world, order_ids = random_task(random.Random(seed))
        if brute_force.solve(world, order_ids) is not None:
            solvable += 1
    assert 0.3 * len(SEEDS) <= solvable <= 0.7 * len(SEEDS)
