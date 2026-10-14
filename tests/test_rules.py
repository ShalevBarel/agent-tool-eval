import random

import pytest

from agent_eval.env.tools import check_schedule
from agent_eval.env.world import Order, TimeWindow
from agent_eval.solver.rules import RULES, by_closing, by_opening, earliest_next
from random_worlds import random_task


@pytest.fixture
def trap(city):
    """Three orders where every rule goes to the bank too early.

    From the warehouse, the school (o1) is reached at 25, the market (o2) at
    37 but it opens at 60, and the bank (o3) at 44. The only order on time is
    o1 o2 o3, ending at the bank at 67: from the bank, the market is 77
    minutes away.
    """
    city.orders.clear()
    city.add_order(Order("o1", "school", TimeWindow(0, 60)))
    city.add_order(Order("o2", "market", TimeWindow(60, 120)))
    city.add_order(Order("o3", "bank", TimeWindow(20, 80)))
    return city


def test_by_opening(city):
    assert by_opening(city) == ["o2", "o4", "o1", "o3"]


def test_by_closing(city):
    assert by_closing(city) == ["o2", "o1", "o3", "o4"]


def test_earliest_next(city):
    del city.orders["o4"]
    assert earliest_next(city) == ["o1", "o2", "o3"]


def test_earliest_next_gets_stuck_on_an_unreachable_order(city):
    assert earliest_next(city) is None


def test_earliest_next_breaks_a_tie_by_id(city):
    city.orders.clear()
    city.add_order(Order("o1", "market", TimeWindow(50, 100)))
    city.add_order(Order("o2", "school", TimeWindow(50, 100)))
    assert earliest_next(city) == ["o1", "o2"]


def test_rules_on_the_trap(trap):
    assert by_opening(trap) == ["o1", "o3", "o2"]
    assert by_closing(trap) == ["o1", "o3", "o2"]
    assert earliest_next(trap) is None
    assert check_schedule(trap, ["o1", "o2", "o3"])["finish_minute"] == 67


def test_rules_leave_the_world_as_it_was(city):
    orders = dict(city.orders)
    for rule in RULES:
        rule(city)
    assert city.orders == orders


@pytest.mark.parametrize("seed", range(200))
def test_rules_give_every_order_once(seed):
    world, order_ids = random_task(random.Random(seed))
    for rule in RULES:
        chosen = rule(world)
        if chosen is not None:
            assert sorted(chosen) == sorted(order_ids)


@pytest.mark.parametrize("seed", range(200))
def test_earliest_next_is_on_time_when_it_finishes(seed):
    world, _ = random_task(random.Random(seed))
    chosen = earliest_next(world)
    if chosen is not None:
        assert check_schedule(world, chosen)["on_time"]
