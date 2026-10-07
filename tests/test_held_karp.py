"""Tests for the Held-Karp solver's table, and for a task too large to try every order.

The tables use orders o1, o2 and o3 of the test city as bits 0, 1 and 2 of
the mask. The tests that both solvers share are in test_solver.py.
"""

from agent_eval.env.world import Order, TimeWindow, World
from agent_eval.solver import Solution
from agent_eval.solver.held_karp import earliest_deliveries, solve


def test_first_deliveries_come_straight_from_the_start(city, travel):
    earliest, _ = earliest_deliveries(city, travel, ["o1", "o2", "o3"])
    assert earliest[(0b001, 0)] == 30
    assert earliest[(0b010, 1)] == 37
    assert earliest[(0b100, 2)] == 90


def test_an_unreachable_order_never_enters_the_table(city, travel):
    earliest, previous = earliest_deliveries(city, travel, ["o1", "o4"])
    assert earliest == {(0b01, 0): 30}
    assert previous == {}


def test_table_for_three_orders(city, travel):
    earliest, _ = earliest_deliveries(city, travel, ["o1", "o2", "o3"])
    assert earliest == {
        (0b001, 0): 30,
        (0b010, 1): 37,
        (0b100, 2): 90,
        (0b011, 0): 49,
        (0b011, 1): 42,
        (0b101, 2): 90,
        (0b110, 2): 90,
        (0b111, 2): 90,
    }


def test_each_state_remembers_the_order_before_it(city, travel):
    _, previous = earliest_deliveries(city, travel, ["o1", "o2", "o3"])
    assert previous[(0b011, 0)] == 1
    assert previous[(0b011, 1)] == 0
    assert previous[(0b101, 2)] == 0
    assert previous[(0b110, 2)] == 1
    assert previous[(0b111, 2)] in (0, 1)
    assert len(previous) == 5


def street(count: int) -> World:
    """A straight two-way street with the courier at one end and an order at each other place.

    Places are 10 minutes apart, and order oi is i places from the start. The
    farthest order closes when the courier first gets there, and every other
    order opens 30 minutes later. So the only fast plan drives straight to the
    far end, then delivers the rest on the way back.
    """
    world = World()
    world.add_place("p0")
    for i in range(1, count + 1):
        world.add_place(f"p{i}")
        world.add_road(f"p{i - 1}", f"p{i}", 10)
        world.add_road(f"p{i}", f"p{i - 1}", 10)
        if i == count:
            window = TimeWindow(0, 10 * count)
        else:
            window = TimeWindow(10 * count + 30, 1000)
        world.add_order(Order(f"o{i}", f"p{i}", window))
    world.set_start("p0", 0)
    return world


def test_twelve_orders_far_beyond_trying_every_order():
    world = street(12)
    order_ids = [f"o{i}" for i in range(1, 13)]
    back_from_the_far_end = [f"o{i}" for i in range(12, 0, -1)]
    assert solve(world, order_ids) == Solution(back_from_the_far_end, 250)
