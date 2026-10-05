import pytest

from agent_eval.env.world import Order, TimeWindow
from agent_eval.solver import Solution
from agent_eval.solver.brute_force import solve


def test_solve_one_order(city):
    assert solve(city, ["o1"]) == Solution(["o1"], 30)


def test_solve_the_worked_example(city):
    assert solve(city, ["o1", "o2"]) == Solution(["o1", "o2"], 42)


def test_the_order_of_the_input_does_not_matter(city):
    assert solve(city, ["o2", "o1"]) == Solution(["o1", "o2"], 42)


def test_solve_three_orders_with_two_best_answers(city):
    solution = solve(city, ["o1", "o2", "o3"])
    assert solution.finish_minute == 90
    assert solution.order_ids in (["o1", "o2", "o3"], ["o2", "o1", "o3"])


def test_without_the_closure_both_orders_of_the_example_tie(city):
    city.closed.clear()
    solution = solve(city, ["o1", "o2"])
    assert solution.finish_minute == 42
    assert solution.order_ids in (["o1", "o2"], ["o2", "o1"])


def test_solve_from_a_later_start(city):
    city.set_start("warehouse", 50)
    assert solve(city, ["o3"]) == Solution(["o3"], 94)


def test_unreachable_order_means_no_solution(city):
    assert solve(city, ["o1", "o4"]) is None


def test_orders_that_fit_alone_but_not_together(city):
    city.add_order(Order("o5", "restaurant", TimeWindow(0, 30)))
    assert solve(city, ["o5"]) == Solution(["o5"], 25)
    assert solve(city, ["o1"]) == Solution(["o1"], 30)
    assert solve(city, ["o1", "o5"]) is None


def test_solve_no_orders_fails(city):
    with pytest.raises(ValueError, match="no orders to deliver"):
        solve(city, [])
