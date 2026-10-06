"""A solver that tries every delivery order."""

from itertools import permutations

from agent_eval.env.routing import travel_minutes
from agent_eval.env.schedule import simulate
from agent_eval.env.world import World
from agent_eval.solver import Solution


def solve(world: World, order_ids: list[str]) -> Solution | None:
    """Return a delivery order that finishes earliest, or None if none is on time.

    Tries all n! orders of the n orders, so it only suits small tasks. When
    several orders finish at the same minute, returns the first one tried.

    Raises:
        ValueError: If order_ids is empty.
    """
    if not order_ids:
        raise ValueError("no orders to deliver")

    places = [world.start] + [world.orders[order_id].place for order_id in order_ids]
    travel = travel_minutes(world, places)

    best = None
    for candidate in permutations(order_ids):
        schedule = simulate(world, travel, list(candidate))
        if schedule.problem is not None:
            continue
        finish = schedule.stops[-1].delivery
        if best is None or finish < best.finish_minute:
            best = Solution(list(candidate), finish)

    return best
