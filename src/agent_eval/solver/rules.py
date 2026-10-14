"""Simple rules that choose a delivery order without searching.

Each rule is fast and often finds the best order. The task generator uses
them to keep only tasks that need a real search: on a kept task, no rule
reaches the earliest possible finish.
"""

from agent_eval.env.tools import check_schedule
from agent_eval.env.world import World


def by_opening(world: World) -> list[str]:
    """Return the order ids sorted by when their windows open, ties in id order."""
    return sorted(world.orders, key=lambda order_id: world.orders[order_id].window.start)


def by_closing(world: World) -> list[str]:
    """Return the order ids sorted by when their windows close, ties in id order."""
    return sorted(world.orders, key=lambda order_id: world.orders[order_id].window.end)


def earliest_next(world: World) -> list[str] | None:
    """Build a delivery order by always going next to the order delivered earliest.

    At each step, every order not chosen yet is tried after the ones chosen
    so far, and the one with the earliest delivery minute among those on
    time comes next. A tie goes to the order that comes first in
    world.orders.

    Returns None if at some step no remaining order can be delivered on time.
    """
    chosen = []
    while len(chosen) < len(world.orders):
        best = None
        best_minute = None
        for order_id in world.orders:
            if order_id in chosen:
                continue
            report = check_schedule(world, chosen + [order_id])
            if report["on_time"] and (best is None or report["finish_minute"] < best_minute):
                best = order_id
                best_minute = report["finish_minute"]
        if best is None:
            return None
        chosen.append(best)
    return chosen


# Every rule, each a function from a world to a delivery order, or None.
RULES = [by_opening, by_closing, earliest_next]
