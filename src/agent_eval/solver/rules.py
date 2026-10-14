"""Simple rules that choose a delivery order without searching.

Each rule is fast and often finds the best order. The task generator uses
them to keep only tasks that need a real search: on a kept task, no rule
reaches the earliest possible finish.
"""

from agent_eval.env.world import World


def by_opening(world: World) -> list[str]:
    """Return the order ids sorted by when their windows open, ties in id order."""
    return sorted(world.orders, key=lambda order_id: world.orders[order_id].window.start)


def by_closing(world: World) -> list[str]:
    """Return the order ids sorted by when their windows close, ties in id order."""
    raise NotImplementedError


def earliest_next(world: World) -> list[str] | None:
    """Build a delivery order by always going next to the order delivered earliest.

    At each step, every order not chosen yet is tried after the ones chosen
    so far, and the one with the earliest delivery minute among those on
    time comes next. A tie goes to the order that comes first in
    world.orders.

    Returns None if at some step no remaining order can be delivered on time.
    """
    raise NotImplementedError


# Every rule, each a function from a world to a delivery order, or None.
RULES = [by_opening, by_closing, earliest_next]
