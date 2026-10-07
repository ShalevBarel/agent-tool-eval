"""A Held-Karp style solver: dynamic programming over subsets of orders.

A state is the set of orders delivered so far and the order delivered last.
A set is a bitmask: bit i is on when order_ids[i] has been delivered. For each
state the solver keeps only the earliest minute it can be reached. That is
safe because the courier may wait, so arriving earlier never rules out
anything that arriving later allows.
"""

from agent_eval.env.world import Order, World
from agent_eval.solver import Solution

type State = tuple[int, int]


def solve(world: World, order_ids: list[str]) -> Solution | None:
    """Return a delivery order that finishes earliest, or None if none is on time.

    Takes O(2^n * n^2) time and O(2^n * n) memory for n orders, so it handles
    far larger tasks than trying every order. When several orders finish at
    the same minute, which one it returns is unspecified.

    Raises:
        ValueError: If order_ids is empty.
    """
    raise NotImplementedError


def earliest_deliveries(
    world: World, travel: dict[str, dict[str, int]], order_ids: list[str]
) -> tuple[dict[State, int], dict[State, int]]:
    """Fill the table of earliest deliveries, one entry per state.

    A state (mask, last) means the courier has delivered exactly the orders
    whose bits are on in mask, and delivered order_ids[last] last. Returns
    two dicts. The first maps each state the courier can reach on time to the
    earliest minute of that last delivery. The second maps each such state
    with two or more orders to the index of the order delivered just before
    last. States that can't be reached on time appear in neither.

    travel holds the fastest minutes between places, as returned by
    travel_minutes. It must cover the courier's start and every order's place.
    """
    raise NotImplementedError


def _delivery_minute(
    travel: dict[str, dict[str, int]], place: str, minute: int, order: Order
) -> int | None:
    """Return the minute order is delivered if the courier leaves place at minute.

    The courier waits if it arrives before the window opens. Returns None if
    the order's place can't be reached from place, or would be reached after
    the window closes.
    """
    raise NotImplementedError
