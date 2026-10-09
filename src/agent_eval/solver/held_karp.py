"""A Held-Karp style solver: dynamic programming over subsets of orders.

A state is the set of orders delivered so far and the order delivered last.
A set is a bitmask: bit i is on when order_ids[i] has been delivered. For each
state the solver keeps only the earliest minute it can be reached. That is
safe because the courier may wait, so arriving earlier never rules out
anything that arriving later allows.
"""

from agent_eval.env.routing import travel_minutes
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
    if not order_ids:
        raise ValueError("no orders to deliver")

    places = [world.start] + [world.orders[order_id].place for order_id in order_ids]
    travel = travel_minutes(world, places)
    earliest, previous = earliest_deliveries(world, travel, order_ids)

    everything = (1 << len(order_ids)) - 1
    last = None
    for i in range(len(order_ids)):
        if (everything, i) not in earliest:
            continue
        if last is None or earliest[(everything, i)] < earliest[(everything, last)]:
            last = i
    if last is None:
        return None

    finish = earliest[(everything, last)]
    mask = everything
    order = [order_ids[last]]
    while (mask, last) in previous:
        before = previous[(mask, last)]
        mask = mask ^ (1 << last)
        last = before
        order.append(order_ids[last])
    order.reverse()
    return Solution(order, finish)


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
    orders = [world.orders[order_id] for order_id in order_ids]
    earliest = {}
    previous = {}
    for i, order in enumerate(orders):
        minute = _delivery_minute(travel, world.start, world.start_minute, order)
        if minute is not None:
            earliest[(1 << i, i)] = minute

    for mask in range(1, 1 << len(orders)):
        for last in range(len(orders)):
            if (mask, last) not in earliest:
                continue
            for nxt in range(len(orders)):
                if mask & (1 << nxt):
                    continue
                minute = _delivery_minute(
                    travel, orders[last].place, earliest[(mask, last)], orders[nxt]
                )
                if minute is None:
                    continue
                state = (mask | (1 << nxt), nxt)
                if state not in earliest or minute < earliest[state]:
                    earliest[state] = minute
                    previous[state] = last

    return earliest, previous


def _delivery_minute(
    travel: dict[str, dict[str, int]], place: str, minute: int, order: Order
) -> int | None:
    """Return the minute order is delivered if the courier leaves place at minute.

    The courier waits if it arrives before the window opens. Returns None if
    the order's place can't be reached from place, or would be reached after
    the window closes.
    """
    if order.place not in travel[place]:
        return None
    arrival = travel[place][order.place] + minute
    if arrival > order.window.end:
        return None
    return max(arrival, order.window.start)
