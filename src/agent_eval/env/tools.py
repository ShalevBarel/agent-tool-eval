"""Tools the agent can call.

Each tool takes the world as its first argument, supplied by the agent loop.
The remaining arguments come from the model. Tools return JSON-serializable
values and raise ToolError on invalid arguments.
"""

from dataclasses import dataclass

from agent_eval.env.world import World


class ToolError(Exception):
    """Raised when the model calls a tool with invalid arguments."""


@dataclass(frozen=True)
class Answer:
    """The agent's final answer: the delivery order, or that there is no solution."""

    solvable: bool
    order_ids: list[str]


def get_order(world: World, order_id: str) -> dict:
    """Return an order's place and time window.

    Example:
        {"order_id": "o1", "place": "school", "window_start": 30, "window_end": 60}

    Raises:
        ToolError: If no order has this id.
    """
    if order_id not in world.orders:
        raise ToolError(f"no order with id {order_id}")
    
    order = world.orders[order_id]
    return {
        "order_id": order.order_id,
        "place": order.place,
        "window_start": order.window.start,
        "window_end": order.window.end,
    }


def roads_from(world: World, place: str) -> list[dict]:
    """Return the roads out of a place, sorted by destination.

    Closed roads are included, with "open" set to False.

    Example item:
        {"to": "market", "minutes": 30, "open": True}

    Raises:
        ToolError: If the place is unknown.
    """
    if place not in world.roads:
        raise ToolError(f"unknown place: {place}")

    roads_from_place = []
    for dest, time in sorted(world.roads[place].items()):
        roads_from_place.append({"to": dest, "minutes": time, "open": world.is_open(place, dest)})

    return roads_from_place


def route_minutes(world: World, route: list[str]) -> int:
    """Return the minutes it takes to drive a route, stop by stop.

    A route with a single stop takes 0 minutes. Every stop is checked
    before any road, so an unknown place is reported first.

    Raises:
        ToolError: If the route is empty, a stop is unknown, or a road
            on the route is missing or closed.
    """
    if not route:
        raise ToolError("route is empty")
    for place in route:
        if place not in world.roads:
            raise ToolError(f"unknown place: {place}")

    total = 0
    for i in range(len(route) - 1):
        start = route[i]
        end = route[i + 1]

        if end not in world.roads[start]:
            raise ToolError(f"no road from {start} to {end}")
        if not world.is_open(start, end):
            raise ToolError(f"the road from {start} to {end} is closed today")
        
        total += world.roads[start][end]

    return total


def submit_answer(world: World, solvable: bool, order_ids: list[str]) -> Answer:
    """Validate the agent's final answer and return it.

    order_ids is the delivery order. It must be empty when solvable is
    False, and non-empty when solvable is True.

    Raises:
        ToolError: If order_ids doesn't match solvable, names an unknown
            order, or lists an order twice.
    """
    if solvable and not order_ids:
        raise ToolError("a solvable answer must list the orders to deliver")
    if not solvable and order_ids:
        raise ToolError("an answer of no solution must not list orders")
    seen = set()
    for order_id in order_ids:
        if order_id not in world.orders:
            raise ToolError(f"no order with id {order_id}")
        if order_id in seen:
            raise ToolError(f"order {order_id} appears more than once")
        seen.add(order_id)

    return Answer(solvable, order_ids)
