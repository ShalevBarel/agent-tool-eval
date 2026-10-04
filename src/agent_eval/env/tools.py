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
    raise NotImplementedError


def roads_from(world: World, place: str) -> list[dict]:
    """Return the roads out of a place, sorted by destination.

    Closed roads are included, with "open" set to False.

    Example item:
        {"to": "market", "minutes": 30, "open": True}

    Raises:
        ToolError: If the place is unknown.
    """
    raise NotImplementedError


def route_minutes(world: World, route: list[str]) -> int:
    """Return the minutes it takes to drive a route, stop by stop.

    A route with a single stop takes 0 minutes. Every stop is checked
    before any road, so an unknown place is reported first.

    Raises:
        ToolError: If the route is empty, a stop is unknown, or a road
            on the route is missing or closed.
    """
    raise NotImplementedError


def submit_answer(world: World, solvable: bool, order_ids: list[str]) -> Answer:
    """Validate the agent's final answer and return it.

    order_ids is the delivery order. It must be empty when solvable is
    False, and non-empty when solvable is True.

    Raises:
        ToolError: If order_ids doesn't match solvable, names an unknown
            order, or lists an order twice.
    """
    raise NotImplementedError
