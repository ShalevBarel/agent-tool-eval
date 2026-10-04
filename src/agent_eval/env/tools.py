"""The tools the agent can call.

Each tool is a plain Python function that knows nothing about the model.
The first parameter is always the world. The agent loop fills it in, and
the model never sees it. The other parameters are what the model sends.

Tools return plain values: strings, numbers, booleans, and lists and dicts
of them. The agent loop turns the result into text for the model.

When the model sends arguments that make no sense, a tool raises ToolError.
The agent loop sends the message back to the model, so the message has to
say exactly what was wrong.
"""

from dataclasses import dataclass

from agent_eval.env.world import World


class ToolError(Exception):
    """The model called a tool with bad arguments. The message goes back to it."""


@dataclass(frozen=True)
class Answer:
    """The agent's final answer: the order to deliver in, or that it can't be done."""

    solvable: bool
    order_ids: list[str]


def get_order(world: World, order_id: str) -> dict:
    """Return the details of an order.

    get_order(city, "o1") -> {"order_id": "o1", "place": "school",
                              "window_start": 30, "window_end": 60}

    For an unknown id, raise ToolError("no order with id o9").
    """
    raise NotImplementedError


def roads_from(world: World, place: str) -> list[dict]:
    """Return the roads out of place, sorted by the place they lead to.

    Closed roads are included, marked with "open": False.
    roads_from(city, "office") -> [
        {"to": "market", "minutes": 30, "open": True},
        {"to": "school", "minutes": 8, "open": False},
        {"to": "warehouse", "minutes": 10, "open": True},
    ]

    For an unknown place, raise ToolError("unknown place: mall").
    """
    raise NotImplementedError


def route_minutes(world: World, route: list[str]) -> int:
    """Return how many minutes it takes to drive along route, stop by stop.

    route_minutes(city, ["warehouse", "office", "market"]) -> 40
    A route with a single stop takes 0 minutes.

    Check these in order, and raise ToolError with messages like:
      empty route:                  "route is empty"
      any stop unknown:             "unknown place: mall"
      first road that is missing:   "no road from warehouse to bank"
      first road that is closed:    "the road from office to school is closed today"
    Check every stop before you check any road.
    """
    raise NotImplementedError


def submit_answer(world: World, solvable: bool, order_ids: list[str]) -> Answer:
    """Check the agent's final answer, and return it as an Answer.

    solvable is False when no order of deliveries works, and then order_ids
    must be empty. Otherwise order_ids is the order to deliver in.

    Check these in order, and raise ToolError with messages like:
      solvable, but no orders:      "a solvable answer must list the orders to deliver"
      not solvable, but orders:     "an answer of no solution must not list orders"
      unknown id:                   "no order with id o9"
      same id twice:                "order o1 appears more than once"
    """
    raise NotImplementedError
