"""Driving a delivery order from the courier's start, one stop at a time."""

from dataclasses import dataclass

from agent_eval.env.world import World


@dataclass(frozen=True)
class Stop:
    """One delivery: when the courier arrives, and when it delivers after any wait."""

    order_id: str
    place: str
    arrival: int
    delivery: int


@dataclass(frozen=True)
class Schedule:
    """The result of driving a delivery order.

    Attributes:
        stops: The deliveries made, in order, up to the first problem.
        problem: Why the delivery order fails, or None if every delivery
            is on time.
    """

    stops: list[Stop]
    problem: str | None


def simulate(world: World, travel: dict[str, dict[str, int]], order_ids: list[str]) -> Schedule:
    """Drive the orders in the given order, from the courier's start.

    The courier takes the fastest open route to each order, and waits if it
    arrives before the window opens. The simulation stops at the first order
    that can't be reached, or that the courier would reach after its window
    closes.

    travel holds the fastest minutes between places, as returned by
    travel_minutes. It must cover the courier's start and every order's place.
    """
    raise NotImplementedError
