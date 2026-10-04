"""The world the agent works in: places, one-way roads, closures and orders.

Minutes count from the start of the working day, so minute 90 is 01:30.
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class TimeWindow:
    """The minutes in which an order may arrive, from start to end, both included.

    Check these in order when a window is created, and raise ValueError:
      start below 0:      "a window cannot start before minute 0"
      start after end:    "start must not be after end"
    """

    # Add two int fields here: start and end.

    def __post_init__(self) -> None:
        raise NotImplementedError

    def contains(self, minute: int) -> bool:
        """Return True if minute is inside the window."""
        raise NotImplementedError


@dataclass(frozen=True)
class Order:
    """A delivery to a place, which must arrive inside its time window."""

    # Add three fields here: order_id (a str), place (a str), and window (a TimeWindow).


@dataclass
class World:
    """A road network, today's road closures, and the orders waiting for delivery.

    roads maps each place to the places you can drive to from it,
    and how many minutes each drive takes. Roads are one-way.
    closed holds the roads that are closed today, as (start, end) pairs.
    orders maps an order id to its Order.

    The methods below raise ValueError on bad input. That means a bug in
    our own code, such as the task generator, and not a mistake by the model.
    """

    roads: dict[str, dict[str, int]] = field(default_factory=dict)
    closed: set[tuple[str, str]] = field(default_factory=set)
    orders: dict[str, Order] = field(default_factory=dict)

    def add_place(self, place: str) -> None:
        """Add a place, with no roads out of it yet.

        If the place exists, raise ValueError("place warehouse already exists").
        """
        raise NotImplementedError

    def add_road(self, start: str, end: str, minutes: int) -> None:
        """Add a one-way road from start to end.

        Check these in order, and raise ValueError with messages like:
          start or end unknown:   "unknown place: mall"
          minutes below 1:        "a road must take at least 1 minute"
          road already there:     "road from warehouse to office already exists"
        """
        raise NotImplementedError

    def close_road(self, start: str, end: str) -> None:
        """Mark the road from start to end as closed today.

        If there is no such road, raise ValueError("no road from warehouse to bank").
        """
        raise NotImplementedError

    def is_open(self, start: str, end: str) -> bool:
        """Return True if there is a road from start to end, and it isn't closed."""
        raise NotImplementedError

    def add_order(self, order: Order) -> None:
        """Add an order.

        Check these in order, and raise ValueError with messages like:
          id taken:           "order o1 already exists"
          place unknown:      "unknown place: mall"
        """
        raise NotImplementedError
