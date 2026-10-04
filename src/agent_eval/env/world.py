"""The world the agent works in: places, one-way roads, closures and orders.

Times are whole minutes from the start of the working day.
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class TimeWindow:
    """An inclusive range of minutes in which a delivery may arrive.

    Raises:
        ValueError: If start is negative or after end.
    """

    start: int
    end: int

    def __post_init__(self) -> None:
        if self.start < 0:
            raise ValueError("a window cannot start before minute 0")
        if self.start > self.end:
            raise ValueError("start must not be after end")

    def contains(self, minute: int) -> bool:
        """Return whether minute falls inside the window."""
        return self.start <= minute <= self.end


@dataclass(frozen=True)
class Order:
    """A delivery to a place, which must arrive inside its time window."""

    order_id: str
    place: str
    window: TimeWindow


@dataclass
class World:
    """A road network, today's road closures, and the orders waiting for delivery.

    Attributes:
        roads: For each place, the places reachable from it by a one-way
            road, and the minutes each road takes.
        closed: Roads closed today, as (start, end) pairs.
        orders: Orders by id.

    Invalid input raises ValueError. It signals a bug in the calling code,
    such as the task generator, not a mistake by the model.
    """

    roads: dict[str, dict[str, int]] = field(default_factory=dict)
    closed: set[tuple[str, str]] = field(default_factory=set)
    orders: dict[str, Order] = field(default_factory=dict)

    def add_place(self, place: str) -> None:
        """Add a place with no roads out of it.

        Raises:
            ValueError: If the place already exists.
        """
        if place in self.roads:
            raise ValueError(f"place {place} already exists")

        self.roads[place] = {}

    def add_road(self, start: str, end: str, minutes: int) -> None:
        """Add a one-way road from start to end.

        Raises:
            ValueError: If either place is unknown, minutes is below 1,
                or the road already exists.
        """
        if start not in self.roads:
            raise ValueError(f"unknown place: {start}")
        if end not in self.roads:
            raise ValueError(f"unknown place: {end}")
        if minutes < 1:
             raise ValueError("a road must take at least 1 minute")
        if end in self.roads[start]:
            raise ValueError(f"road from {start} to {end} already exists")

        self.roads[start][end] = minutes

    def close_road(self, start: str, end: str) -> None:
        """Mark the road from start to end as closed today.

        Raises:
            ValueError: If there is no such road.
        """
        if end not in self.roads.get(start, {}):
            raise ValueError(f"no road from {start} to {end}")
        
        self.closed.add((start, end))
        

    def is_open(self, start: str, end: str) -> bool:
        """Return whether a road from start to end exists and is open today."""
        return (end in self.roads.get(start, {})) and ((start, end) not in self.closed)

    def add_order(self, order: Order) -> None:
        """Add an order.

        Raises:
            ValueError: If the order id is taken or the place is unknown.
        """
        if order.order_id in self.orders:
            raise ValueError(f"order {order.order_id} already exists")
        if order.place not in self.roads:
            raise ValueError(f"unknown place: {order.place}")

        self.orders[order.order_id] = order
