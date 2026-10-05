"""Fastest routes over the roads that are open today."""

import heapq

from agent_eval.env.world import World


def dijkstra(world: World, source: str) -> tuple[dict[str, int], dict[str, str]]:
    """Find the fastest open route from source to every place it can reach.

    Returns two dicts. The first maps each reachable place to its minutes
    from source, with source itself at 0. The second maps each reachable
    place other than source to the place just before it on its fastest
    route. Places that can't be reached appear in neither.

    Raises:
        ValueError: If source is unknown.
    """
    raise NotImplementedError


def fastest_route(world: World, start: str, end: str) -> tuple[list[str], int] | None:
    """Return the fastest open route from start to end, and its minutes.

    The route lists every place on the way, start and end included.
    Returns None if end can't be reached.

    Raises:
        ValueError: If start or end is unknown. Start is checked first.
    """
    raise NotImplementedError


def travel_minutes(world: World, places: list[str]) -> dict[str, dict[str, int]]:
    """Return the fastest open-route minutes from each of places to every place it can reach.

    Raises:
        ValueError: If a place is unknown.
    """
    raise NotImplementedError
