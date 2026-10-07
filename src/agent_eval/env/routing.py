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
    if source not in world.roads:
        raise ValueError(f"unknown place: {source}")

    distances = {}
    previous = {}
    for place in world.roads:
        distances[place] = float("inf")
        previous[place] = None
    distances[source] = 0
    queue = [(0, source)]

    while queue:
        curr_dist, curr_place = heapq.heappop(queue)
        if curr_dist > distances[curr_place]:
            continue

        for neighbor in world.roads[curr_place]:
            if not world.is_open(curr_place, neighbor):
                continue
            # Relax method
            if curr_dist + world.roads[curr_place][neighbor] < distances[neighbor]:
                distances[neighbor] = curr_dist + world.roads[curr_place][neighbor]
                previous[neighbor] = curr_place
                heapq.heappush(queue, (distances[neighbor], neighbor))

    # Filter dicts to only reachable places
    final_distances = {k: v for k, v in distances.items() if v != float("inf")}
    final_previous = {k: v for k, v in previous.items() if v is not None}

    return final_distances, final_previous


def fastest_route(world: World, start: str, end: str) -> tuple[list[str], int] | None:
    """Return the fastest open route from start to end, and its minutes.

    The route lists every place on the way, start and end included.
    Returns None if end can't be reached.

    Raises:
        ValueError: If start or end is unknown. Start is checked first.
    """
    if start not in world.roads:
        raise ValueError(f"unknown place: {start}")
    if end not in world.roads:
        raise ValueError(f"unknown place: {end}")

    distances, previous = dijkstra(world, start)
    if end not in distances:
        return None

    route = [end]
    while route[-1] != start:
        route.append(previous[route[-1]])
    route.reverse()
    return route, distances[end]


def travel_minutes(world: World, places: list[str]) -> dict[str, dict[str, int]]:
    """Return the fastest open-route minutes from each of places to every place it can reach.

    Raises:
        ValueError: If a place is unknown.
    """
    travel = {}
    for place in places:
        distances, _ = dijkstra(world, place)
        travel[place] = distances
    return travel
