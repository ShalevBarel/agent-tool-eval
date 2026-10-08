"""Cross-check Dijkstra against networkx on many random worlds.

Each seed builds a different random world, so a failing test names the
seed that reproduces it.
"""

import random

import networkx as nx
import pytest

from agent_eval.env.routing import dijkstra, fastest_route
from agent_eval.env.tools import route_minutes
from agent_eval.env.world import World
from random_worlds import random_world


def open_roads_graph(world: World) -> nx.DiGraph:
    """The same world as a networkx graph, with only the open roads."""
    graph = nx.DiGraph()
    graph.add_nodes_from(world.roads)
    for start, ends in world.roads.items():
        for end, minutes in ends.items():
            if world.is_open(start, end):
                graph.add_edge(start, end, weight=minutes)
    return graph


@pytest.mark.parametrize("seed", range(100))
def test_dijkstra_matches_networkx(seed):
    world = random_world(random.Random(seed))
    graph = open_roads_graph(world)
    for source in world.roads:
        minutes, _ = dijkstra(world, source)
        assert minutes == nx.single_source_dijkstra_path_length(graph, source)


@pytest.mark.parametrize("seed", range(100))
def test_fastest_routes_are_drivable_and_take_the_minutes_they_claim(seed):
    world = random_world(random.Random(seed))
    for start in world.roads:
        minutes, _ = dijkstra(world, start)
        for end in minutes:
            route, total = fastest_route(world, start, end)
            assert route[0] == start
            assert route[-1] == end
            assert route_minutes(world, route) == total == minutes[end]
