"""Random worlds and tasks for tests that compare two implementations.

Every function takes a seeded random.Random, so the same seed always builds
the same world, and a failing test can name the seed that reproduces it.
"""

import random

from agent_eval.env.world import Order, TimeWindow, World


def random_world(rng: random.Random, road_chance: float = 0.3) -> World:
    """A world with up to 12 places, random one-way roads, and some of them closed.

    Each ordered pair of places gets a road with probability road_chance, and
    about a fifth of the roads are closed.
    """
    world = World()
    places = [f"p{i}" for i in range(rng.randint(1, 12))]
    for place in places:
        world.add_place(place)
    for start in places:
        for end in places:
            if start != end and rng.random() < road_chance:
                world.add_road(start, end, rng.randint(1, 30))
                if rng.random() < 0.2:
                    world.close_road(start, end)
    return world


def random_task(rng: random.Random) -> tuple[World, list[str]]:
    """A random world with 1 to 6 orders, a random start, and random time windows.

    Roads are denser than in random_world's default, so that about half of
    the tasks can be solved.
    """
    world = random_world(rng, road_chance=0.5)
    places = list(world.roads)
    order_ids = [f"o{i}" for i in range(1, rng.randint(1, 6) + 1)]
    for order_id in order_ids:
        opens = rng.randint(0, 120)
        window = TimeWindow(opens, opens + rng.randint(0, 90))
        world.add_order(Order(order_id, rng.choice(places), window))
    world.set_start(rng.choice(places), rng.randint(0, 30))
    return world, order_ids
