"""Shared test setup. pytest loads this file before the tests in this folder."""

import pytest

from agent_eval.env.world import Order, TimeWindow, World


@pytest.fixture
def city() -> World:
    """A small hand-made city. Every test that asks for it gets a fresh copy.

    A parcel courier starts at the warehouse. Two-way streets are two one-way
    roads. The road from office to school is closed, and no road leads to the
    gas station, so order o4 can't be delivered.
    """
    world = World()
    for place in ["warehouse", "office", "school", "market", "bank", "restaurant", "gas station"]:
        world.add_place(place)

    world.add_road("warehouse", "office", 10)
    world.add_road("office", "warehouse", 10)
    world.add_road("warehouse", "school", 25)
    world.add_road("office", "school", 8)
    world.add_road("office", "market", 30)
    world.add_road("school", "market", 12)
    world.add_road("market", "school", 12)
    world.add_road("market", "bank", 7)
    world.add_road("bank", "restaurant", 15)
    world.add_road("restaurant", "warehouse", 25)
    world.add_road("warehouse", "restaurant", 25)
    world.add_road("gas station", "warehouse", 5)
    world.close_road("office", "school")

    world.add_order(Order("o1", "school", TimeWindow(30, 60)))
    world.add_order(Order("o2", "market", TimeWindow(0, 45)))
    world.add_order(Order("o3", "bank", TimeWindow(90, 120)))
    world.add_order(Order("o4", "gas station", TimeWindow(0, 200)))
    return world
