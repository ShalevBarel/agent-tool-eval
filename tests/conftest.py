"""Shared test setup. pytest loads this file before the tests in this folder."""

import pytest

from agent_eval.env.world import Order, TimeWindow, World


@pytest.fixture
def city() -> World:
    """A small hand-made city. Every test that asks for it gets a fresh copy.

    A parcel courier starts at the warehouse at minute 0. Two-way streets are
    two one-way roads. The road from office to school is closed, and no road
    leads to the gas station, so order o4 can't be delivered.
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

    world.set_start("warehouse", 0)
    return world


@pytest.fixture
def travel() -> dict[str, dict[str, int]]:
    """The fastest open-route minutes in the test city, worked out by hand.

    One row for the warehouse, and one for each order's place that the
    courier can reach. A place missing from a row can't be reached from it.
    """
    return {
        "warehouse": {
            "warehouse": 0,
            "office": 10,
            "school": 25,
            "restaurant": 25,
            "market": 37,
            "bank": 44,
        },
        "school": {
            "school": 0,
            "market": 12,
            "bank": 19,
            "restaurant": 34,
            "warehouse": 59,
            "office": 69,
        },
        "market": {
            "market": 0,
            "bank": 7,
            "school": 12,
            "restaurant": 22,
            "warehouse": 47,
            "office": 57,
        },
        "bank": {
            "bank": 0,
            "restaurant": 15,
            "warehouse": 40,
            "office": 50,
            "school": 65,
            "market": 77,
        },
    }
