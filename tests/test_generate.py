import random
import subprocess
import sys
from pathlib import Path

import pytest

from agent_eval.env.generate import (
    LATEST_OPEN,
    LEVELS,
    MAX_PLACES,
    MAX_WIDTH,
    MIN_PLACES,
    MIN_WIDTH,
    add_random_orders,
    each_order_alone_on_time,
    generate,
    random_city,
)
from agent_eval.env.routing import dijkstra
from agent_eval.env.tasks import load_tasks
from agent_eval.env.tools import check_schedule
from agent_eval.env.world import Order, TimeWindow
from agent_eval.solver import brute_force, held_karp

SAVED_TASKS = Path(__file__).resolve().parent.parent / "data" / "tasks.jsonl"


@pytest.fixture(scope="module")
def tasks():
    """Twenty tasks at each difficulty from seed 0, generated once for this file."""
    return generate(seed=0, per_level=20)


@pytest.mark.parametrize("seed", range(20))
def test_random_city_is_connected_while_roads_are_open(seed):
    world = random_city(random.Random(seed))
    places = list(world.roads)
    assert MIN_PLACES <= len(places) <= MAX_PLACES
    assert places == [f"p{i}" for i in range(len(places))]
    assert world.start == "p0"
    assert world.start_minute == 0
    world.closed.clear()
    for place in places:
        distances, _ = dijkstra(world, place)
        assert len(distances) == len(places)


@pytest.mark.parametrize("count", [1, 3, 5])
def test_add_random_orders(count):
    rng = random.Random(count)
    world = random_city(rng)
    add_random_orders(rng, world, count)
    assert list(world.orders) == [f"o{i}" for i in range(1, count + 1)]
    places = [order.place for order in world.orders.values()]
    assert len(set(places)) == count
    assert "p0" not in places
    for order in world.orders.values():
        assert 0 <= order.window.start <= LATEST_OPEN
        assert MIN_WIDTH <= order.window.end - order.window.start <= MAX_WIDTH


def test_add_random_orders_repeats_with_the_same_seed():
    first = random_city(random.Random(5))
    second = random_city(random.Random(5))
    add_random_orders(random.Random(9), first, 4)
    add_random_orders(random.Random(9), second, 4)
    assert first.orders == second.orders


def test_each_order_alone_on_time_finds_an_unreachable_order(city):
    assert not each_order_alone_on_time(city)


def test_each_order_alone_on_time_but_not_together(city):
    del city.orders["o4"]
    del city.orders["o1"]
    del city.orders["o3"]
    city.add_order(Order("o5", "restaurant", TimeWindow(0, 30)))
    assert each_order_alone_on_time(city)
    assert held_karp.solve(city, ["o2", "o5"]) is None


def test_counts(tasks):
    assert len(tasks) == 60
    for difficulty in LEVELS:
        level = [task for task in tasks if task.difficulty == difficulty]
        assert len(level) == 20
        assert sum(1 for task in level if task.answer is None) == 3


def test_levels_come_in_order_with_numbered_ids(tasks):
    assert [task.difficulty for task in tasks] == ["easy"] * 20 + ["medium"] * 20 + ["hard"] * 20
    assert [task.task_id for task in tasks] == [f"task-{i:03d}" for i in range(60)]


def test_order_counts_match_the_level(tasks):
    for task in tasks:
        fewest, most = LEVELS[task.difficulty]
        assert fewest <= len(task.world.orders) <= most
    assert {len(task.world.orders) for task in tasks} == {1, 2, 3, 4, 5}


def test_answers_are_on_time(tasks):
    for task in tasks:
        if task.answer is None:
            continue
        assert sorted(task.answer.order_ids) == sorted(task.world.orders)
        report = check_schedule(task.world, task.answer.order_ids)
        assert report["on_time"]
        assert report["finish_minute"] == task.answer.finish_minute


def test_answers_match_brute_force(tasks):
    for task in tasks:
        best = brute_force.solve(task.world, list(task.world.orders))
        if task.answer is None:
            assert best is None
        else:
            assert best.finish_minute == task.answer.finish_minute


def test_impossible_tasks_hide_the_problem(tasks):
    for task in tasks:
        if task.answer is None and len(task.world.orders) > 1:
            assert each_order_alone_on_time(task.world)


def test_same_seed_same_tasks():
    assert generate(seed=3, per_level=5) == generate(seed=3, per_level=5)


def test_different_seeds_differ():
    assert generate(seed=3, per_level=5) != generate(seed=4, per_level=5)


def test_impossible_share():
    tasks = generate(seed=1, per_level=10, impossible_share=0.5)
    for difficulty in LEVELS:
        level = [task for task in tasks if task.difficulty == difficulty]
        assert sum(1 for task in level if task.answer is None) == 5


def test_no_impossible_tasks():
    tasks = generate(seed=1, per_level=4, impossible_share=0)
    assert all(task.answer is not None for task in tasks)


def test_same_seed_same_file_in_new_processes(tmp_path):
    """Each run is a fresh Python process, so set order can differ between them."""
    paths = [tmp_path / "first.jsonl", tmp_path / "second.jsonl"]
    for path in paths:
        command = [sys.executable, "-m", "agent_eval.env.generate"]
        command += ["--seed", "0", "--per-level", "10", "--out", str(path)]
        subprocess.run(command, check=True, capture_output=True)
    assert paths[0].read_bytes() == paths[1].read_bytes()


def test_saved_tasks_match_the_generator():
    """The committed task file is exactly what seed 0 generates, 50 tasks per level."""
    assert load_tasks(SAVED_TASKS) == generate(seed=0, per_level=50)
