import json

import pytest

from agent_eval.env.tasks import (
    Task,
    load_tasks,
    save_tasks,
    task_from_dict,
    task_to_dict,
    world_from_dict,
    world_to_dict,
)
from agent_eval.env.world import World
from agent_eval.solver import Solution


def through_json(data: dict) -> dict:
    """Return data as json would read it back: tuples become lists."""
    return json.loads(json.dumps(data))


def test_world_to_dict(city):
    data = through_json(world_to_dict(city))
    assert data["roads"]["warehouse"] == {"office": 10, "school": 25, "restaurant": 25}
    assert data["roads"]["gas station"] == {"warehouse": 5}
    assert data["closed"] == [["office", "school"]]
    assert len(data["orders"]) == 4
    assert data["orders"][0] == {
        "order_id": "o1",
        "place": "school",
        "window_start": 30,
        "window_end": 60,
    }
    assert data["start"] == "warehouse"
    assert data["start_minute"] == 0


def test_world_to_dict_sorts_closed_roads(city):
    city.close_road("market", "bank")
    city.close_road("bank", "restaurant")
    assert through_json(world_to_dict(city))["closed"] == [
        ["bank", "restaurant"],
        ["market", "bank"],
        ["office", "school"],
    ]


def test_world_round_trip(city):
    assert world_from_dict(through_json(world_to_dict(city))) == city


def test_world_without_start_round_trip():
    world = World()
    world.add_place("p0")
    data = through_json(world_to_dict(world))
    assert data["start"] is None
    assert world_from_dict(data) == world


def test_world_from_dict_checks_closed_roads(city):
    data = through_json(world_to_dict(city))
    data["closed"].append(["bank", "school"])
    with pytest.raises(ValueError):
        world_from_dict(data)


def test_task_round_trip(city):
    task = Task("task-000", "medium", city, Solution(["o2", "o1"], 49))
    data = through_json(task_to_dict(task))
    assert data["task_id"] == "task-000"
    assert data["difficulty"] == "medium"
    assert data["answer"] == {"order_ids": ["o2", "o1"], "finish_minute": 49}
    assert task_from_dict(data) == task


def test_task_without_answer_round_trip(city):
    task = Task("task-001", "hard", city, None)
    data = through_json(task_to_dict(task))
    assert data["answer"] is None
    assert task_from_dict(data) == task


def test_save_and_load(city, tmp_path):
    tasks = [
        Task("task-000", "easy", city, Solution(["o1"], 30)),
        Task("task-001", "hard", city, None),
    ]
    path = tmp_path / "tasks.jsonl"
    save_tasks(tasks, path)
    assert load_tasks(path) == tasks


def test_saved_file_has_one_task_per_line(city, tmp_path):
    tasks = [
        Task("task-000", "easy", city, Solution(["o1"], 30)),
        Task("task-001", "hard", city, None),
    ]
    path = tmp_path / "tasks.jsonl"
    save_tasks(tasks, path)
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert json.loads(lines[1])["task_id"] == "task-001"


def test_save_replaces_the_file(city, tmp_path):
    path = tmp_path / "tasks.jsonl"
    save_tasks([Task("task-000", "easy", city, None)], path)
    save_tasks([Task("task-001", "hard", city, None)], path)
    assert [task.task_id for task in load_tasks(path)] == ["task-001"]
