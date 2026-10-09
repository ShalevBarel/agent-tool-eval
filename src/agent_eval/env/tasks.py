"""Delivery tasks, and how they are stored as JSON Lines: one task per line."""

import json
from dataclasses import dataclass
from pathlib import Path

from agent_eval.env.world import Order, TimeWindow, World
from agent_eval.solver import Solution


@dataclass(frozen=True)
class Task:
    """A world whose orders must all be delivered, and the exact answer.

    Attributes:
        task_id: A unique id, such as "task-007".
        difficulty: "easy", "medium" or "hard", set by the number of orders.
        world: The city, today's closures, the orders and the courier's start.
        answer: A delivery order that finishes earliest, or None if no
            delivery order is on time.
    """

    task_id: str
    difficulty: str
    world: World
    answer: Solution | None


def world_to_dict(world: World) -> dict:
    """Return the world as a dict that json can write.

    Closed roads are sorted, so the same world always gives the same dict.

    Example:
        {
            "roads": {"p0": {"p1": 12}, "p1": {"p0": 12}},
            "closed": [("p1", "p0")],
            "orders": [{"order_id": "o1", "place": "p1", "window_start": 30, "window_end": 60}],
            "start": "p0",
            "start_minute": 0,
        }
    """
    orders = []
    for order in world.orders.values():
        orders.append(
            {
                "order_id": order.order_id,
                "place": order.place,
                "window_start": order.window.start,
                "window_end": order.window.end,
            }
        )

    return {
        "roads": {place: dict(ends) for place, ends in world.roads.items()},
        "closed": sorted(world.closed),
        "orders": orders,
        "start": world.start,
        "start_minute": world.start_minute,
    }


def world_from_dict(data: dict) -> World:
    """Build a world from a dict made by world_to_dict.

    Raises:
        ValueError: If the dict describes an invalid world, such as a road
            to an unknown place.
    """
    world = World()
    for place in data["roads"]:
        world.add_place(place)
    for start, end in data["roads"].items():
        for end, minutes in end.items():
            world.add_road(start, end, minutes)
    for start, end in data["closed"]:
        world.close_road(start, end)
    for order in data["orders"]:
        world.add_order(
            Order(
                order["order_id"],
                order["place"],
                TimeWindow(order["window_start"], order["window_end"]),
            )
        )
    if data["start"] is not None:
        world.set_start(data["start"], data["start_minute"])
    return world


def task_to_dict(task: Task) -> dict:
    """Return the task as a dict that json can write. A missing answer becomes None."""
    if task.answer is None:
        answer = None
    else:
        answer = {
            "order_ids": task.answer.order_ids,
            "finish_minute": task.answer.finish_minute,
        }
    return {
        "task_id": task.task_id,
        "difficulty": task.difficulty,
        "world": world_to_dict(task.world),
        "answer": answer,
    }


def task_from_dict(data: dict) -> Task:
    """Build a task from a dict made by task_to_dict."""
    if data["answer"] is None:
        answer = None
    else:
        answer = Solution(data["answer"]["order_ids"], data["answer"]["finish_minute"])
    return Task(data["task_id"], data["difficulty"], world_from_dict(data["world"]), answer)


def save_tasks(tasks: list[Task], path: Path) -> None:
    """Write the tasks to a JSON Lines file, one task per line, replacing the file."""
    with path.open("w", encoding="utf-8") as file:
        for task in tasks:
            file.write(json.dumps(task_to_dict(task)) + "\n")


def load_tasks(path: Path) -> list[Task]:
    """Read the tasks from a JSON Lines file made by save_tasks."""
    tasks = []
    with path.open(encoding="utf-8") as file:
        for line in file:
            tasks.append(task_from_dict(json.loads(line)))
    return tasks
