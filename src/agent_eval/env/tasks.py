"""Delivery tasks, and how they are stored as JSON Lines: one task per line."""

from dataclasses import dataclass
from pathlib import Path

from agent_eval.env.world import World
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
    raise NotImplementedError


def world_from_dict(data: dict) -> World:
    """Build a world from a dict made by world_to_dict.

    Raises:
        ValueError: If the dict describes an invalid world, such as a road
            to an unknown place.
    """
    raise NotImplementedError


def task_to_dict(task: Task) -> dict:
    """Return the task as a dict that json can write. A missing answer becomes None."""
    raise NotImplementedError


def task_from_dict(data: dict) -> Task:
    """Build a task from a dict made by task_to_dict."""
    raise NotImplementedError


def save_tasks(tasks: list[Task], path: Path) -> None:
    """Write the tasks to a JSON Lines file, one task per line, replacing the file."""
    raise NotImplementedError


def load_tasks(path: Path) -> list[Task]:
    """Read the tasks from a JSON Lines file made by save_tasks."""
    raise NotImplementedError
