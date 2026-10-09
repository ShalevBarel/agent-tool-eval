"""Seeded generation of delivery tasks, each with its exact answer.

The same seed always gives the same tasks, byte for byte. Run it with:

    uv run python -m agent_eval.env.generate --seed 0
"""

import argparse
import random
from pathlib import Path

from agent_eval.env.tasks import Task, save_tasks
from agent_eval.env.world import World

# The number of orders at each difficulty, fewest to most.
LEVELS = {"easy": (1, 1), "medium": (2, 3), "hard": (4, 5)}

MIN_PLACES, MAX_PLACES = 6, 10
MIN_MINUTES, MAX_MINUTES = 5, 25
TWO_WAY_CHANCE = 0.6
CLOSED_CHANCE = 0.1
LATEST_OPEN = 150
MIN_WIDTH, MAX_WIDTH = 30, 90


def random_city(rng: random.Random) -> World:
    """A random city of numbered places. The courier starts at p0, at minute 0.

    A random tree of two-way streets links all the places first, so every
    place can reach every other while all roads are open. Then come as many
    extra roads as places, some of them one-way, and about a tenth of all
    roads are closed for the day, which can cut places off.
    """
    world = World()
    places = [f"p{i}" for i in range(rng.randint(MIN_PLACES, MAX_PLACES))]
    for place in places:
        world.add_place(place)

    for i in range(1, len(places)):
        neighbor = places[rng.randrange(i)]
        minutes = rng.randint(MIN_MINUTES, MAX_MINUTES)
        world.add_road(places[i], neighbor, minutes)
        world.add_road(neighbor, places[i], minutes)

    for _ in range(len(places)):
        start, end = rng.sample(places, 2)
        if end in world.roads[start]:
            continue
        minutes = rng.randint(MIN_MINUTES, MAX_MINUTES)
        world.add_road(start, end, minutes)
        if rng.random() < TWO_WAY_CHANCE and start not in world.roads[end]:
            world.add_road(end, start, minutes)

    for start in places:
        for end in world.roads[start]:
            if rng.random() < CLOSED_CHANCE:
                world.close_road(start, end)

    world.set_start("p0", 0)
    return world


def add_random_orders(rng: random.Random, world: World, count: int) -> None:
    """Add orders o1 to o{count} at distinct places other than the start.

    Each window opens at a random minute up to LATEST_OPEN and stays open
    MIN_WIDTH to MAX_WIDTH minutes.
    """
    raise NotImplementedError


def each_order_alone_on_time(world: World) -> bool:
    """Return whether every order could be delivered on time if it were the only one."""
    raise NotImplementedError


def generate(seed: int, per_level: int, impossible_share: float = 0.15) -> list[Task]:
    """Generate per_level tasks at each difficulty, easy first.

    At each difficulty, round(per_level * impossible_share) of the tasks have
    no solution. A task with no solution and several orders is kept only if
    each of its orders alone could be delivered on time, so the problem lies
    in combining them. Random tasks are drawn until every count is met.
    """
    raise NotImplementedError


def main(argv: list[str] | None = None) -> None:
    """Generate tasks and write them to a JSON Lines file."""
    parser = argparse.ArgumentParser(description="Generate delivery tasks with exact answers.")
    parser.add_argument("--seed", type=int, default=0, help="random seed (default: 0)")
    parser.add_argument(
        "--per-level", type=int, default=50, help="tasks at each difficulty (default: 50)"
    )
    parser.add_argument("--out", type=Path, default=Path("data/tasks.jsonl"), help="output file")
    args = parser.parse_args(argv)

    tasks = generate(args.seed, args.per_level)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    save_tasks(tasks, args.out)

    for difficulty in LEVELS:
        level = [task for task in tasks if task.difficulty == difficulty]
        impossible = sum(1 for task in level if task.answer is None)
        print(f"{difficulty:6}  {len(level)} tasks, {impossible} with no solution")
    print(f"wrote {len(tasks)} tasks to {args.out}")


if __name__ == "__main__":
    main()
