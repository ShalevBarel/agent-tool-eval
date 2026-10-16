"""Run the agent on one task and show what it did. It calls the API, so it costs money.

    uv run python -m agent_eval.agent.run --task task-100
    uv run python -m agent_eval.agent.run --task task-100 --config configs/sonnet-low.toml

The API key is read from the ANTHROPIC_API_KEY environment variable.
"""

import argparse
import json
from pathlib import Path

import anthropic

from agent_eval.agent.loop import REQUEST_TIMEOUT_SECONDS, RunResult, run_task
from agent_eval.env.tasks import Task, load_tasks
from agent_eval.env.tools import check_schedule
from agent_eval.experiment.configs import load_config
from agent_eval.experiment.cost import cost_usd
from agent_eval.experiment.grading import grade

OUTPUT_WIDTH = 100


def show(task: Task, result: RunResult) -> None:
    """Print every tool call of a run, then how it ended next to the exact answer."""
    for call in result.tool_calls:
        output = call.output
        if len(output) > OUTPUT_WIDTH:
            output = output[:OUTPUT_WIDTH] + " ..."
        print(f"turn {call.turn:2}  {call.name} {json.dumps(call.tool_input)}")
        print(f"         {'error: ' if call.is_error else ''}{output}")
    print()
    print(f"outcome  {result.outcome}, after {result.turns} turns and {result.seconds:.1f} s")
    print(
        f"tokens   {result.input_tokens} in, {result.output_tokens} out, "
        f"{result.cache_write_tokens} written to the cache, {result.cache_read_tokens} read from it"
    )

    if result.answer is None:
        print("agent    no answer")
    elif not result.answer.solvable:
        print("agent    no solution")
    else:
        order = " ".join(result.answer.order_ids)
        schedule = check_schedule(task.world, result.answer.order_ids)
        if schedule["on_time"]:
            print(f"agent    {order}, last delivery at minute {schedule['finish_minute']}")
        else:
            print(f"agent    {order}, not on time: {schedule['problem']}")
    if task.answer is None:
        print("exact    no solution")
    else:
        order = " ".join(task.answer.order_ids)
        print(f"exact    {order}, last delivery at minute {task.answer.finish_minute}")


def main(argv: list[str] | None = None) -> None:
    """Run the agent on the task named on the command line, then grade and price the run."""
    parser = argparse.ArgumentParser(description="Run the agent on one task.")
    parser.add_argument("--task", required=True, help="task id, such as task-100")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/haiku-medium.toml"),
        help="a configuration file (default: configs/haiku-medium.toml)",
    )
    parser.add_argument("--tasks-file", type=Path, default=Path("data/tasks.jsonl"))
    args = parser.parse_args(argv)

    try:
        config = load_config(args.config)
    except ValueError as error:
        parser.error(str(error))
    tasks = {task.task_id: task for task in load_tasks(args.tasks_file)}
    if args.task not in tasks:
        parser.error(f"no task with id {args.task}")
    task = tasks[args.task]
    print(f"{task.task_id}, {task.difficulty}, orders {' '.join(task.world.orders)}")
    print()

    result = run_task(anthropic.Anthropic(timeout=REQUEST_TIMEOUT_SECONDS), task, config)
    show(task, result)
    print(f"verdict  {grade(task, result.answer)}")
    cost = cost_usd(
        config.model,
        result.input_tokens,
        result.output_tokens,
        result.cache_write_tokens,
        result.cache_read_tokens,
    )
    print(f"cost     ${cost:.4f}")


if __name__ == "__main__":
    main()
