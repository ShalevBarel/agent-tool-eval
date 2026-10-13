"""Run the agent on many tasks and log every run. It calls the API, so it costs money.

    uv run python -m agent_eval.experiment.run_tasks --sample 20 --out runs/pilot.jsonl

Each run is graded, priced and added to the log as soon as it ends, so a
crash loses only the run in progress. The API key is read from the
ANTHROPIC_API_KEY environment variable.
"""

import argparse
import random
from collections import Counter
from pathlib import Path

import anthropic

from agent_eval.agent.loop import AgentConfig, run_task
from agent_eval.env.generate import LEVELS
from agent_eval.env.tasks import Task, load_tasks
from agent_eval.experiment.grading import VERDICTS
from agent_eval.experiment.run_log import append_record, run_record, summarize


def choose_tasks(tasks: list[Task], sample: int | None, seed: int) -> list[Task]:
    """Return a random sample of the tasks, or all of them if sample is None, sorted by id.

    The same seed always gives the same sample.
    """
    if sample is not None:
        tasks = random.Random(seed).sample(tasks, sample)
    return sorted(tasks, key=lambda task: task.task_id)


def show_summary(records: list[dict]) -> None:
    """Print the runs, correct answers and cost by difficulty and overall, then the verdicts."""
    summary = summarize(records)
    print("          runs  correct  share      cost  per correct")
    for group in [*LEVELS, "all"]:
        if group not in summary:
            continue
        counts = summary[group]
        share = counts["correct"] / counts["runs"]
        cost = f"${counts['cost_usd']:.4f}"
        if counts["correct"]:
            per_correct = f"${counts['cost_usd'] / counts['correct']:.4f}"
        else:
            per_correct = "-"
        print(
            f"{group:8} {counts['runs']:5} {counts['correct']:8} {share:6.0%}"
            f" {cost:>9} {per_correct:>12}"
        )

    print()
    verdicts = Counter(record["verdict"] for record in records)
    for verdict in VERDICTS:
        if verdicts[verdict]:
            print(f"{verdict:16} {verdicts[verdict]}")


def main(argv: list[str] | None = None) -> None:
    """Run the agent on the tasks chosen on the command line, logging each run."""
    parser = argparse.ArgumentParser(description="Run the agent on many tasks and log every run.")
    parser.add_argument("--out", type=Path, required=True, help="the run log, a new .jsonl file")
    parser.add_argument("--sample", type=int, help="run on this many random tasks (default: all)")
    parser.add_argument("--seed", type=int, default=0, help="seed of the sample (default: 0)")
    parser.add_argument("--model", default="claude-haiku-5-5", help="default: claude-haiku-5-5")
    parser.add_argument("--effort", default="medium", help="default: medium")
    parser.add_argument("--tasks-file", type=Path, default=Path("data/tasks.jsonl"))
    args = parser.parse_args(argv)

    if args.out.exists():
        parser.error(f"{args.out} already exists. Choose a new file, so runs never mix.")
    tasks = load_tasks(args.tasks_file)
    if args.sample is not None and not 1 <= args.sample <= len(tasks):
        parser.error(f"--sample must be between 1 and {len(tasks)}")
    tasks = choose_tasks(tasks, args.sample, args.seed)

    config = AgentConfig(args.model, args.effort)
    # Many runs in a row can hit the account's per-minute limits. On a "too many
    # requests" or "overloaded" error, the client waits and tries again.
    client = anthropic.Anthropic(max_retries=10)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    records = []
    for number, task in enumerate(tasks, start=1):
        result = run_task(client, task, config)
        record = run_record(task, config, result)
        append_record(args.out, record)
        records.append(record)
        print(
            f"{number:3}/{len(tasks)}  {task.task_id}  {task.difficulty:6}  {record['verdict']:15}"
            f"  {result.turns:2} turns  {result.seconds:5.1f} s  ${record['cost_usd']:.4f}"
        )

    print()
    show_summary(records)


if __name__ == "__main__":
    main()
