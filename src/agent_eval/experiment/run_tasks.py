"""Run the agent on many tasks, each one or more times, and log every run.

Each run is graded, priced and added to the log as soon as it ends. Runs
that the log already has are skipped, so a command that stopped for any
reason picks up where it left off when it starts again.
"""

import random
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from anthropic import Anthropic

from agent_eval.agent.loop import AgentConfig, run_task
from agent_eval.env.generate import LEVELS
from agent_eval.env.tasks import Task
from agent_eval.experiment.cost import cost_usd
from agent_eval.experiment.grading import VERDICTS
from agent_eval.experiment.run_log import (
    append_record,
    load_records,
    run_record,
    runs_to_do,
    spent_usd,
    summarize,
)


def choose_tasks(tasks: list[Task], sample: int | None, seed: int) -> list[Task]:
    """Return a random sample of the tasks, or all of them if sample is None, sorted by id.

    The same seed always gives the same sample.
    """
    if sample is not None:
        tasks = random.Random(seed).sample(tasks, sample)
    return sorted(tasks, key=lambda task: task.task_id)


def check_log_matches(records: list[dict], config: AgentConfig) -> None:
    """Make sure every run in a log used this configuration.

    Raises:
        ValueError: If some run used another model, effort or tool descriptions.
    """
    wanted = (config.model, config.effort, config.tool_specs)
    for record in records:
        found = (record["model"], record["effort"], record["tool_specs"])
        if found != wanted:
            raise ValueError(f"the log has runs of {found}, not {wanted}. Use another log file.")


def run_tasks(
    client: Anthropic,
    tasks: list[Task],
    config: AgentConfig,
    repeats: int,
    budget_usd: float,
    log_path: Path,
    workers: int = 4,
) -> list[dict]:
    """Run the agent on every task repeats times, and log every run as it ends.

    Runs that the log already has are skipped. The rest run in batches of
    `workers` at a time, each in its own thread, since a run mostly waits
    for the API. Before each batch, the command stops if the runs in the log
    already cost budget_usd or more, so it can go over by one batch at most.

    If a run fails with an error, for example when the API is still
    overloaded after the client's retries, the other runs of its batch are
    still logged, and then the error is raised.

    Returns every record in the log, old and new.

    Raises:
        ValueError: If the log has runs of another configuration.
    """
    records = load_records(log_path) if log_path.exists() else []
    check_log_matches(records, config)
    to_do = runs_to_do(tasks, repeats, records)
    print(f"{len(to_do)} runs to do, {len(records)} already in {log_path}")
    log_path.parent.mkdir(parents=True, exist_ok=True)

    finished = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for start in range(0, len(to_do), workers):
            spent = spent_usd(records)
            if spent >= budget_usd:
                print(f"stopped: the log's runs cost ${spent:.4f}, the budget is ${budget_usd:.2f}")
                break
            batch = to_do[start : start + workers]
            futures = [pool.submit(run_task, client, task, config) for task, _ in batch]
            error = None
            for (task, repeat), future in zip(batch, futures):
                try:
                    result = future.result()
                except Exception as exception:
                    error = error or exception
                    continue
                record = run_record(task, config, result, repeat)
                append_record(log_path, record)
                records.append(record)
                finished += 1
                print(
                    f"{finished:4}/{len(to_do)}  {task.task_id}  repeat {repeat}"
                    f"  {task.difficulty:6}  {record['verdict']:15}  {result.turns:2} turns"
                    f"  {result.seconds:5.1f} s  ${record['cost_usd']:.4f}"
                )
            if error is not None:
                raise error
    return records


def cost_without_cache(record: dict) -> float:
    """Return what a run would have cost if every input token were read without the cache."""
    input_tokens = (
        record["input_tokens"] + record["cache_write_tokens"] + record["cache_read_tokens"]
    )
    return cost_usd(record["model"], input_tokens, record["output_tokens"])


def show_summary(records: list[dict]) -> None:
    """Print the runs, correct answers and cost by difficulty and overall, then more detail.

    After the table come the verdicts, how the runs ended, and what the
    cache saved.
    """
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
    if not records:
        return

    print()
    verdicts = Counter(record["verdict"] for record in records)
    for verdict in VERDICTS:
        if verdicts[verdict]:
            print(f"{verdict:16} {verdicts[verdict]}")

    print()
    outcomes = Counter(record["outcome"] for record in records)
    print("ended by     " + ", ".join(f"{name} {count}" for name, count in outcomes.most_common()))
    spent = spent_usd(records)
    without = sum(cost_without_cache(record) for record in records)
    print(f"per run      ${spent / len(records):.4f} on average")
    if without > 0:
        saved = without - spent
        print(f"cache saved  ${saved:.4f}, {saved / without:.0%} of the cost without it")
