"""Run logs as tables: one row per run, or one row per task of a configuration."""

from pathlib import Path

import pandas as pd

from agent_eval.experiment.run_log import load_records


def load_runs(paths: list[Path]) -> pd.DataFrame:
    """Return the runs of several logs as one table, with one row per run.

    Each row has the fields of its record, plus "config", the name of its
    log file without the extension, such as "haiku-high".
    """
    frames = [pd.DataFrame(load_records(path)).assign(config=path.stem) for path in paths]
    return pd.concat(frames, ignore_index=True)


def task_totals(runs: pd.DataFrame) -> pd.DataFrame:
    """Return one row per configuration and task, with totals over the task's runs.

    The columns are config, task_id, difficulty, runs (how many runs the
    task has), correct (how many of them are correct) and cost_usd (what
    they cost together). The rows are sorted by configuration, then by task.

    Example:
        config      task_id   difficulty  runs  correct  cost_usd
        haiku-high  task-000  easy           3        3    0.0012
        haiku-high  task-001  easy           3        2    0.0019
    """
    return (
        runs.assign(correct=runs["verdict"] == "correct")
        .groupby(["config", "task_id"], as_index=False)
        .agg(
            difficulty=("difficulty", "first"),
            runs=("verdict", "size"),
            correct=("correct", "sum"),
            cost_usd=("cost_usd", "sum"),
        )
    )
