"""The results of every configuration, with bootstrap confidence intervals."""

import numpy as np
import pandas as pd

from agent_eval.analysis.bootstrap import Estimate, ratio_estimate, resample_indices
from agent_eval.analysis.tables import task_totals
from agent_eval.env.generate import LEVELS

N_RESAMPLES = 10_000
LEVEL = 0.95


def shared_tasks(totals: pd.DataFrame) -> pd.DataFrame:
    """Return the tasks and their difficulty, after checking every configuration ran them all.

    Raises:
        ValueError: If two configurations ran different tasks.
    """
    tasks = None
    for config, rows in totals.groupby("config"):
        if tasks is None:
            first, tasks = config, rows[["task_id", "difficulty"]].reset_index(drop=True)
        elif rows["task_id"].tolist() != tasks["task_id"].tolist():
            raise ValueError(f"{config} and {first} ran different tasks")
    if tasks is None:
        raise ValueError("no runs")
    return tasks


def estimates(
    runs: pd.DataFrame, n_resamples: int = N_RESAMPLES, seed: int = 0
) -> dict[str, dict[str, Estimate]]:
    """Return each configuration's success rate and cost per correct answer, with intervals.

    The intervals resample tasks within each difficulty, and every
    configuration gets the same resamples. Configurations come best first.

    Example:
        {"haiku-high": {"success": Estimate(0.93, 0.90, 0.95),
                        "cost_per_correct": Estimate(0.0051, 0.0047, 0.0056)}}

    Raises:
        ValueError: If two configurations ran different tasks.
    """
    totals = task_totals(runs)
    tasks = shared_tasks(totals)
    rng = np.random.default_rng(seed)
    indices = resample_indices(tasks["difficulty"].to_numpy(), n_resamples, rng)
    results = {}
    for config, rows in totals.groupby("config"):
        correct = rows["correct"].to_numpy()
        results[config] = {
            "success": ratio_estimate(correct, rows["runs"].to_numpy(), indices, LEVEL),
            "cost_per_correct": ratio_estimate(
                rows["cost_usd"].to_numpy(), correct, indices, LEVEL
            ),
        }
    return dict(sorted(results.items(), key=lambda item: -item[1]["success"].value))


def report(runs: pd.DataFrame, n_resamples: int = N_RESAMPLES, seed: int = 0) -> str:
    """Return a table of every configuration's results, then its success by difficulty.

    Raises:
        ValueError: If two configurations ran different tasks.
    """
    results = estimates(runs, n_resamples, seed)
    width = max(len("config"), *(len(config) for config in results))
    interval = f"{LEVEL:.0%} interval"
    lines = [f"{'config':{width}}  runs  success  {interval:15}  per correct  {interval}"]
    totals = task_totals(runs)
    sums = totals.groupby("config")[["runs", "correct"]].sum()
    for config, result in results.items():
        success, cost = result["success"], result["cost_per_correct"]
        success_interval = f"{success.low:.1%} to {success.high:.1%}"
        cost_value = f"${cost.value:.4f}"
        lines.append(
            f"{config:{width}}  {sums.loc[config, 'runs']:4}  {success.value:7.1%}"
            f"  {success_interval:15}  {cost_value:>11}  ${cost.low:.4f} to ${cost.high:.4f}"
        )

    by_difficulty = totals.groupby(["config", "difficulty"])[["runs", "correct"]].sum()
    shares = (by_difficulty["correct"] / by_difficulty["runs"]).unstack()
    levels = [level for level in LEVELS if level in shares.columns]
    lines += ["", f"{'success by difficulty':{width}}" + "".join(f"  {lvl:>6}" for lvl in levels)]
    for config in results:
        row = "".join(f"  {shares.loc[config, lvl]:6.0%}" for lvl in levels)
        lines.append(f"{config:{width}}{row}")

    lines += [
        "",
        f"Intervals: {n_resamples} bootstrap resamples of the tasks within each difficulty,"
        f" seed {seed}.",
    ]
    return "\n".join(lines)
