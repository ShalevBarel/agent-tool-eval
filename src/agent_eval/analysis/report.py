"""The results of every configuration, and paired comparisons between them."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from agent_eval.analysis.bootstrap import (
    Estimate,
    percentile_estimate,
    ratio_estimate,
    resample_indices,
    resampled_ratios,
)
from agent_eval.analysis.compare import holm, sign_flip_test, tasks_needed
from agent_eval.analysis.tables import task_totals
from agent_eval.env.generate import LEVELS

N_RESAMPLES = 10_000
N_FLIPS = 10_000
LEVEL = 0.95

# The planned comparisons, as (a, b): what changing configuration b into a does.
# The first is the research question: is a strong model at low effort cheaper per
# correct answer than a cheap model at high effort? The others change one thing.
COMPARISONS = [
    ("sonnet-low", "haiku-high"),
    ("haiku-high", "haiku-medium"),
    ("sonnet-high", "sonnet-low"),
    ("sonnet-high", "haiku-high"),
    ("haiku-medium-improved-tools", "haiku-medium"),
]

# The difference in success rate that the power analysis asks about: 5 points.
DETECTABLE_DIFFERENCE = 0.05


@dataclass(frozen=True)
class Comparison:
    """Configuration a against configuration b, over the same tasks.

    Attributes:
        a: The first configuration's name.
        b: The second configuration's name.
        success_difference: a's success rate minus b's, with a paired interval.
        cost_ratio: a's cost per correct answer over b's, with a paired interval.
        p_value: The sign-flip test's p-value for the difference in success.
        holm_p_value: The p-value after Holm's adjustment over all the comparisons made.
        tasks_needed: How many tasks would detect a 5-point difference in success
            with power 0.8 at level 0.05, given how much a and b differ task by task.
    """

    a: str
    b: str
    success_difference: Estimate
    cost_ratio: Estimate
    p_value: float
    holm_p_value: float
    tasks_needed: int


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


def estimates_by_difficulty(
    runs: pd.DataFrame, n_resamples: int = N_RESAMPLES, seed: int = 0
) -> dict[str, dict[str, Estimate]]:
    """Return each configuration's success rate on each difficulty, with intervals.

    The intervals of a difficulty resample its own tasks, and every
    configuration gets the same resamples. Difficulties come easiest first,
    and configurations in alphabetical order.

    Example:
        {"haiku-high": {"easy": Estimate(1.0, 1.0, 1.0),
                        "medium": Estimate(0.95, 0.91, 0.98),
                        "hard": Estimate(0.83, 0.75, 0.91)}}

    Raises:
        ValueError: If two configurations ran different tasks.
    """
    totals = task_totals(runs)
    tasks = shared_tasks(totals)
    rng = np.random.default_rng(seed)
    results = {config: {} for config in sorted(totals["config"].unique())}
    for level in LEVELS:
        count = int((tasks["difficulty"] == level).sum())
        if count == 0:
            continue
        indices = rng.choice(count, size=(n_resamples, count))
        for config, rows in totals.groupby("config"):
            rows = rows[rows["difficulty"] == level]
            results[config][level] = ratio_estimate(
                rows["correct"].to_numpy(), rows["runs"].to_numpy(), indices, LEVEL
            )
    return results


def comparisons(
    runs: pd.DataFrame,
    pairs: list[tuple[str, str]] = COMPARISONS,
    n_resamples: int = N_RESAMPLES,
    n_flips: int = N_FLIPS,
    seed: int = 0,
) -> list[Comparison]:
    """Return paired comparisons between configurations, in the order of pairs.

    Pairs whose configurations are not both in runs are left out, and Holm's
    adjustment is over the comparisons that remain. The intervals use the
    same resamples of the tasks as estimates, and a resampled task brings
    the runs of both configurations. The test compares each task's share of
    correct runs.

    Raises:
        ValueError: If two configurations ran different tasks.
    """
    totals = task_totals(runs)
    tasks = shared_tasks(totals)
    by_config = {config: rows for config, rows in totals.groupby("config")}
    pairs = [(a, b) for a, b in pairs if a in by_config and b in by_config]
    if not pairs:
        return []
    rng = np.random.default_rng(seed)
    indices = resample_indices(tasks["difficulty"].to_numpy(), n_resamples, rng)

    scores = {
        config: (rows["correct"] / rows["runs"]).to_numpy() for config, rows in by_config.items()
    }
    differences = [scores[a] - scores[b] for a, b in pairs]
    p_values = [sign_flip_test(task_differences, n_flips, rng) for task_differences in differences]

    results = []
    for (a, b), task_differences, p_value, holm_p_value in zip(
        pairs, differences, p_values, holm(p_values)
    ):
        first, second = by_config[a], by_config[b]
        results.append(
            Comparison(
                a,
                b,
                success_difference=_paired(first, second, "correct", "runs", np.subtract, indices),
                cost_ratio=_paired(first, second, "cost_usd", "correct", np.divide, indices),
                p_value=p_value,
                holm_p_value=holm_p_value,
                tasks_needed=tasks_needed(task_differences.std(ddof=1), DETECTABLE_DIFFERENCE),
            )
        )
    return results


def _paired(
    first: pd.DataFrame,
    second: pd.DataFrame,
    numerator: str,
    denominator: str,
    combine: np.ufunc,
    indices: np.ndarray,
) -> Estimate:
    """Return combine(first's ratio, second's ratio), with an interval from the same resamples.

    A configuration's ratio is the sum of its numerator column over the sum
    of its denominator column, such as correct answers over runs.
    """
    values, resampled = [], []
    for rows in (first, second):
        numerators, denominators = rows[numerator].to_numpy(), rows[denominator].to_numpy()
        values.append(numerators.sum() / denominators.sum())
        resampled.append(resampled_ratios(numerators, denominators, indices))
    return percentile_estimate(combine(*values), combine(*resampled), LEVEL)


def report(
    runs: pd.DataFrame,
    n_resamples: int = N_RESAMPLES,
    seed: int = 0,
    pairs: list[tuple[str, str]] = COMPARISONS,
) -> str:
    """Return a table of every configuration's results, its success by difficulty, and comparisons.

    The comparisons are those of pairs whose configurations are in runs.

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
    found = comparisons(runs, pairs, n_resamples, seed=seed)
    if found:
        lines += ["", *_comparison_lines(found)]
    return "\n".join(lines)


def _comparison_lines(found: list[Comparison]) -> list[str]:
    """Return the tables of paired comparisons: success first, then cost."""
    names = [f"{comparison.a} vs {comparison.b}" for comparison in found]
    width = max(len("a vs b"), *(len(name) for name in names))
    interval = f"{LEVEL:.0%} interval"
    lines = [
        f"{'a vs b':{width}}  {'a minus b':>9}  {interval:14}  {'p':>6}  {'Holm p':>6}"
        "  tasks for 5 pts"
    ]
    for name, comparison in zip(names, found):
        difference = comparison.success_difference
        points = f"{100 * difference.value:+.1f} pts"
        points_interval = f"{100 * difference.low:+.1f} to {100 * difference.high:+.1f}"
        lines.append(
            f"{name:{width}}  {points:>9}  {points_interval:14}  {comparison.p_value:6.4f}"
            f"  {comparison.holm_p_value:6.4f}  {comparison.tasks_needed:15}"
        )

    lines += ["", f"{'a vs b':{width}}  cost per correct, a over b  {interval}"]
    for name, comparison in zip(names, found):
        ratio = comparison.cost_ratio
        lines.append(
            f"{name:{width}}  {f'{ratio.value:.2f}x':>26}  {ratio.low:.2f}x to {ratio.high:.2f}x"
        )

    lines += [
        "",
        "Comparisons are paired: both configurations ran the same tasks, and the intervals",
        "resample those tasks. p: a sign-flip test of the per-task differences in success,",
        f"{N_FLIPS} flips. Holm p: adjusted over the {len(found)} comparisons. Tasks for 5 pts:",
        "tasks needed to detect a 5-point difference at level 0.05 with power 0.8.",
    ]
    return lines
