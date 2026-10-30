"""Plots of the results: success against cost, and success by difficulty."""

import math
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from matplotlib.colors import to_rgba
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator, PercentFormatter

from agent_eval.analysis.bootstrap import Estimate
from agent_eval.analysis.report import LEVEL, N_RESAMPLES, estimates, estimates_by_difficulty

# One color per model, given to the models in alphabetical order. These three
# stay apart for color-blind readers too.
MODEL_COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]

# Text colors, darkest first.
INK = "#0b0b0b"
SECONDARY_INK = "#52514e"
MUTED_INK = "#898781"

# The look of every plot: small dark text, a light grid behind the data, no box.
STYLE = {
    "font.size": 9,
    "text.color": INK,
    "axes.titlesize": 10,
    "axes.titlecolor": INK,
    "axes.labelcolor": SECONDARY_INK,
    "axes.edgecolor": "#c3c2b7",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": "#e1e0d9",
    "grid.linewidth": 0.8,
    "xtick.color": MUTED_INK,
    "ytick.color": MUTED_INK,
    "xtick.labelcolor": SECONDARY_INK,
    "ytick.labelcolor": SECONDARY_INK,
    "figure.facecolor": "white",
    "legend.frameon": False,
}

# Dots per inch of the saved files, sharp on high-resolution screens.
DPI = 200


def save_plots(
    runs: pd.DataFrame, out: Path, n_resamples: int = N_RESAMPLES, seed: int = 0
) -> list[Path]:
    """Draw both plots of the runs into the folder out, as PNG files, and return their paths.

    The intervals come from the same resamples as the report's, so the two agree.

    Raises:
        ValueError: If two configurations ran different tasks, if a
            configuration ran more than one model, or if there are more
            models than colors.
    """
    results = estimates(runs, n_resamples, seed)
    by_difficulty = estimates_by_difficulty(runs, n_resamples, seed)
    models = config_models(runs)
    figures = {
        "cost-vs-success.png": cost_vs_success(results, models),
        "success-by-difficulty.png": success_by_difficulty(
            {config: by_difficulty[config] for config in results}, models
        ),
    }
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, figure in figures.items():
        path = out / name
        figure.savefig(path, dpi=DPI)
        paths.append(path)
    return paths


def config_models(runs: pd.DataFrame) -> dict[str, str]:
    """Return the model of every configuration in runs.

    Raises:
        ValueError: If a configuration's runs used more than one model.
    """
    models = {}
    for config, names in runs.groupby("config")["model"].unique().items():
        if len(names) > 1:
            raise ValueError(f"{config} ran more than one model: {', '.join(sorted(names))}")
        models[config] = names[0]
    return models


def model_colors(models: dict[str, str]) -> dict[str, str]:
    """Return a color for every model, given each configuration's model.

    Raises:
        ValueError: If there are more models than colors.
    """
    names = sorted(set(models.values()))
    if len(names) > len(MODEL_COLORS):
        raise ValueError(f"{len(names)} models, but only {len(MODEL_COLORS)} colors")
    return dict(zip(names, MODEL_COLORS))


def cost_vs_success(results: dict[str, dict[str, Estimate]], models: dict[str, str]) -> Figure:
    """Return a plot of each configuration's success rate against its cost per correct answer.

    results is what estimates returns, and models gives each configuration's
    model, which colors its point. The bars are the confidence intervals.
    Costs can differ tenfold, so the cost axis is logarithmic: equal
    distances are equal ratios.
    """
    colors = model_colors(models)
    with matplotlib.rc_context(STYLE):
        figure = Figure(figsize=(7, 4.4), layout="constrained")
        axes = figure.subplots()
        axes.set_xscale("log")
        for config, result in results.items():
            success, cost = result["success"], result["cost_per_correct"]
            color = colors[models[config]]
            axes.errorbar(
                cost.value,
                100 * success.value,
                xerr=[[cost.value - cost.low], [cost.high - cost.value]],
                yerr=[
                    [100 * (success.value - success.low)],
                    [100 * (success.high - success.value)],
                ],
                fmt="o",
                color=color,
                ecolor=to_rgba(color, 0.55),
                elinewidth=1.5,
                markersize=8,
                markeredgecolor="white",
                markeredgewidth=1.5,
            )

        lows = [result["cost_per_correct"].low for result in results.values()]
        highs = [result["cost_per_correct"].high for result in results.values()]
        left, right = min(lows) / 2, max(highs) * 1.6
        axes.set_xlim(left, right)
        # The success axis starts at a multiple of 10 at least 5 points below
        # the lowest interval, and ends at 100%.
        lowest = min(100 * result["success"].low for result in results.values())
        axes.set_ylim(10 * math.floor((lowest - 5) / 10), 100)

        # Where each point sits, as fractions of the plot's width and height.
        points = [
            axes.transAxes.inverted().transform(
                axes.transData.transform(
                    (result["cost_per_correct"].value, 100 * result["success"].value)
                )
            )
            for result in results.values()
        ]
        for (config, result), side in zip(results.items(), _label_sides(points)):
            success, cost = result["success"], result["cost_per_correct"]
            # A label starts a little past the end of the point's interval.
            x = cost.high * 1.08 if side == "right" else cost.low / 1.08
            axes.text(
                x,
                100 * success.value,
                config,
                ha="left" if side == "right" else "right",
                va="center",
                color=INK,
            )

        axes.xaxis.set_major_locator(FixedLocator(_one_two_five(left, right)))
        axes.xaxis.set_minor_locator(NullLocator())
        axes.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"${value:g}"))
        axes.yaxis.set_major_formatter(PercentFormatter(decimals=0))
        axes.set_xlabel("Cost per correct answer (log scale)")
        axes.set_ylabel("Success rate")
        axes.annotate(
            "better",
            xy=(0.02, 0.98),
            xytext=(0.09, 0.88),
            xycoords="axes fraction",
            color=MUTED_INK,
            arrowprops={"arrowstyle": "->", "color": MUTED_INK},
        )
        axes.legend(handles=_model_handles(colors), loc="lower right")
        axes.set_title(
            f"Success rate against cost per correct answer, with {LEVEL:.0%} bootstrap intervals",
            loc="left",
        )
    return figure


def success_by_difficulty(
    by_difficulty: dict[str, dict[str, Estimate]], models: dict[str, str]
) -> Figure:
    """Return a panel for every difficulty, with a bar for each configuration's success rate.

    by_difficulty is what estimates_by_difficulty returns, with the
    configurations in the order to list them, top to bottom. models gives
    each configuration's model, which colors its bars. The lines on the bars
    are the confidence intervals.
    """
    colors = model_colors(models)
    configs = list(by_difficulty)
    levels = list(by_difficulty[configs[0]])
    positions = np.arange(len(configs))
    with matplotlib.rc_context(STYLE):
        figure = Figure(figsize=(8, 0.45 * len(configs) + 1.4), layout="constrained")
        figure.get_layout_engine().set(wspace=0.08)
        panels = np.atleast_1d(figure.subplots(1, len(levels), sharex=True, sharey=True))
        for panel, level in zip(panels, levels):
            found = [by_difficulty[config][level] for config in configs]
            values = np.array([100 * estimate.value for estimate in found])
            panel.barh(
                positions,
                values,
                height=0.5,
                color=[colors[models[config]] for config in configs],
            )
            panel.errorbar(
                values,
                positions,
                xerr=[
                    values - [100 * estimate.low for estimate in found],
                    [100 * estimate.high for estimate in found] - values,
                ],
                fmt="none",
                ecolor=INK,
                elinewidth=1,
            )
            # A bar's value goes inside it, unless the bar is too short to hold it.
            for position, value in zip(positions, values):
                if value >= 15:
                    panel.text(3, position, f"{value:.0f}%", va="center", color="white", size=8)
                else:
                    panel.text(value + 3, position, f"{value:.0f}%", va="center", size=8)
            panel.set_title(level.capitalize())
            panel.grid(axis="y", visible=False)
            panel.tick_params(axis="y", length=0, labelcolor=INK)
        panels[0].set_yticks(positions, configs)
        panels[0].invert_yaxis()
        panels[0].set_xlim(0, 100)
        panels[0].xaxis.set_major_locator(FixedLocator([0, 50, 100]))
        panels[0].xaxis.set_major_formatter(PercentFormatter(decimals=0))
        figure.suptitle(
            f"Success rate by difficulty, with {LEVEL:.0%} bootstrap intervals",
            x=0.01,
            ha="left",
            size=10,
        )
        figure.legend(handles=_model_handles(colors), loc="outside lower center", ncols=len(colors))
    return figure


def _model_handles(colors: dict[str, str]) -> list[Line2D]:
    """Return legend entries that show each model's color."""
    return [
        Line2D([], [], marker="o", linestyle="none", color=color, markersize=8, label=model)
        for model, color in colors.items()
    ]


def _label_sides(points: list[np.ndarray]) -> list[str]:
    """Return the side of each point's label, "left" or "right".

    points are positions as fractions of the plot's width and height. A
    label goes to the right of its point, unless the point is in the right
    part of the plot, or another point sits close to it on the right.
    """
    sides = []
    for i, (x, y) in enumerate(points):
        crowded = any(
            0 < other_x - x < 0.2 and abs(other_y - y) < 0.08
            for j, (other_x, other_y) in enumerate(points)
            if j != i
        )
        sides.append("left" if x > 0.7 or crowded else "right")
    return sides


def _one_two_five(low: float, high: float) -> list[float]:
    """Return the numbers 1, 2 and 5 times a power of ten that lie between low and high."""
    first, last = math.floor(math.log10(low)), math.ceil(math.log10(high))
    ticks = [m * 10.0**power for power in range(first, last + 1) for m in (1, 2, 5)]
    return [tick for tick in ticks if low <= tick <= high]
