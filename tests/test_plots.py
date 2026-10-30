import pandas as pd
import pytest

from agent_eval.analysis.plots import (
    MODEL_COLORS,
    config_models,
    cost_vs_success,
    model_colors,
    save_plots,
    success_by_difficulty,
)
from agent_eval.analysis.report import estimates, estimates_by_difficulty

DIFFICULTIES = {"task-000": "easy", "task-001": "easy", "task-002": "hard", "task-003": "hard"}

# 1 is a correct run, 0 a wrong one.
OUTCOMES = {
    "cheap": {"task-000": [1, 1], "task-001": [1, 1], "task-002": [1, 0], "task-003": [1, 1]},
    "strong": {"task-000": [1, 1], "task-001": [1, 0], "task-002": [0, 0], "task-003": [0, 1]},
    "cheap-again": {"task-000": [1, 0], "task-001": [1, 1], "task-002": [0, 0], "task-003": [0, 0]},
}
MODELS = {"cheap": "model-a", "strong": "model-b", "cheap-again": "model-a"}
COSTS = {"cheap": 0.01, "strong": 0.05, "cheap-again": 0.01}


@pytest.fixture
def runs() -> pd.DataFrame:
    """Three configurations of two models on two easy and two hard tasks, two repeats each."""
    rows = []
    for config, by_task in OUTCOMES.items():
        for task_id, outcomes in by_task.items():
            for repeat, outcome in enumerate(outcomes):
                verdict = "correct" if outcome else "not_earliest"
                difficulty = DIFFICULTIES[task_id]
                rows.append(
                    (config, MODELS[config], task_id, difficulty, repeat, verdict, COSTS[config])
                )
    columns = ["config", "model", "task_id", "difficulty", "repeat", "verdict", "cost_usd"]
    return pd.DataFrame(rows, columns=columns)


def test_config_models(runs):
    assert config_models(runs) == MODELS


def test_config_models_refuse_two_models_in_one_configuration(runs):
    runs.loc[0, "model"] = "model-b"
    with pytest.raises(ValueError, match="cheap ran more than one model"):
        config_models(runs)


def test_model_colors_follow_the_model_not_the_configuration():
    assert model_colors(MODELS) == {"model-a": MODEL_COLORS[0], "model-b": MODEL_COLORS[1]}


def test_model_colors_refuse_more_models_than_colors():
    models = {f"config-{i}": f"model-{i}" for i in range(len(MODEL_COLORS) + 1)}
    with pytest.raises(ValueError, match="models, but only"):
        model_colors(models)


def points(figure) -> list[tuple[float, float, str]]:
    """Return the x, y and color of every point in the figure's plot."""
    lines = [line for line in figure.axes[0].lines if line.get_marker() == "o"]
    return [(line.get_xdata()[0], line.get_ydata()[0], line.get_color()) for line in lines]


def test_cost_vs_success_puts_each_configuration_at_its_estimates(runs):
    results = estimates(runs, n_resamples=200)
    found = points(cost_vs_success(results, MODELS))
    assert len(found) == len(results)
    for (x, y, color), (config, result) in zip(found, results.items()):
        assert x == pytest.approx(result["cost_per_correct"].value)
        assert y == pytest.approx(100 * result["success"].value)
        assert color == model_colors(MODELS)[MODELS[config]]


def test_cost_vs_success_names_every_point(runs):
    figure = cost_vs_success(estimates(runs, n_resamples=200), MODELS)
    names = {text.get_text() for text in figure.axes[0].texts}
    assert set(MODELS) <= names


def test_cost_vs_success_has_a_logarithmic_cost_axis(runs):
    figure = cost_vs_success(estimates(runs, n_resamples=200), MODELS)
    assert figure.axes[0].get_xscale() == "log"


def test_success_by_difficulty_has_a_panel_for_every_difficulty(runs):
    figure = success_by_difficulty(estimates_by_difficulty(runs, n_resamples=200), MODELS)
    assert [panel.get_title() for panel in figure.axes] == ["Easy", "Hard"]


def test_success_by_difficulty_has_a_bar_for_every_configuration(runs):
    by_difficulty = estimates_by_difficulty(runs, n_resamples=200)
    figure = success_by_difficulty(by_difficulty, MODELS)
    for panel, level in zip(figure.axes, ["easy", "hard"]):
        widths = [bar.get_width() for bar in panel.patches]
        assert widths == pytest.approx([100 * by_difficulty[c][level].value for c in by_difficulty])


def test_success_by_difficulty_lists_configurations_in_the_order_given(runs):
    by_difficulty = estimates_by_difficulty(runs, n_resamples=200)
    order = ["strong", "cheap-again", "cheap"]
    figure = success_by_difficulty({config: by_difficulty[config] for config in order}, MODELS)
    assert [label.get_text() for label in figure.axes[0].get_yticklabels()] == order
    assert figure.axes[0].yaxis_inverted()


def test_save_plots_writes_two_png_files(runs, tmp_path):
    paths = save_plots(runs, tmp_path / "figures", n_resamples=100)
    assert [path.name for path in paths] == ["cost-vs-success.png", "success-by-difficulty.png"]
    for path in paths:
        assert path.read_bytes().startswith(b"\x89PNG")


def test_save_plots_need_the_same_tasks_for_every_configuration(runs, tmp_path):
    fewer = runs[~((runs["config"] == "strong") & (runs["task_id"] == "task-003"))]
    with pytest.raises(ValueError, match="ran different tasks"):
        save_plots(fewer, tmp_path, n_resamples=10)
