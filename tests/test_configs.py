from pathlib import Path

import pytest

from agent_eval.agent.loop import AgentConfig
from agent_eval.experiment.configs import load_config

CONFIGS = Path(__file__).resolve().parent.parent / "configs"


def write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "config.toml"
    path.write_text(text, encoding="utf-8")
    return path


def test_load_a_config(tmp_path):
    path = write(tmp_path, 'model = "claude-haiku-5-5"\neffort = "high"\ntool_specs = "improved"\n')
    assert load_config(path) == AgentConfig("claude-haiku-5-5", "high", tool_specs="improved")


def test_limits_are_optional_settings(tmp_path):
    text = 'model = "claude-sonnet-5-5"\neffort = "low"\ntool_specs = "basic"\nmax_tokens = 20000\n'
    assert load_config(write(tmp_path, text)).max_tokens == 20000


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ('effort = "high"\ntool_specs = "basic"\n', "missing key model"),
        ('model = "claude-haiku-5-5"\ntool_specs = "basic"\n', "missing key effort"),
        ('model = "claude-haiku-5-5"\neffort = "high"\n', "missing key tool_specs"),
        (
            'model = "claude-haiku-5-5"\neffort = "high"\ntool_specs = "basic"\ntemperature = 0\n',
            "unknown key temperature",
        ),
        ('model = "claude-haiku"\neffort = "high"\ntool_specs = "basic"\n', "no price for model"),
        ('model = "claude-haiku-5-5"\neffort = "hgih"\ntool_specs = "basic"\n', "effort must be"),
        (
            'model = "claude-haiku-5-5"\neffort = "high"\ntool_specs = "best"\n',
            "tool_specs must be",
        ),
    ],
)
def test_mistakes_are_caught_before_any_run(tmp_path, text, message):
    with pytest.raises(ValueError, match=message):
        load_config(write(tmp_path, text))


def test_the_five_configurations_of_the_experiment():
    configs = {path.stem: load_config(path) for path in sorted(CONFIGS.glob("*.toml"))}
    assert configs == {
        "haiku-high": AgentConfig("claude-haiku-5-5", "high"),
        "haiku-medium": AgentConfig("claude-haiku-5-5", "medium"),
        "haiku-medium-improved-tools": AgentConfig(
            "claude-haiku-5-5", "medium", tool_specs="improved"
        ),
        "sonnet-high": AgentConfig("claude-sonnet-5-5", "high"),
        "sonnet-low": AgentConfig("claude-sonnet-5-5", "low"),
    }
