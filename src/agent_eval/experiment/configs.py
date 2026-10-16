"""Agent configurations stored as TOML files, one file per configuration.

A file names the model, the effort and the tool descriptions:

    model = "claude-haiku-5-5"
    effort = "medium"
    tool_specs = "basic"

It may also set max_turns and max_tokens. Adding a configuration to the
experiment means adding a file, with no change to the code.
"""

import tomllib
from dataclasses import fields
from pathlib import Path

from agent_eval.agent.loop import AgentConfig
from agent_eval.agent.tool_specs import TOOL_SPEC_SETS
from agent_eval.experiment.cost import PRICES

REQUIRED = ["model", "effort", "tool_specs"]
EFFORTS = ["low", "medium", "high", "xhigh", "max"]


def load_config(path: Path) -> AgentConfig:
    """Read an agent configuration from a TOML file.

    Raises:
        ValueError: If a key is missing or unknown, if the model has no
            price, or if the effort or the tool descriptions are unknown.
            Checking this before any run means a typo costs nothing.
    """
    with path.open("rb") as file:
        values = tomllib.load(file)

    known = {field.name for field in fields(AgentConfig)}
    for key in values:
        if key not in known:
            raise ValueError(f"{path}: unknown key {key}")
    for key in REQUIRED:
        if key not in values:
            raise ValueError(f"{path}: missing key {key}")
    if values["model"] not in PRICES:
        raise ValueError(f"{path}: no price for model {values['model']}")
    if values["effort"] not in EFFORTS:
        raise ValueError(f"{path}: effort must be one of {', '.join(EFFORTS)}")
    if values["tool_specs"] not in TOOL_SPEC_SETS:
        raise ValueError(f"{path}: tool_specs must be one of {', '.join(TOOL_SPEC_SETS)}")
    return AgentConfig(**values)
