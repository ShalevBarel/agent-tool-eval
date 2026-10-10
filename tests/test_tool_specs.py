import typing

import pytest

from agent_eval.agent.loop import TOOLS
from agent_eval.agent.tool_specs import TOOL_SPECS
from agent_eval.env import tools

NAMES = [
    "get_order",
    "roads_from",
    "route_minutes",
    "shortest_route",
    "check_schedule",
    "submit_answer",
]

# The JSON Schema for each type hint the tools use.
SCHEMA_FOR = {
    str: {"type": "string"},
    bool: {"type": "boolean"},
    list[str]: {"type": "array", "items": {"type": "string"}},
}


def spec_named(name: str) -> dict:
    return next(spec for spec in TOOL_SPECS if spec["name"] == name)


def model_arguments(name: str) -> dict:
    """The tool's type hints, without world, which the loop supplies, and the return type."""
    hints = typing.get_type_hints(getattr(tools, name))
    del hints["world"], hints["return"]
    return hints


def test_one_spec_for_each_tool_in_order():
    assert [spec["name"] for spec in TOOL_SPECS] == NAMES


def test_the_loop_can_run_every_tool():
    assert set(TOOLS) == set(NAMES)


@pytest.mark.parametrize("name", NAMES)
def test_spec_has_the_four_keys(name):
    spec = spec_named(name)
    assert set(spec) == {"name", "description", "input_schema", "strict"}
    assert spec["strict"] is True
    assert spec["description"].strip()


@pytest.mark.parametrize("name", NAMES)
def test_schema_lists_the_tool_arguments_in_order(name):
    schema = spec_named(name)["input_schema"]
    arguments = list(model_arguments(name))
    assert schema["type"] == "object"
    assert list(schema["properties"]) == arguments
    assert schema["required"] == arguments
    assert schema["additionalProperties"] is False


@pytest.mark.parametrize("name", NAMES)
def test_each_argument_has_its_type_and_a_description(name):
    properties = spec_named(name)["input_schema"]["properties"]
    for argument, hint in model_arguments(name).items():
        prop = dict(properties[argument])
        assert prop.pop("description", "").strip(), f"{argument} has no description"
        assert prop == SCHEMA_FOR[hint], f"wrong type for {argument}"
