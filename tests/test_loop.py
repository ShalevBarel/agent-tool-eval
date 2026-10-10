import json

import pytest

from agent_eval.agent.loop import (
    SYSTEM_PROMPT,
    TOOLS,
    AgentConfig,
    ToolCall,
    call_tool,
    run_task,
    task_prompt,
)
from agent_eval.agent.tool_specs import TOOL_SPECS
from agent_eval.env.tasks import Task
from agent_eval.env.tools import Answer, get_order, shortest_route
from fake_client import FakeClient, response, text, thinking, tool_use

CONFIG = AgentConfig("claude-haiku-5-5", "medium")


@pytest.fixture
def task(city) -> Task:
    return Task("task-test", "hard", city, None)


def submit(order_ids: list[str], block_id: str = "toolu_9"):
    """A call to submit_answer with a solvable answer."""
    return tool_use("submit_answer", {"solvable": True, "order_ids": order_ids}, block_id)


def get(order_id: str, block_id: str = "toolu_1"):
    """A call to get_order."""
    return tool_use("get_order", {"order_id": order_id}, block_id)


# task_prompt


def test_task_prompt(city):
    assert task_prompt(city) == (
        "The courier starts at warehouse at minute 0. Deliver these orders: o1, o2, o3, o4."
    )


def test_task_prompt_uses_the_start(city):
    city.set_start("office", 15)
    assert task_prompt(city).startswith("The courier starts at office at minute 15. ")


# call_tool


def test_call_tool_returns_the_result(city):
    assert call_tool(city, "get_order", {"order_id": "o1"}) == (get_order(city, "o1"), None)


def test_call_tool_passes_every_argument(city):
    result = call_tool(city, "shortest_route", {"start": "warehouse", "end": "market"})
    assert result == (shortest_route(city, "warehouse", "market"), None)


def test_call_tool_returns_a_tool_error(city):
    assert call_tool(city, "get_order", {"order_id": "o9"}) == (None, "no order with id o9")


def test_call_unknown_tool(city):
    assert call_tool(city, "teleport", {"place": "bank"}) == (None, "unknown tool: teleport")


def test_call_submit_answer_returns_the_answer(city):
    result = call_tool(city, "submit_answer", {"solvable": False, "order_ids": []})
    assert result == (Answer(False, []), None)


def test_call_tool_lets_bugs_through(city, monkeypatch):
    def broken(world):
        raise ValueError("a bug in our code")

    monkeypatch.setitem(TOOLS, "broken", broken)
    with pytest.raises(ValueError, match="a bug in our code"):
        call_tool(city, "broken", {})


# run_task: the requests


def test_first_request(task):
    client = FakeClient([response(text("I give up."), stop_reason="end_turn")])
    run_task(client, task, CONFIG)
    assert client.messages.requests == [
        {
            "model": "claude-haiku-5-5",
            "max_tokens": 16000,
            "system": SYSTEM_PROMPT,
            "tools": TOOL_SPECS,
            "messages": [{"role": "user", "content": task_prompt(task.world)}],
            "output_config": {"effort": "medium"},
        }
    ]


def test_request_follows_the_config(task):
    config = AgentConfig("claude-sonnet-5-5", "high", max_tokens=8000)
    client = FakeClient([response(text("I give up."), stop_reason="end_turn")])
    run_task(client, task, config)
    request = client.messages.requests[0]
    assert request["model"] == "claude-sonnet-5-5"
    assert request["max_tokens"] == 8000
    assert request["output_config"] == {"effort": "high"}


def test_tool_result_goes_back_to_the_model(task):
    first = response(get("o1", "toolu_a"))
    client = FakeClient([first, response(submit(["o1"]))])
    run_task(client, task, CONFIG)
    messages = client.messages.requests[1]["messages"]
    assert len(messages) == 3
    assert messages[1] == {"role": "assistant", "content": first.content}
    assert messages[2] == {
        "role": "user",
        "content": [
            {
                "type": "tool_result",
                "tool_use_id": "toolu_a",
                "content": json.dumps(get_order(task.world, "o1")),
                "is_error": False,
            }
        ],
    }


def test_thinking_and_text_go_back_unchanged(task):
    first = response(thinking("sig-1"), text("Let me look."), get("o1"))
    client = FakeClient([first, response(submit(["o1"]))])
    run_task(client, task, CONFIG)
    assert client.messages.requests[1]["messages"][1]["content"] == first.content


def test_history_grows_by_two_messages_a_turn(task):
    second = response(get("o2", "toolu_b"))
    client = FakeClient([response(get("o1", "toolu_a")), second, response(submit(["o1", "o2"]))])
    run_task(client, task, CONFIG)
    third_request = client.messages.requests[2]["messages"]
    assert len(third_request) == 5
    assert third_request[:3] == client.messages.requests[1]["messages"]
    assert third_request[3] == {"role": "assistant", "content": second.content}
    assert third_request[4]["content"][0]["tool_use_id"] == "toolu_b"


def test_parallel_calls_go_back_in_one_message(task):
    first = response(get("o1", "toolu_a"), get("o2", "toolu_b"))
    client = FakeClient([first, response(submit(["o1", "o2"]))])
    run_task(client, task, CONFIG)
    messages = client.messages.requests[1]["messages"]
    assert len(messages) == 3
    assert [block["tool_use_id"] for block in messages[2]["content"]] == ["toolu_a", "toolu_b"]


# run_task: errors go back to the model


def test_tool_error_goes_back_and_the_run_goes_on(task):
    client = FakeClient([response(get("o9", "toolu_a")), response(submit(["o1"]))])
    result = run_task(client, task, CONFIG)
    sent_back = client.messages.requests[1]["messages"][2]["content"][0]
    assert sent_back["content"] == "no order with id o9"
    assert sent_back["is_error"] is True
    assert result.outcome == "answered"


def test_unknown_tool_goes_back_as_an_error(task):
    teleport = tool_use("teleport", {"place": "bank"}, "toolu_a")
    client = FakeClient([response(teleport), response(submit(["o1"]))])
    run_task(client, task, CONFIG)
    sent_back = client.messages.requests[1]["messages"][2]["content"][0]
    assert sent_back["content"] == "unknown tool: teleport"
    assert sent_back["is_error"] is True


def test_invalid_answer_goes_back_as_an_error(task):
    invalid = tool_use("submit_answer", {"solvable": False, "order_ids": ["o1"]}, "toolu_a")
    client = FakeClient([response(invalid), response(submit(["o2", "o1"]))])
    result = run_task(client, task, CONFIG)
    sent_back = client.messages.requests[1]["messages"][2]["content"][0]
    assert sent_back["content"] == "an answer of no solution must not list orders"
    assert result.answer == Answer(True, ["o2", "o1"])
    assert result.turns == 2


# run_task: how a run ends


def test_answer_ends_the_run(task):
    client = FakeClient([response(submit(["o2", "o1"]))])
    result = run_task(client, task, CONFIG)
    assert result.outcome == "answered"
    assert result.answer == Answer(True, ["o2", "o1"])
    assert result.turns == 1
    assert len(client.messages.requests) == 1


def test_no_solution_is_an_answer(task):
    no_solution = tool_use("submit_answer", {"solvable": False, "order_ids": []})
    result = run_task(FakeClient([response(no_solution)]), task, CONFIG)
    assert result.outcome == "answered"
    assert result.answer == Answer(False, [])


def test_calls_after_the_answer_are_not_run(task):
    client = FakeClient([response(submit(["o1"]), get("o2"))])
    result = run_task(client, task, CONFIG)
    assert [call.name for call in result.tool_calls] == ["submit_answer"]


def test_stopping_without_an_answer(task):
    client = FakeClient([response(text("There is no way to do this."), stop_reason="end_turn")])
    result = run_task(client, task, CONFIG)
    assert result.outcome == "end_turn"
    assert result.answer is None
    assert result.turns == 1
    assert result.tool_calls == []


@pytest.mark.parametrize("stop_reason", ["max_tokens", "refusal"])
def test_other_stop_reasons_end_the_run(task, stop_reason):
    client = FakeClient([response(thinking(), stop_reason=stop_reason)])
    result = run_task(client, task, CONFIG)
    assert result.outcome == stop_reason
    assert result.answer is None


def test_turn_limit(task):
    config = AgentConfig("claude-haiku-5-5", "medium", max_turns=3)
    client = FakeClient([response(get("o1", f"toolu_{i}")) for i in range(3)])
    result = run_task(client, task, config)
    assert result.outcome == "turn_limit"
    assert result.answer is None
    assert result.turns == 3
    assert len(client.messages.requests) == 3


# run_task: the record


def test_every_tool_call_is_recorded(task):
    first = response(get("o1", "toolu_a"), get("o9", "toolu_b"))
    client = FakeClient([first, response(submit(["o1"]))])
    result = run_task(client, task, CONFIG)
    order = json.dumps(get_order(task.world, "o1"))
    answer = {"solvable": True, "order_ids": ["o1"]}
    assert result.tool_calls == [
        ToolCall(1, "get_order", {"order_id": "o1"}, order, False),
        ToolCall(1, "get_order", {"order_id": "o9"}, "no order with id o9", True),
        ToolCall(2, "submit_answer", answer, "answer received", False),
    ]


def test_tokens_are_summed_over_turns(task):
    first = response(get("o1"), input_tokens=500, output_tokens=40)
    second = response(submit(["o1"]), input_tokens=700, output_tokens=60)
    result = run_task(FakeClient([first, second]), task, CONFIG)
    assert result.input_tokens == 1200
    assert result.output_tokens == 100


def test_result_names_the_task_and_times_the_run(task):
    result = run_task(FakeClient([response(submit(["o1"]))]), task, CONFIG)
    assert result.task_id == "task-test"
    assert result.seconds >= 0
