"""The agent loop: the model calls tools until it submits an answer."""

import json
import time
from dataclasses import dataclass

from anthropic import Anthropic

from agent_eval.agent.tool_specs import TOOL_SPECS
from agent_eval.env.tasks import Task
from agent_eval.env.tools import (
    Answer,
    ToolError,
    check_schedule,
    get_order,
    roads_from,
    route_minutes,
    shortest_route,
    submit_answer,
)
from agent_eval.env.world import World

SYSTEM_PROMPT = """\
You plan the deliveries of a courier in a small city.

The city has numbered places joined by one-way roads. A two-way street is two
roads, one in each direction. Some roads are closed today and can't be used.
Times are whole minutes from the start of the day.

Each order must be delivered at its place within its time window, and both
ends of a window count as on time. The courier delivers the orders one at a
time, in an order you choose, and always takes the fastest open route to the
next order. If it arrives before a window opens, it waits until it opens.

Find a delivery order that delivers every order on time and makes the last
delivery as early as possible. If no delivery order is on time, the answer is
that there is no solution.

Use the tools to learn about the orders and the roads. When you know the
answer, call submit_answer."""

# Every tool the model can call, by the name in its spec.
TOOLS = {
    "get_order": get_order,
    "roads_from": roads_from,
    "route_minutes": route_minutes,
    "shortest_route": shortest_route,
    "check_schedule": check_schedule,
    "submit_answer": submit_answer,
}


@dataclass(frozen=True)
class AgentConfig:
    """How the agent runs: the model, how hard it thinks, and the limits of a run.

    Attributes:
        model: The model id, such as "claude-haiku-5-5".
        effort: How much the model thinks: "low", "medium", "high", "xhigh"
            or "max".
        max_turns: The most requests to the model in one run.
        max_tokens: The most tokens in one response, thinking included.
    """

    model: str
    effort: str
    max_turns: int = 30
    max_tokens: int = 16000


@dataclass(frozen=True)
class ToolCall:
    """One tool call by the model, and what was sent back.

    Attributes:
        turn: The request to the model that made the call, counting from 1.
        name: The tool's name.
        tool_input: The arguments, as the model sent them.
        output: The text sent back to the model.
        is_error: Whether the call failed.
    """

    turn: int
    name: str
    tool_input: dict
    output: str
    is_error: bool


@dataclass(frozen=True)
class RunResult:
    """What happened in one run of the agent on one task.

    Attributes:
        task_id: The task's id.
        outcome: "answered" if the model submitted a valid answer,
            "turn_limit" if it used every turn without one, and otherwise
            the stop reason of the response that ended the run, such as
            "end_turn" when the model stopped calling tools.
        answer: The submitted answer, or None.
        turns: The number of requests to the model.
        tool_calls: Every tool call, in order.
        input_tokens: Input tokens, summed over all requests.
        output_tokens: Output tokens, thinking included, summed over all requests.
        seconds: How long the run took.
    """

    task_id: str
    outcome: str
    answer: Answer | None
    turns: int
    tool_calls: list[ToolCall]
    input_tokens: int
    output_tokens: int
    seconds: float


def task_prompt(world: World) -> str:
    """Return the first message of a task: the courier's start and the order ids.

    Example:
        "The courier starts at p0 at minute 0. Deliver these orders: o1, o2."
    """
    order_ids = ", ".join(world.orders)
    return (
        f"The courier starts at {world.start} at minute {world.start_minute}. "
        f"Deliver these orders: {order_ids}."
    )


def call_tool(world: World, name: str, tool_input: dict) -> tuple[object, str | None]:
    """Run a tool the model called.

    Returns the tool's result and None. If the tool is unknown, or raises
    ToolError, returns None and an error message for the model. Any other
    exception is a bug, and propagates.
    """
    if name not in TOOLS:
        return None, f"unknown tool: {name}"
    try:
        return TOOLS[name](world, **tool_input), None
    except ToolError as error:
        return None, str(error)


def run_task(client: Anthropic, task: Task, config: AgentConfig) -> RunResult:
    """Let the model work on a task with the tools until it submits an answer.

    The run also ends when a response stops without a tool call, or after
    config.max_turns requests. Each response goes back to the model as it
    came, thinking blocks included, followed by one message with the results
    of all its tool calls. Tool calls after a valid answer are not run.
    """
    started = time.perf_counter()
    messages = [{"role": "user", "content": task_prompt(task.world)}]
    tool_calls = []
    answer = None
    outcome = "turn_limit"
    turns = 0
    input_tokens = 0
    output_tokens = 0

    while turns < config.max_turns:
        response = client.messages.create(
            model=config.model,
            max_tokens=config.max_tokens,
            system=SYSTEM_PROMPT,
            tools=TOOL_SPECS,
            messages=messages,
            output_config={"effort": config.effort},
        )
        turns += 1
        input_tokens += response.usage.input_tokens
        output_tokens += response.usage.output_tokens
        if response.stop_reason != "tool_use":
            outcome = response.stop_reason
            break

        results = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            result, error = call_tool(task.world, block.name, block.input)
            if error is not None:
                output = error
            elif isinstance(result, Answer):
                output = "answer received"
                answer = result
            else:
                output = json.dumps(result)
            tool_calls.append(ToolCall(turns, block.name, block.input, output, error is not None))
            results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": output,
                    "is_error": error is not None,
                }
            )
            if answer is not None:
                break

        if answer is not None:
            outcome = "answered"
            break
        messages.append({"role": "assistant", "content": response.content})
        messages.append({"role": "user", "content": results})

    return RunResult(
        task.task_id,
        outcome,
        answer,
        turns,
        tool_calls,
        input_tokens,
        output_tokens,
        time.perf_counter() - started,
    )
