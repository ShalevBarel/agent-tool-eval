"""The agent loop: the model calls tools until it submits an answer."""

from dataclasses import dataclass

from anthropic import Anthropic

from agent_eval.env.tasks import Task
from agent_eval.env.tools import (
    Answer,
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
    raise NotImplementedError


def call_tool(world: World, name: str, tool_input: dict) -> tuple[object, str | None]:
    """Run a tool the model called.

    Returns the tool's result and None. If the tool is unknown, or raises
    ToolError, returns None and an error message for the model. Any other
    exception is a bug, and propagates.
    """
    raise NotImplementedError


def run_task(client: Anthropic, task: Task, config: AgentConfig) -> RunResult:
    """Let the model work on a task with the tools until it submits an answer.

    The run also ends when a response stops without a tool call, or after
    config.max_turns requests. Each response goes back to the model as it
    came, thinking blocks included, followed by one message with the results
    of all its tool calls. Tool calls after a valid answer are not run.
    """
    raise NotImplementedError
