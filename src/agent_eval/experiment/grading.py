"""Grading an agent's answer against the exact answer of its task."""

from agent_eval.env.tasks import Task
from agent_eval.env.tools import Answer

# Every verdict that grade returns. Only "correct" is a success.
VERDICTS = [
    "correct",
    "no_answer",
    "missed_solution",
    "missing_orders",
    "not_on_time",
    "not_earliest",
]


def grade(task: Task, answer: Answer | None) -> str:
    """Return the verdict on an agent's answer to a task.

    An answer that there is no solution is correct when the task has none.
    A delivery order is correct when it lists every order of the task,
    delivers each one on time, and makes the last delivery at the same
    minute as the exact answer. It may differ from the exact answer's
    order, since two delivery orders can finish at the same minute.

    The answer is one that submit_answer accepted, so it names no unknown
    order and no order twice.

    Verdicts:
        "correct": The answer is right.
        "no_answer": The run ended without an answer.
        "missed_solution": The answer is that there is no solution, but
            there is one.
        "missing_orders": The delivery order leaves out some orders.
        "not_on_time": Some order can't be reached, or is reached after its
            window closes.
        "not_earliest": Every order is on time, but the last delivery is
            later than in the exact answer.

    Raises:
        ValueError: If the answer delivers every order on time and the task
            has no exact answer, or the answer finishes earlier than the
            exact answer. Either one is a bug in the solver, not a mistake
            by the agent.
    """
    raise NotImplementedError
