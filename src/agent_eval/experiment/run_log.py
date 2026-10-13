"""The run log: a JSON Lines file with one line per run, graded and priced."""

from pathlib import Path

from agent_eval.agent.loop import AgentConfig, RunResult
from agent_eval.env.tasks import Task


def run_record(task: Task, config: AgentConfig, result: RunResult) -> dict:
    """Return a run's line in the log, as a dict that json can write.

    The line has every field of the RunResult, then the task's difficulty,
    the model and effort, the verdict on the answer, and the cost.

    Example:
        {
            "task_id": "task-057",
            "outcome": "answered",
            "answer": {"solvable": True, "order_ids": ["o2", "o1"]},
            "turns": 4,
            "tool_calls": [{"turn": 1, "name": "get_order", ...}, ...],
            "input_tokens": 15210,
            "output_tokens": 1460,
            "seconds": 11.8,
            "difficulty": "medium",
            "model": "claude-haiku-5-5",
            "effort": "medium",
            "verdict": "correct",
            "cost_usd": 0.002251,
        }
    """
    raise NotImplementedError


def append_record(path: Path, record: dict) -> None:
    """Add a record to the end of a run log as one line, creating the file if needed."""
    raise NotImplementedError


def load_records(path: Path) -> list[dict]:
    """Read every record of a run log, in order."""
    raise NotImplementedError


def summarize(records: list[dict]) -> dict[str, dict]:
    """Count the runs, the correct answers and the cost, by difficulty and over all runs.

    A difficulty with no runs is left out, and no records give an empty dict.

    Example:
        {
            "easy": {"runs": 4, "correct": 4, "cost_usd": 0.0031},
            "hard": {"runs": 8, "correct": 5, "cost_usd": 0.0384},
            "all": {"runs": 12, "correct": 9, "cost_usd": 0.0415},
        }
    """
    raise NotImplementedError
