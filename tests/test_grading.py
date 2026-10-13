import copy
from pathlib import Path

import pytest

from agent_eval.env.tasks import Task, load_tasks
from agent_eval.env.tools import Answer
from agent_eval.experiment.grading import VERDICTS, grade
from agent_eval.solver import Solution
from agent_eval.solver.held_karp import solve

SAVED_TASKS = Path(__file__).resolve().parent.parent / "data" / "tasks.jsonl"


# A test that asks for several of these tasks gets one city for all of them,
# so each task works on its own copy.


@pytest.fixture
def two_orders(city) -> Task:
    """o1 and o2 of the test city. o1 then o2 finishes at 42, and o2 then o1 at 49."""
    world = copy.deepcopy(city)
    del world.orders["o3"], world.orders["o4"]
    return Task("task-two", "medium", world, Solution(["o1", "o2"], 42))


@pytest.fixture
def three_orders(city) -> Task:
    """o1, o2 and o3 of the test city. o1 o2 o3 and o2 o1 o3 both finish at 90."""
    world = copy.deepcopy(city)
    del world.orders["o4"]
    return Task("task-three", "hard", world, Solution(["o1", "o2", "o3"], 90))


@pytest.fixture
def no_solution(city) -> Task:
    """All four orders of the test city. No road leads to o4, at the gas station."""
    return Task("task-none", "hard", copy.deepcopy(city), None)


def test_the_hand_made_answers_match_the_solver(two_orders, three_orders, no_solution):
    for task in [two_orders, three_orders]:
        found = solve(task.world, list(task.world.orders))
        assert found.finish_minute == task.answer.finish_minute
    assert solve(no_solution.world, list(no_solution.world.orders)) is None


# Correct answers


def test_the_exact_answer_is_correct(three_orders):
    assert grade(three_orders, Answer(True, ["o1", "o2", "o3"])) == "correct"


def test_another_order_that_finishes_as_early_is_correct(three_orders):
    assert grade(three_orders, Answer(True, ["o2", "o1", "o3"])) == "correct"


def test_no_solution_is_correct_when_there_is_none(no_solution):
    assert grade(no_solution, Answer(False, [])) == "correct"


def test_every_exact_answer_in_the_tasks_file_is_correct():
    for task in load_tasks(SAVED_TASKS):
        if task.answer is None:
            answer = Answer(False, [])
        else:
            answer = Answer(True, task.answer.order_ids)
        assert grade(task, answer) == "correct", task.task_id


# Wrong answers


def test_no_answer(three_orders):
    assert grade(three_orders, None) == "no_answer"


def test_no_solution_when_there_is_one(three_orders):
    assert grade(three_orders, Answer(False, [])) == "missed_solution"


def test_missing_orders(three_orders):
    assert grade(three_orders, Answer(True, ["o1", "o2"])) == "missing_orders"


def test_missing_orders_is_checked_before_the_schedule(no_solution):
    # o1 o2 o3 is on time, but o4 is missing.
    assert grade(no_solution, Answer(True, ["o1", "o2", "o3"])) == "missing_orders"


def test_late(three_orders):
    # After o3 at the bank at minute 90, the courier reaches the market long after 45.
    assert grade(three_orders, Answer(True, ["o1", "o3", "o2"])) == "not_on_time"


def test_unreachable(no_solution):
    assert grade(no_solution, Answer(True, ["o1", "o2", "o3", "o4"])) == "not_on_time"


def test_on_time_but_not_earliest(two_orders):
    assert grade(two_orders, Answer(True, ["o2", "o1"])) == "not_earliest"


def test_every_verdict_is_listed(two_orders, three_orders, no_solution):
    verdicts = {
        grade(three_orders, Answer(True, ["o1", "o2", "o3"])),
        grade(three_orders, None),
        grade(three_orders, Answer(False, [])),
        grade(three_orders, Answer(True, ["o1", "o2"])),
        grade(no_solution, Answer(True, ["o1", "o2", "o3", "o4"])),
        grade(two_orders, Answer(True, ["o2", "o1"])),
    }
    assert verdicts == set(VERDICTS)


# Bugs in the exact answer


def test_an_answer_that_beats_the_exact_answer_is_a_bug(city):
    del city.orders["o4"]
    wrong = Task("task-wrong", "hard", city, Solution(["o1", "o2", "o3"], 100))
    with pytest.raises(ValueError, match="task-wrong"):
        grade(wrong, Answer(True, ["o1", "o2", "o3"]))


def test_an_answer_to_a_task_with_no_solution_is_a_bug(city):
    del city.orders["o4"]
    wrong = Task("task-wrong", "hard", city, None)
    with pytest.raises(ValueError, match="task-wrong"):
        grade(wrong, Answer(True, ["o1", "o2", "o3"]))
