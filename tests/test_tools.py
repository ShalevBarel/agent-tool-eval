import pytest

from agent_eval.env.tools import (
    Answer,
    ToolError,
    get_order,
    roads_from,
    route_minutes,
    submit_answer,
)


# get_order: your tests go here. Part 6 of lesson 4 lists what to cover.

def test_get_order_returns_its_details(city):
    assert get_order(city, "o1") == {
        "order_id": "o1",
        "place": "school",
        "window_start": 30,
        "window_end": 60,
    }


# roads_from

def test_roads_from_lists_roads_sorted(city):
    assert roads_from(city, "office") == [
        {"to": "market", "minutes": 30, "open": True},
        {"to": "school", "minutes": 8, "open": False},
        {"to": "warehouse", "minutes": 10, "open": True},
    ]


def test_roads_from_place_with_no_roads_out(city):
    city.add_place("park")
    assert roads_from(city, "park") == []


def test_roads_from_unknown_place_fails(city):
    with pytest.raises(ToolError, match="unknown place: mall"):
        roads_from(city, "mall")


# route_minutes

def test_route_minutes_adds_up_the_roads(city):
    assert route_minutes(city, ["warehouse", "office", "market"]) == 40


def test_route_minutes_of_a_longer_route(city):
    assert route_minutes(city, ["warehouse", "school", "market", "bank", "restaurant", "warehouse"]) == 84


def test_route_with_a_single_stop_takes_no_time(city):
    assert route_minutes(city, ["school"]) == 0


def test_empty_route_fails(city):
    with pytest.raises(ToolError, match="route is empty"):
        route_minutes(city, [])


def test_route_through_unknown_place_fails(city):
    with pytest.raises(ToolError, match="unknown place: mall"):
        route_minutes(city, ["warehouse", "mall"])


def test_unknown_places_are_found_before_bad_roads(city):
    with pytest.raises(ToolError, match="unknown place: mall"):
        route_minutes(city, ["warehouse", "bank", "mall"])


def test_route_on_missing_road_fails(city):
    with pytest.raises(ToolError, match="no road from warehouse to bank"):
        route_minutes(city, ["warehouse", "bank"])


def test_route_on_closed_road_fails(city):
    with pytest.raises(ToolError, match="the road from office to school is closed today"):
        route_minutes(city, ["warehouse", "office", "school"])


def test_route_reports_the_first_bad_road(city):
    with pytest.raises(ToolError, match="no road from warehouse to bank"):
        route_minutes(city, ["warehouse", "bank", "gas station"])


# submit_answer

def test_submit_a_delivery_order(city):
    answer = submit_answer(city, True, ["o2", "o1"])
    assert answer == Answer(True, ["o2", "o1"])


def test_submit_no_solution(city):
    assert submit_answer(city, False, []) == Answer(False, [])


def test_solvable_answer_needs_orders(city):
    with pytest.raises(ToolError, match="a solvable answer must list the orders to deliver"):
        submit_answer(city, True, [])


def test_no_solution_must_not_list_orders(city):
    with pytest.raises(ToolError, match="an answer of no solution must not list orders"):
        submit_answer(city, False, ["o1"])


def test_submit_unknown_order_fails(city):
    with pytest.raises(ToolError, match="no order with id o9"):
        submit_answer(city, True, ["o1", "o9"])


def test_submit_same_order_twice_fails(city):
    with pytest.raises(ToolError, match="order o1 appears more than once"):
        submit_answer(city, True, ["o1", "o2", "o1"])
