import pytest

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


# get_order

def test_get_order_returns_its_details(city):
    assert get_order(city, "o1") == {
        "order_id": "o1",
        "place": "school",
        "window_start": 30,
        "window_end": 60,
    }

def test_get_another_order_returns_its_details(city):
    assert get_order(city, "o3") == {
            "order_id": "o3",
            "place": "bank",
            "window_start": 90,
            "window_end": 120,
    }

def test_correct_error(city):
    with pytest.raises(ToolError, match="no order with id o9"):
        get_order(city, "o9")

def test_letter_sensitive(city):
    with pytest.raises(ToolError, match="no order with id O1"):
        get_order(city, "O1")

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


# shortest_route

def test_shortest_route_to_the_market(city):
    assert shortest_route(city, "warehouse", "market") == {
        "reachable": True,
        "route": ["warehouse", "school", "market"],
        "minutes": 37,
    }


def test_shortest_route_avoids_the_closed_road(city):
    assert shortest_route(city, "office", "school") == {
        "reachable": True,
        "route": ["office", "warehouse", "school"],
        "minutes": 35,
    }


def test_shortest_route_to_the_same_place(city):
    assert shortest_route(city, "school", "school") == {
        "reachable": True,
        "route": ["school"],
        "minutes": 0,
    }


def test_shortest_route_to_unreachable_place_is_not_an_error(city):
    assert shortest_route(city, "warehouse", "gas station") == {
        "reachable": False,
        "route": [],
        "minutes": None,
    }


def test_shortest_route_from_unknown_place_fails(city):
    with pytest.raises(ToolError, match="unknown place: mall"):
        shortest_route(city, "mall", "school")


def test_shortest_route_to_unknown_place_fails(city):
    with pytest.raises(ToolError, match="unknown place: mall"):
        shortest_route(city, "school", "mall")


def test_shortest_route_checks_the_start_first(city):
    with pytest.raises(ToolError, match="unknown place: mall"):
        shortest_route(city, "mall", "park")


# check_schedule

def test_check_schedule_of_the_worked_example(city):
    assert check_schedule(city, ["o1", "o2"]) == {
        "on_time": True,
        "finish_minute": 42,
        "problem": None,
        "stops": [
            {"order_id": "o1", "place": "school", "arrival": 25, "delivery": 30},
            {"order_id": "o2", "place": "market", "arrival": 42, "delivery": 42},
        ],
    }


def test_check_the_first_part_of_a_delivery_order(city):
    assert check_schedule(city, ["o2"]) == {
        "on_time": True,
        "finish_minute": 37,
        "problem": None,
        "stops": [
            {"order_id": "o2", "place": "market", "arrival": 37, "delivery": 37},
        ],
    }


def test_check_schedule_with_a_late_order(city):
    assert check_schedule(city, ["o3", "o1"]) == {
        "on_time": False,
        "finish_minute": None,
        "problem": "order o1 arrives at minute 155, after its window closes at 60",
        "stops": [
            {"order_id": "o3", "place": "bank", "arrival": 44, "delivery": 90},
        ],
    }


def test_check_schedule_with_an_unreachable_order(city):
    assert check_schedule(city, ["o4"]) == {
        "on_time": False,
        "finish_minute": None,
        "problem": "order o4 can't be reached from warehouse",
        "stops": [],
    }


def test_check_schedule_of_no_orders_fails(city):
    with pytest.raises(ToolError, match="no orders to check"):
        check_schedule(city, [])


def test_check_schedule_with_unknown_order_fails(city):
    with pytest.raises(ToolError, match="no order with id o9"):
        check_schedule(city, ["o1", "o9"])


def test_check_schedule_with_the_same_order_twice_fails(city):
    with pytest.raises(ToolError, match="order o1 appears more than once"):
        check_schedule(city, ["o1", "o2", "o1"])


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
