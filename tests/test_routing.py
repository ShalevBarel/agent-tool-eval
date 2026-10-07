import pytest

from agent_eval.env.routing import dijkstra, fastest_route, travel_minutes

# dijkstra


def test_dijkstra_minutes_from_the_warehouse(city):
    minutes, _ = dijkstra(city, "warehouse")
    assert minutes == {
        "warehouse": 0,
        "office": 10,
        "school": 25,
        "restaurant": 25,
        "market": 37,
        "bank": 44,
    }


def test_dijkstra_previous_places_from_the_warehouse(city):
    _, previous = dijkstra(city, "warehouse")
    assert previous == {
        "office": "warehouse",
        "school": "warehouse",
        "restaurant": "warehouse",
        "market": "school",
        "bank": "market",
    }


def test_dijkstra_uses_a_road_once_it_reopens(city):
    city.closed.clear()
    minutes, previous = dijkstra(city, "warehouse")
    assert minutes["school"] == 18
    assert minutes["market"] == 30
    assert minutes["bank"] == 37
    assert previous["school"] == "office"


def test_dijkstra_reaches_everything_from_the_gas_station(city):
    minutes, _ = dijkstra(city, "gas station")
    assert minutes == {
        "gas station": 0,
        "warehouse": 5,
        "office": 15,
        "school": 30,
        "restaurant": 30,
        "market": 42,
        "bank": 49,
    }


def test_dijkstra_from_a_place_with_no_roads_out(city):
    city.add_place("park")
    assert dijkstra(city, "park") == ({"park": 0}, {})


def test_dijkstra_from_unknown_place_fails(city):
    with pytest.raises(ValueError, match="unknown place: mall"):
        dijkstra(city, "mall")


# fastest_route


def test_fastest_route_to_the_bank(city):
    assert fastest_route(city, "warehouse", "bank") == (
        ["warehouse", "school", "market", "bank"],
        44,
    )


def test_fastest_route_avoids_the_closed_road(city):
    assert fastest_route(city, "office", "school") == (
        ["office", "warehouse", "school"],
        35,
    )


def test_fastest_route_to_the_same_place(city):
    assert fastest_route(city, "school", "school") == (["school"], 0)


def test_fastest_route_to_unreachable_place(city):
    assert fastest_route(city, "warehouse", "gas station") is None


def test_fastest_route_to_unknown_place_fails(city):
    with pytest.raises(ValueError, match="unknown place: mall"):
        fastest_route(city, "warehouse", "mall")


def test_fastest_route_checks_the_start_first(city):
    with pytest.raises(ValueError, match="unknown place: mall"):
        fastest_route(city, "mall", "park")


# travel_minutes


def test_travel_minutes_match_the_table_worked_out_by_hand(city, travel):
    assert travel_minutes(city, ["warehouse", "school", "market", "bank"]) == travel


def test_travel_minutes_of_no_places(city):
    assert travel_minutes(city, []) == {}


def test_travel_minutes_from_unknown_place_fails(city):
    with pytest.raises(ValueError, match="unknown place: mall"):
        travel_minutes(city, ["warehouse", "mall"])
