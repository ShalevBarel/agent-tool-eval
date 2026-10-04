import pytest

from agent_eval.env.world import Order, TimeWindow, World


# TimeWindow

def test_window_contains_its_edges_and_inside():
    window = TimeWindow(30, 60)
    assert window.contains(30)
    assert window.contains(45)
    assert window.contains(60)


def test_window_does_not_contain_minutes_outside():
    window = TimeWindow(30, 60)
    assert not window.contains(29)
    assert not window.contains(61)


def test_window_of_a_single_minute():
    assert TimeWindow(40, 40).contains(40)


def test_window_cannot_start_before_minute_zero():
    with pytest.raises(ValueError, match="a window cannot start before minute 0"):
        TimeWindow(-5, 10)


def test_window_cannot_start_after_it_ends():
    with pytest.raises(ValueError, match="start must not be after end"):
        TimeWindow(60, 30)


# Order

def test_order_holds_its_fields():
    order = Order("o1", "school", TimeWindow(30, 60))
    assert order.order_id == "o1"
    assert order.place == "school"
    assert order.window == TimeWindow(30, 60)


# World: places and roads

def test_new_world_is_empty():
    world = World()
    assert world.roads == {}
    assert world.closed == set()
    assert world.orders == {}


def test_add_place():
    world = World()
    world.add_place("warehouse")
    assert world.roads == {"warehouse": {}}


def test_add_place_twice_fails():
    world = World()
    world.add_place("warehouse")
    with pytest.raises(ValueError, match="place warehouse already exists"):
        world.add_place("warehouse")


def test_add_road_is_one_way():
    world = World()
    world.add_place("warehouse")
    world.add_place("office")
    world.add_road("warehouse", "office", 10)
    assert world.roads["warehouse"] == {"office": 10}
    assert world.roads["office"] == {}


def test_add_road_from_unknown_place_fails():
    world = World()
    world.add_place("warehouse")
    with pytest.raises(ValueError, match="unknown place: mall"):
        world.add_road("mall", "warehouse", 10)


def test_add_road_to_unknown_place_fails():
    world = World()
    world.add_place("warehouse")
    with pytest.raises(ValueError, match="unknown place: mall"):
        world.add_road("warehouse", "mall", 10)


def test_road_must_take_at_least_one_minute():
    world = World()
    world.add_place("warehouse")
    world.add_place("office")
    with pytest.raises(ValueError, match="a road must take at least 1 minute"):
        world.add_road("warehouse", "office", 0)


def test_add_same_road_twice_fails():
    world = World()
    world.add_place("warehouse")
    world.add_place("office")
    world.add_road("warehouse", "office", 10)
    with pytest.raises(ValueError, match="road from warehouse to office already exists"):
        world.add_road("warehouse", "office", 12)


# World: closures

def test_close_road(city):
    city.close_road("warehouse", "school")
    assert ("warehouse", "school") in city.closed
    assert city.roads["warehouse"]["school"] == 25


def test_close_missing_road_fails(city):
    with pytest.raises(ValueError, match="no road from warehouse to bank"):
        city.close_road("warehouse", "bank")


def test_close_road_from_unknown_place_fails(city):
    with pytest.raises(ValueError, match="no road from mall to warehouse"):
        city.close_road("mall", "warehouse")


def test_is_open(city):
    assert city.is_open("warehouse", "office")
    assert not city.is_open("office", "school")
    assert not city.is_open("warehouse", "bank")
    assert not city.is_open("mall", "warehouse")


# World: orders

def test_add_order(city):
    order = Order("o5", "office", TimeWindow(0, 30))
    city.add_order(order)
    assert city.orders["o5"] == order


def test_add_order_with_taken_id_fails(city):
    with pytest.raises(ValueError, match="order o1 already exists"):
        city.add_order(Order("o1", "office", TimeWindow(0, 30)))


def test_add_order_to_unknown_place_fails(city):
    with pytest.raises(ValueError, match="unknown place: mall"):
        city.add_order(Order("o5", "mall", TimeWindow(0, 30)))


# The test city itself

def test_city_has_what_the_lesson_shows(city):
    assert sorted(city.roads) == [
        "bank", "gas station", "market", "office", "restaurant", "school", "warehouse",
    ]
    assert sum(len(ends) for ends in city.roads.values()) == 12
    assert city.closed == {("office", "school")}
    assert sorted(city.orders) == ["o1", "o2", "o3", "o4"]
