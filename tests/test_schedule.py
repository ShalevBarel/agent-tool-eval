from agent_eval.env.schedule import Schedule, Stop, simulate


def test_simulate_waits_for_a_window_to_open(city, travel):
    assert simulate(city, travel, ["o1"]) == Schedule([Stop("o1", "school", 25, 30)], None)


def test_simulate_the_better_order_of_the_worked_example(city, travel):
    assert simulate(city, travel, ["o1", "o2"]) == Schedule(
        [Stop("o1", "school", 25, 30), Stop("o2", "market", 42, 42)],
        None,
    )


def test_simulate_the_other_order_of_the_worked_example(city, travel):
    assert simulate(city, travel, ["o2", "o1"]) == Schedule(
        [Stop("o2", "market", 37, 37), Stop("o1", "school", 49, 49)],
        None,
    )


def test_simulate_stops_at_a_late_order(city, travel):
    assert simulate(city, travel, ["o3", "o1", "o2"]) == Schedule(
        [Stop("o3", "bank", 44, 90)],
        "order o1 arrives at minute 155, after its window closes at 60",
    )


def test_simulate_stops_at_an_unreachable_order(city, travel):
    assert simulate(city, travel, ["o1", "o4"]) == Schedule(
        [Stop("o1", "school", 25, 30)],
        "order o4 can't be reached from school",
    )


def test_simulate_an_unreachable_first_order(city, travel):
    assert simulate(city, travel, ["o4"]) == Schedule(
        [],
        "order o4 can't be reached from warehouse",
    )


def test_simulate_starts_at_the_start_minute(city, travel):
    city.set_start("warehouse", 10)
    assert simulate(city, travel, ["o1"]) == Schedule([Stop("o1", "school", 35, 35)], None)


def test_arriving_as_the_window_closes_is_on_time(city, travel):
    city.set_start("warehouse", 35)
    assert simulate(city, travel, ["o1"]) == Schedule([Stop("o1", "school", 60, 60)], None)


def test_arriving_a_minute_after_the_window_closes_is_late(city, travel):
    city.set_start("warehouse", 36)
    assert simulate(city, travel, ["o1"]) == Schedule(
        [],
        "order o1 arrives at minute 61, after its window closes at 60",
    )


def test_simulate_from_another_start(city, travel):
    city.set_start("market", 0)
    assert simulate(city, travel, ["o2", "o3"]) == Schedule(
        [Stop("o2", "market", 0, 0), Stop("o3", "bank", 7, 90)],
        None,
    )
