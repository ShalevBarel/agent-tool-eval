import pytest

from agent_eval.experiment.cost import PRICES, Price, cost_usd


def test_prices_from_the_pricing_page():
    assert PRICES == {
        "claude-haiku-5-5": Price(input=0.10, output=0.50, cache_write=0.125, cache_read=0.01),
        "claude-sonnet-5-5": Price(input=2.00, output=10.00, cache_write=2.50, cache_read=0.10),
    }


def test_a_million_input_tokens():
    assert cost_usd("claude-haiku-5-5", 1_000_000, 0) == pytest.approx(0.10)


def test_a_million_output_tokens():
    assert cost_usd("claude-haiku-5-5", 0, 1_000_000) == pytest.approx(0.50)


def test_a_million_tokens_written_to_the_cache():
    assert cost_usd("claude-haiku-5-5", 0, 0, 1_000_000, 0) == pytest.approx(0.125)


def test_a_million_tokens_read_from_the_cache():
    assert cost_usd("claude-sonnet-5-5", 0, 0, 0, 1_000_000) == pytest.approx(0.10)


def test_cache_tokens_are_optional():
    assert cost_usd("claude-haiku-5-5", 19, 41) == cost_usd("claude-haiku-5-5", 19, 41, 0, 0)


def test_all_four_add_up():
    assert cost_usd("claude-haiku-5-5", 2210, 1460, 3900, 9100) == pytest.approx(0.0015295)


def test_each_model_has_its_own_price():
    assert cost_usd("claude-sonnet-5-5", 1000, 100, 2000, 10_000) == pytest.approx(0.009)


def test_no_tokens_cost_nothing():
    for model in PRICES:
        assert cost_usd(model, 0, 0, 0, 0) == 0


def test_unknown_model():
    with pytest.raises(ValueError, match="no price for model claude-unknown"):
        cost_usd("claude-unknown", 100, 10)
