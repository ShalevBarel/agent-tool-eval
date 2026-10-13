import pytest

from agent_eval.experiment.cost import PRICES, cost_usd


def test_a_million_input_tokens():
    assert cost_usd("claude-haiku-5-5", 1_000_000, 0) == pytest.approx(0.10)


def test_a_million_output_tokens():
    assert cost_usd("claude-haiku-5-5", 0, 1_000_000) == pytest.approx(0.50)


def test_input_and_output_add_up():
    assert cost_usd("claude-haiku-5-5", 19, 41) == pytest.approx(0.0000224)


def test_each_model_has_its_own_price():
    assert cost_usd("claude-sonnet-5-5", 1000, 100) == pytest.approx(0.003)


def test_no_tokens_cost_nothing():
    for model in PRICES:
        assert cost_usd(model, 0, 0) == 0


def test_unknown_model():
    with pytest.raises(ValueError, match="no price for model claude-unknown"):
        cost_usd("claude-unknown", 100, 10)
