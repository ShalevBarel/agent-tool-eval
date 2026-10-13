"""What the tokens of a run cost."""

# Dollars per million tokens, as (input, output). Thinking is billed as output.
PRICES = {
    "claude-haiku-5-5": (0.10, 0.50),
    "claude-sonnet-5-5": (2.00, 10.00),
}


def cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    """Return the cost in dollars of input and output tokens on a model.

    Raises:
        ValueError: If the model has no price.
    """
    if model not in PRICES:
        raise ValueError(f"no price for model {model}")
    return (input_tokens * PRICES[model][0] + output_tokens * PRICES[model][1]) / 1_000_000
