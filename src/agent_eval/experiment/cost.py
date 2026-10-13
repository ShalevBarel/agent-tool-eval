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
    raise NotImplementedError
