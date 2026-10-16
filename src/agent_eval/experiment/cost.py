"""What the tokens of a run cost."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Price:
    """A model's prices in dollars per million tokens.

    Attributes:
        input: Input read without the cache.
        output: Output, thinking included.
        cache_write: Input written to the cache.
        cache_read: Input read from the cache.
    """

    input: float
    output: float
    cache_write: float
    cache_read: float


# Dollars per million tokens, as (input, output). Thinking is billed as output.
PRICES = {
    "claude-haiku-5-5": Price(input=0.10, output=0.50, cache_write=0.125, cache_read=0.01),
    "claude-sonnet-5-5": Price(input=2.00, output=10.00, cache_write=2.50, cache_read=0.10),
}


def cost_usd(
    model: str,
    input_tokens: int,
    output_tokens: int,
    cache_write_tokens: int = 0,
    cache_read_tokens: int = 0,
) -> float:
    """Return the cost in dollars of input and output tokens on a model.

    Raises:
        ValueError: If the model has no price.
    """
    if model not in PRICES:
        raise ValueError(f"no price for model {model}")
    price = PRICES[model]
    total = (
        input_tokens * price.input
        + output_tokens * price.output
        + cache_write_tokens * price.cache_write
        + cache_read_tokens * price.cache_read
    )
    return total / 1_000_000
