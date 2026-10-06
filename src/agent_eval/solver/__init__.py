"""Exact solvers that find the best delivery order for a task."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Solution:
    """A delivery order that finishes earliest, and the minute of its last delivery."""

    order_ids: list[str]
    finish_minute: int
