"""A stand-in for the Anthropic client, so the agent loop runs in tests without the API.

The fake replays responses that a test scripts in advance, and keeps a copy
of every request it gets. The responses are real SDK objects, so the loop
reads them exactly as it reads the API's.
"""

import copy

from anthropic.types import Message, TextBlock, ThinkingBlock, ToolUseBlock, Usage


class FakeMessages:
    """The client's messages.create, replaying scripted responses in order."""

    def __init__(self, responses: list[Message]) -> None:
        self.responses = list(responses)
        self.requests: list[dict] = []

    def create(self, **request) -> Message:
        # A deep copy, so later changes to the loop's message list don't show up here.
        self.requests.append(copy.deepcopy(request))
        assert self.responses, "the loop asked for more responses than the test scripted"
        return self.responses.pop(0)


class FakeClient:
    """Has the one part of the Anthropic client that the loop uses."""

    def __init__(self, responses: list[Message]) -> None:
        self.messages = FakeMessages(responses)


def response(*content, stop_reason="tool_use", input_tokens=100, output_tokens=10) -> Message:
    """A response from the model with the given content blocks."""
    return Message(
        id="msg_test",
        type="message",
        role="assistant",
        model="claude-haiku-5-5",
        content=list(content),
        stop_reason=stop_reason,
        stop_sequence=None,
        usage=Usage(input_tokens=input_tokens, output_tokens=output_tokens),
    )


def tool_use(name: str, tool_input: dict, block_id: str = "toolu_1") -> ToolUseBlock:
    """A tool call by the model."""
    return ToolUseBlock(type="tool_use", id=block_id, name=name, input=tool_input)


def text(words: str) -> TextBlock:
    """Text the model writes."""
    return TextBlock(type="text", text=words)


def thinking(signature: str = "sig") -> ThinkingBlock:
    """A thinking block as the API returns it by default: no text, only a signature."""
    return ThinkingBlock(type="thinking", thinking="", signature=signature)
