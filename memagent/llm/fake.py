"""FakeLLM: a scripted LLMClient for tests. Never touches the network."""

import asyncio
from collections.abc import Callable, Sequence

from memagent.llm.types import LLMResponse, Message, ToolCall, ToolSpec, Usage

FAKE_MODEL = "fake"

type Script = Sequence[LLMResponse] | Callable[[Sequence[Message]], LLMResponse]


def reply(text: str, input_tokens: int = 10, output_tokens: int = 5) -> LLMResponse:
    """A final answer with no tool calls."""
    return LLMResponse(
        text=text, usage=Usage(input_tokens, output_tokens), model=FAKE_MODEL, tool_calls=[]
    )


def call_tool(name: str, call_id: str = "call_1", **arguments: object) -> LLMResponse:
    """A response asking for exactly one tool."""
    return LLMResponse(
        text="",
        usage=Usage(10, 5),
        model=FAKE_MODEL,
        tool_calls=[ToolCall(call_id=call_id, name=name, arguments=dict(arguments))],
    )


class FakeLLM:
    """Returns scripted responses in order, or computes one from the input messages.

    `calls` records every input so tests can check what the model was shown.
    `delay` makes each call await, so concurrency tests can force interleaving.
    """

    def __init__(self, script: Script, delay: float = 0.0) -> None:
        self._script = script
        self._delay = delay
        self._index = 0
        self.calls: list[list[Message]] = []

    async def complete(
        self, messages: Sequence[Message], tools: Sequence[ToolSpec], model: str
    ) -> LLMResponse:
        self.calls.append(list(messages))
        if self._delay:
            await asyncio.sleep(self._delay)
        if callable(self._script):
            return self._script(messages)
        if self._index >= len(self._script):
            raise AssertionError(f"FakeLLM script exhausted after {self._index} calls")
        response = self._script[self._index]
        self._index += 1
        return response
