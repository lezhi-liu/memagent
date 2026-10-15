"""The interface every LLM provider implements."""

from collections.abc import Sequence
from typing import Protocol

from memagent.llm.types import LLMResponse, Message, ToolSpec


class LLMError(RuntimeError):
    """The provider failed after its own retries. The loop records it as an error event."""


class LLMClient(Protocol):
    async def complete(
        self, messages: Sequence[Message], tools: Sequence[ToolSpec], model: str
    ) -> LLMResponse: ...
