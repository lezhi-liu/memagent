"""The project's own LLM types.

Only `memagent/llm/openai_provider.py` knows OpenAI's formats. Everything else (agent loop,
history rebuild, tools, tests) speaks these types, so swapping providers touches one file.
"""

from dataclasses import dataclass, field
from typing import Any

# JSON as it comes out of json.loads. Kept as `Any` leaves on purpose: a recursive
# alias makes every tool's argument access noisy without catching real bugs.
type JsonDict = dict[str, Any]


@dataclass(frozen=True)
class SystemMessage:
    text: str


@dataclass(frozen=True)
class UserMessage:
    text: str


@dataclass(frozen=True)
class AssistantMessage:
    text: str


@dataclass(frozen=True)
class ToolCallMessage:
    """The model asked for a tool. Stored so the next request shows the model its own call."""

    call_id: str
    name: str
    arguments: JsonDict


@dataclass(frozen=True)
class ToolResultMessage:
    call_id: str
    name: str
    output: str
    is_error: bool


type Message = SystemMessage | UserMessage | AssistantMessage | ToolCallMessage | ToolResultMessage


@dataclass(frozen=True)
class ToolSpec:
    """What the model is told about a tool. `parameters` is a JSON Schema object."""

    name: str
    description: str
    parameters: JsonDict


@dataclass(frozen=True)
class ToolCall:
    call_id: str
    name: str
    arguments: JsonDict
    # Set when the model produced arguments that are not a JSON object. The loop turns
    # this into an error tool_result instead of crashing.
    arguments_error: str | None = None


@dataclass(frozen=True)
class Usage:
    input_tokens: int
    output_tokens: int
    # Part of input_tokens that was served from the prompt cache (billed cheaper).
    cached_input_tokens: int = 0
    search_calls: int = 0


@dataclass(frozen=True)
class LLMResponse:
    text: str
    usage: Usage
    model: str
    tool_calls: list[ToolCall] = field(default_factory=list[ToolCall])
