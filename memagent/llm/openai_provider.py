"""OpenAI Responses API provider for the agent model.

This is the only module that knows OpenAI's message and tool-call formats. The two pure
functions `to_responses_input` and `from_responses_output` do the translation and are
unit-tested without network access.

Choices (see docs/walkthrough/S1.md):
- `store=False` and no `previous_response_id`: history is rebuilt from our own (scrubbed)
  events every call, so OpenAI-side conversation state would be a second, unscrubbed copy.
- Reasoning items are not round-tripped. They are provider-specific and opaque; the cost
  is that a reasoning model re-derives its reasoning between tool steps.
"""

import json
from collections.abc import Sequence
from typing import Any, cast

import openai
from openai import AsyncOpenAI
from openai.types.responses import (
    EasyInputMessageParam,
    FunctionToolParam,
    Response,
    ResponseFunctionToolCallParam,
    ResponseInputItemParam,
)
from openai.types.responses.response_input_item_param import FunctionCallOutput

from memagent.llm.base import LLMError
from memagent.llm.types import (
    AssistantMessage,
    LLMResponse,
    Message,
    SystemMessage,
    ToolCall,
    ToolCallMessage,
    ToolResultMessage,
    ToolSpec,
    Usage,
    UserMessage,
)

# function_call_output has no error flag, so errors are marked in the text itself.
ERROR_PREFIX = "ERROR: "


def to_responses_input(
    messages: Sequence[Message],
) -> tuple[str | None, list[ResponseInputItemParam]]:
    """Our messages -> (instructions, input items). System messages become `instructions`."""
    instructions: list[str] = []
    items: list[ResponseInputItemParam] = []
    for message in messages:
        match message:
            case SystemMessage(text=text):
                instructions.append(text)
            case UserMessage(text=text):
                items.append(EasyInputMessageParam(role="user", content=text))
            case AssistantMessage(text=text):
                items.append(EasyInputMessageParam(role="assistant", content=text))
            case ToolCallMessage(call_id=call_id, name=name, arguments=arguments):
                items.append(
                    ResponseFunctionToolCallParam(
                        type="function_call",
                        call_id=call_id,
                        name=name,
                        arguments=json.dumps(arguments, ensure_ascii=False),
                    )
                )
            case ToolResultMessage(call_id=call_id, output=output, is_error=is_error):
                items.append(
                    FunctionCallOutput(
                        type="function_call_output",
                        call_id=call_id,
                        output=(ERROR_PREFIX + output) if is_error else output,
                    )
                )
    return ("\n\n".join(instructions) or None), items


def to_function_tools(tools: Sequence[ToolSpec]) -> list[FunctionToolParam]:
    return [
        FunctionToolParam(
            type="function",
            name=tool.name,
            description=tool.description,
            parameters=tool.parameters,
            # Non-strict: strict mode requires every property to be listed as required,
            # which would make optional tool arguments awkward.
            strict=False,
        )
        for tool in tools
    ]


def _parse_arguments(raw: str) -> tuple[dict[str, Any], str | None]:
    """Model-produced JSON can be malformed; report that instead of raising."""
    try:
        parsed: Any = json.loads(raw) if raw else {}
    except json.JSONDecodeError as exc:
        return {}, f"arguments are not valid JSON: {exc.msg}"
    if not isinstance(parsed, dict):
        return {}, "arguments must be a JSON object"
    return cast(dict[str, Any], parsed), None


def usage_from_response(response: Response) -> Usage:
    if response.usage is None:
        return Usage(input_tokens=0, output_tokens=0)
    return Usage(
        input_tokens=response.usage.input_tokens,
        output_tokens=response.usage.output_tokens,
        cached_input_tokens=response.usage.input_tokens_details.cached_tokens,
    )


def from_responses_output(response: Response) -> LLMResponse:
    texts: list[str] = []
    tool_calls: list[ToolCall] = []
    for item in response.output:
        if item.type == "message":
            for content in item.content:
                if content.type == "output_text":
                    texts.append(content.text)
        elif item.type == "function_call":
            arguments, error = _parse_arguments(item.arguments)
            tool_calls.append(
                ToolCall(
                    call_id=item.call_id,
                    name=item.name,
                    arguments=arguments,
                    arguments_error=error,
                )
            )
    return LLMResponse(
        text="".join(texts),
        tool_calls=tool_calls,
        usage=usage_from_response(response),
        model=response.model,
    )


class OpenAIResponsesClient:
    def __init__(self, client: AsyncOpenAI) -> None:
        self._client = client

    async def complete(
        self, messages: Sequence[Message], tools: Sequence[ToolSpec], model: str
    ) -> LLMResponse:
        instructions, items = to_responses_input(messages)
        try:
            response = await self._client.responses.create(
                model=model,
                instructions=instructions,
                input=items,
                tools=to_function_tools(tools),
                store=False,
            )
        except openai.OpenAIError as exc:
            # The SDK has already retried transient errors. Only the class name is kept:
            # the message can echo request content, and logs must not hold user content.
            raise LLMError(f"OpenAI request failed: {type(exc).__name__}") from exc
        return from_responses_output(response)
