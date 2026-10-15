"""Translation between our message types and the OpenAI Responses API, tested offline."""

from typing import Any, cast

from openai import AsyncOpenAI

from memagent.llm.openai_provider import (
    ERROR_PREFIX,
    OpenAIResponsesClient,
    from_responses_output,
    to_responses_input,
)
from memagent.llm.types import (
    AssistantMessage,
    SystemMessage,
    ToolCallMessage,
    ToolResultMessage,
    ToolSpec,
    UserMessage,
)
from tests.openai_fixtures import StubOpenAI, function_call_item, make_response, message_item


def test_to_responses_input_maps_every_message_type() -> None:
    instructions, items = to_responses_input(
        [
            SystemMessage("be helpful"),
            UserMessage("hi"),
            AssistantMessage("let me check"),
            ToolCallMessage("call_1", "web_fetch", {"url": "https://example.com"}),
            ToolResultMessage("call_1", "web_fetch", "page text", is_error=False),
            ToolCallMessage("call_2", "web_fetch", {"url": "x"}),
            ToolResultMessage("call_2", "web_fetch", "bad url", is_error=True),
        ]
    )
    assert instructions == "be helpful"
    plain = [dict(cast(dict[str, Any], item)) for item in items]
    assert plain[0] == {"role": "user", "content": "hi"}
    assert plain[1] == {"role": "assistant", "content": "let me check"}
    assert plain[2] == {
        "type": "function_call",
        "call_id": "call_1",
        "name": "web_fetch",
        "arguments": '{"url": "https://example.com"}',
    }
    assert plain[3] == {"type": "function_call_output", "call_id": "call_1", "output": "page text"}
    # function_call_output has no error flag, so the error is marked in the text.
    assert plain[5]["output"] == ERROR_PREFIX + "bad url"


def test_from_responses_output_reads_text_tool_calls_and_usage() -> None:
    response = make_response(
        [message_item("thinking out loud"), function_call_item("web_search", '{"query": "q"}')],
        input_tokens=1000,
        output_tokens=50,
        cached_tokens=600,
    )
    parsed = from_responses_output(response)
    assert parsed.text == "thinking out loud"
    assert len(parsed.tool_calls) == 1
    call = parsed.tool_calls[0]
    assert (call.call_id, call.name, call.arguments) == ("call_1", "web_search", {"query": "q"})
    assert call.arguments_error is None
    assert (parsed.usage.input_tokens, parsed.usage.output_tokens) == (1000, 50)
    assert parsed.usage.cached_input_tokens == 600


def test_malformed_tool_arguments_are_reported_not_raised() -> None:
    parsed = from_responses_output(make_response([function_call_item("echo", "{not json")]))
    assert parsed.tool_calls[0].arguments == {}
    assert parsed.tool_calls[0].arguments_error is not None

    parsed = from_responses_output(make_response([function_call_item("echo", "[1, 2]")]))
    assert parsed.tool_calls[0].arguments_error == "arguments must be a JSON object"


async def test_client_sends_only_function_tools_and_does_not_store() -> None:
    stub = StubOpenAI(make_response([message_item("hello")]))
    client = OpenAIResponsesClient(cast(AsyncOpenAI, stub))
    spec = ToolSpec("web_search", "Search the web.", {"type": "object", "properties": {}})

    result = await client.complete([UserMessage("hi")], [spec], model="configured-model")

    sent = stub.responses.kwargs
    assert sent["model"] == "configured-model"
    assert sent["store"] is False
    # Decision 0005: the agent model only ever gets our function tools, never hosted search.
    assert [tool["type"] for tool in sent["tools"]] == ["function"]
    assert result.text == "hello"
