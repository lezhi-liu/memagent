"""Hand-built OpenAI Response objects and a stub client, so provider code is tested offline."""

from typing import Any

from openai.types.responses import Response, ResponseUsage
from openai.types.responses.response_output_item import ResponseOutputItem
from pydantic import TypeAdapter

_item_adapter: TypeAdapter[ResponseOutputItem] = TypeAdapter(ResponseOutputItem)


def message_item(text: str, cited_urls: tuple[str, ...] = ()) -> ResponseOutputItem:
    annotations = [
        {"type": "url_citation", "url": url, "title": url, "start_index": 0, "end_index": 1}
        for url in cited_urls
    ]
    return _item_adapter.validate_python(
        {
            "type": "message",
            "id": "msg_1",
            "role": "assistant",
            "status": "completed",
            "content": [{"type": "output_text", "text": text, "annotations": annotations}],
        }
    )


def function_call_item(name: str, arguments: str, call_id: str = "call_1") -> ResponseOutputItem:
    return _item_adapter.validate_python(
        {"type": "function_call", "call_id": call_id, "name": name, "arguments": arguments}
    )


def web_search_item(source_urls: tuple[str, ...] = (), item_id: str = "ws_1") -> ResponseOutputItem:
    return _item_adapter.validate_python(
        {
            "type": "web_search_call",
            "id": item_id,
            "status": "completed",
            "action": {
                "type": "search",
                "query": "q",
                "sources": [{"type": "url", "url": url} for url in source_urls],
            },
        }
    )


def make_response(
    output: list[ResponseOutputItem],
    input_tokens: int = 100,
    output_tokens: int = 20,
    cached_tokens: int = 0,
    model: str = "test-model-2025-01-01",
) -> Response:
    usage = ResponseUsage.model_validate(
        {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "input_tokens_details": {"cached_tokens": cached_tokens, "cache_write_tokens": 0},
            "output_tokens_details": {"reasoning_tokens": 0},
        }
    )
    # model_construct skips validation of the many top-level fields we don't use.
    return Response.model_construct(output=output, model=model, usage=usage)


class StubResponses:
    def __init__(self, response: Response) -> None:
        self._response = response
        self.kwargs: dict[str, Any] = {}

    async def create(self, **kwargs: Any) -> Response:
        self.kwargs = kwargs
        return self._response


class StubOpenAI:
    """Stands in for AsyncOpenAI: records the request, returns a canned response."""

    def __init__(self, response: Response) -> None:
        self.responses = StubResponses(response)
