"""OpenAISearchProvider, tested offline against hand-built responses."""

import json
from typing import cast

from openai import AsyncOpenAI

from memagent.tools.web_search import (
    OpenAISearchProvider,
    SearchResult,
    count_searches,
    format_results,
    parse_search_response,
)
from tests.openai_fixtures import StubOpenAI, make_response, message_item, web_search_item


def _results_json(*urls: str) -> str:
    return json.dumps(
        {"results": [{"title": f"T {u}", "url": u, "snippet": f"about {u}"} for u in urls]}
    )


async def test_provider_caps_searches_and_counts_them() -> None:
    response = make_response(
        [
            web_search_item(("https://a.example/",)),
            message_item(_results_json("https://a.example/")),
        ],
        input_tokens=300,
        output_tokens=40,
    )
    stub = StubOpenAI(response)
    provider = OpenAISearchProvider(cast(AsyncOpenAI, stub), model="search-model")

    result = await provider.search("what is pgvector")

    sent = stub.responses.kwargs
    assert sent["model"] == "search-model"
    assert sent["max_tool_calls"] == 1  # the API-level cap
    assert sent["tools"] == [{"type": "web_search"}]
    assert sent["store"] is False
    assert result.usage.search_calls == 1
    assert (result.usage.input_tokens, result.usage.output_tokens) == (300, 40)
    assert result.model == "search-model"  # priced by the configured name
    assert result.results == [
        SearchResult("T https://a.example/", "https://a.example/", "about https://a.example/")
    ]


def test_search_count_is_the_real_number_of_calls() -> None:
    response = make_response(
        [web_search_item(item_id="ws_1"), web_search_item(item_id="ws_2"), message_item("{}")]
    )
    assert count_searches(response) == 2


def test_results_with_urls_the_search_never_returned_are_dropped() -> None:
    response = make_response(
        [
            web_search_item(("https://real.example/",)),
            message_item(
                _results_json(
                    "https://real.example", "https://cited.example/", "https://made-up.example/"
                ),
                cited_urls=("https://cited.example/",),
            ),
        ]
    )
    urls = [r.url for r in parse_search_response(response)]
    # "real" matches a source despite the trailing slash; "made-up" is not a known source.
    assert urls == ["https://real.example", "https://cited.example/"]


def test_format_results() -> None:
    assert format_results([]) == "No results."
    text = format_results([SearchResult("Title", "https://x.example/", "Snippet")])
    assert text == "1. Title\n   https://x.example/\n   Snippet"
