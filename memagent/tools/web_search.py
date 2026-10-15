"""web_search: our own tool, backed by a pluggable SearchProvider.

The agent model only ever sees a plain function tool called `web_search`. The real
provider makes a *separate*, small OpenAI request with SEARCH_MODEL and OpenAI's hosted
`web_search` tool, capped at one search. Why not hand the hosted tool to the agent model
directly (decision 0005): we couldn't cap or count its searches per call, its results
would never pass through our tool_result events (so they'd skip scrubbing and the log),
and the agent would be tied to one provider's built-in tool.
"""

import json
import logging
import time
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Protocol, cast

import openai
from openai import AsyncOpenAI
from openai.types.responses import Response

from memagent import store
from memagent.llm.openai_provider import usage_from_response
from memagent.llm.types import ToolSpec, Usage
from memagent.pricing import PriceTable, cost_usd
from memagent.tools.base import ToolContext, ToolError, ToolOutput, require_str

logger = logging.getLogger(__name__)

MAX_RESULTS = 5
MAX_SNIPPET_CHARS = 300


@dataclass(frozen=True)
class SearchResult:
    title: str
    url: str
    snippet: str


@dataclass(frozen=True)
class SearchResponse:
    results: list[SearchResult]
    usage: Usage  # usage.search_calls = searches actually performed
    model: str


class SearchProvider(Protocol):
    async def search(self, query: str) -> SearchResponse: ...


# ---- fake provider (tests) -------------------------------------------------------------


class FakeSearchProvider:
    """Canned results, no network. `usage` defaults to one free search."""

    def __init__(
        self,
        results: list[SearchResult] | None = None,
        usage: Usage | None = None,
        model: str = "fake",
    ) -> None:
        self._results = results or [
            SearchResult("Example Domain", "https://example.com/", "An example page.")
        ]
        self._usage = usage or Usage(input_tokens=0, output_tokens=0, search_calls=1)
        self._model = model
        self.queries: list[str] = []

    async def search(self, query: str) -> SearchResponse:
        self.queries.append(query)
        return SearchResponse(results=self._results, usage=self._usage, model=self._model)


# ---- OpenAI hosted-search provider -----------------------------------------------------

_INSTRUCTIONS = (
    "You are a search backend. Run exactly one web search for the user's query and "
    f"return up to {MAX_RESULTS} of the most relevant results as JSON. Each result needs "
    "the page title, its URL, and a one- or two-sentence snippet of what the page says. "
    "Only include URLs that the search actually returned."
)

_RESULT_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {
        "results": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "url": {"type": "string"},
                    "snippet": {"type": "string"},
                },
                "required": ["title", "url", "snippet"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["results"],
    "additionalProperties": False,
}


def _normalize_url(url: str) -> str:
    return url.rstrip("/")


def count_searches(response: Response) -> int:
    """Number of hosted search calls actually made (this is what OpenAI bills per call)."""
    return sum(1 for item in response.output if item.type == "web_search_call")


def source_urls(response: Response) -> set[str]:
    """Every URL the search really returned: action sources plus url_citation annotations."""
    urls: set[str] = set()
    for item in response.output:
        if item.type == "web_search_call" and item.action.type == "search":
            urls.update(_normalize_url(s.url) for s in item.action.sources or [])
        elif item.type == "message":
            for content in item.content:
                if content.type == "output_text":
                    urls.update(
                        _normalize_url(a.url)
                        for a in content.annotations
                        if a.type == "url_citation"
                    )
    return urls


def parse_search_response(response: Response) -> list[SearchResult]:
    """Parse the model's JSON and keep only results whose URL the search actually returned.

    The URL check guards against the search model inventing plausible-looking links.
    """
    try:
        data: Any = json.loads(response.output_text or "{}")
    except json.JSONDecodeError as exc:
        raise ToolError("search backend returned malformed results") from exc
    known = source_urls(response)
    payload = cast(dict[str, Any], data) if isinstance(data, dict) else {}
    raw_results: list[Any] = payload.get("results") or []
    results: list[SearchResult] = []
    for raw in raw_results:
        if not isinstance(raw, dict):
            continue
        item = cast(dict[str, Any], raw)
        title, url, snippet = item.get("title"), item.get("url"), item.get("snippet")
        if not (isinstance(title, str) and isinstance(url, str) and isinstance(snippet, str)):
            continue
        if _normalize_url(url) not in known:
            continue
        results.append(SearchResult(title, url, snippet[:MAX_SNIPPET_CHARS]))
    return results[:MAX_RESULTS]


class OpenAISearchProvider:
    def __init__(self, client: AsyncOpenAI, model: str) -> None:
        self._client = client
        self._model = model

    async def search(self, query: str) -> SearchResponse:
        try:
            response = await self._client.responses.create(
                model=self._model,
                instructions=_INSTRUCTIONS,
                input=query,
                tools=[{"type": "web_search"}],
                # The API-level cap: at most one hosted tool call in this request.
                max_tool_calls=1,
                include=["web_search_call.action.sources"],
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "search_results",
                        "schema": _RESULT_SCHEMA,
                        "strict": True,
                    }
                },
                store=False,
            )
        except openai.OpenAIError as exc:
            raise ToolError(f"search request failed: {type(exc).__name__}") from exc

        searches = count_searches(response)
        if searches > 1:
            # Should not happen with max_tool_calls=1; record the true count regardless.
            logger.warning("hosted search made %d calls despite the cap of 1", searches)
        base = usage_from_response(response)
        usage = Usage(
            input_tokens=base.input_tokens,
            output_tokens=base.output_tokens,
            cached_input_tokens=base.cached_input_tokens,
            search_calls=searches,
        )
        return SearchResponse(
            results=parse_search_response(response), usage=usage, model=self._model
        )


# ---- the tool --------------------------------------------------------------------------


def format_results(results: list[SearchResult]) -> str:
    if not results:
        return "No results."
    return "\n\n".join(
        f"{i}. {r.title}\n   {r.url}\n   {r.snippet}" for i, r in enumerate(results, start=1)
    )


class WebSearchTool:
    def __init__(self, provider: SearchProvider, prices: PriceTable) -> None:
        self._provider = provider
        self._prices = prices

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="web_search",
            description="Search the web. Returns a short list of results (title, URL, snippet).",
            parameters={
                "type": "object",
                "properties": {"query": {"type": "string", "description": "Search query"}},
                "required": ["query"],
                "additionalProperties": False,
            },
        )

    async def run(self, ctx: ToolContext, args: dict[str, Any]) -> ToolOutput:
        query = require_str(args, "query")
        started = time.monotonic()
        response = await self._provider.search(query)
        latency_ms = int((time.monotonic() - started) * 1000)
        cost: Decimal = cost_usd(self._prices, response.model, response.usage)
        async with ctx.sessionmaker() as db:
            await store.record_llm_call(
                db,
                user_id=ctx.user_id,
                session_id=ctx.session_id,
                module="search",
                model=response.model,
                usage=response.usage,
                cost_usd=cost,
                latency_ms=latency_ms,
            )
        return ToolOutput(format_results(response.results), cost_usd=cost)
