# 0005: Wrap OpenAI's hosted web search behind our own `web_search` tool

- Status: accepted (S1)
- Code: `memagent/tools/web_search.py`, `memagent/llm/openai_provider.py`

## Context

OpenAI's Responses API offers a hosted `web_search` tool. The simplest option is to list it in the agent model's `tools` and let the model search whenever it wants. We need search, but we also have a $50/month budget, a rule that every model call is visible in `llm_calls`, a rule that everything persisted goes through scrubbing and the event log, and a design that stays provider-switchable.

## Options

1. **Give the hosted tool to the agent model directly.**
2. **Our own `web_search` function tool**, executed by our loop like every other tool. Its provider makes a separate, small OpenAI request with `SEARCH_MODEL`, the hosted search tool, and `max_tool_calls=1`, and returns a short title/URL/snippet list.
3. **A non-OpenAI search API** (Brave, Tavily, Bing). This is a reasonable later swap, but it needs a second key and a second bill.

## Choice

Option 2.

- **Spend control.** With option 1, the agent model decides how many searches to run inside one call, and each one is billed. Here each `web_search` tool call is capped at one search by the API (`max_tool_calls=1`). We count the actual `web_search_call` items and record them in `llm_calls.search_calls`. The step limit then bounds total searches per turn.
- **Visibility.** Search gets its own `llm_calls` row with `module='search'`, its own model, tokens and per-search fee. That makes "search costs" vs "agent costs" a simple GROUP BY.
- **One path for content.** Results come back as a normal `tool_result` event, so they are scrubbed, logged in order, and replayed into history like every other tool output. Hosted results would live only inside OpenAI's response, outside our log.
- **Cheaper model for search.** `SEARCH_MODEL` can be a small model while `AGENT_MODEL` is a strong one.
- **Provider independence.** The agent model only sees a plain function tool. Swapping in another search API changes one class, `SearchProvider`.
- **Hallucinated links are filtered.** Results whose URL is not among the search's real sources or citations are dropped.

## Cost

- **An extra model round-trip per search**: more latency, and the search model's tokens on top of the agent's.
- The agent sees only short snippets, not the full search context the hosted tool could give the agent model. Deeper reading needs an explicit `web_fetch`.
- The search model formats results as JSON. A malformed response becomes an error result rather than a partial answer.
- One more prompt (`_INSTRUCTIONS`) to maintain.
