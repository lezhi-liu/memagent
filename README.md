# memagent

A multi-user, general-purpose AI agent with long-term memory, built one slice at a time.

**Current slice: S1.** Multi-user agent loop and append-only raw log. The agent can search the web, fetch pages, and read and write files in a per-user workspace. Every event is stored per user and scrubbed of secrets first. There is no memory yet: each new session starts blank.

- Design: `docs/design.md` · Slice spec: `docs/slices/S1.md`
- How S1 works: `docs/walkthrough/S1.md` · Decisions: `docs/decisions/`

## From a clean machine to the first conversation

Prerequisites: [uv](https://docs.astral.sh/uv/), Docker (with Compose), and an OpenAI API key.

```bash
# 1. Install Python 3.12 and the dependencies
uv sync

# 2. Start Postgres 16 (+ pgvector). The first start also creates the memagent_test database.
docker compose up -d db

# 3. Configure
cp .env.example .env
#    then edit .env: set OPENAI_API_KEY, AGENT_MODEL and SEARCH_MODEL,
#    and add a price entry for both models to config/prices.toml (the server refuses
#    to start without one, so spend is never untracked)

# 4. Create the schema
uv run alembic upgrade head

# 5. Create a user (prints its API key once; only a hash is stored)
uv run memagent user create alice

# 6. Start the API (leave it running)
uv run uvicorn --factory memagent.api.app:create_default_app

# 7. In another terminal: chat
MEMAGENT_API_KEY=mk_... uv run memagent chat
```

During a chat, the CLI shows each tool the agent called and the turn's cost. Type `/exit` to quit.

## API

All endpoints require `Authorization: Bearer <api_key>`. Another user's session returns 404.

| Method | Path | Returns |
|---|---|---|
| `POST` | `/sessions` | a new session `{id, created_at}` |
| `POST` | `/sessions/{id}/messages` with `{"text": "..."}` | `{reply, steps, cost_usd, stopped_by_limit}` |
| `GET` | `/sessions/{id}/events` | the session's (scrubbed) event log |

## Tests and checks

The tests use FakeLLM and a fake search provider only. They make no paid calls and need no `.env`, but they do need the database running.

```bash
docker compose up -d db
uv run pytest
uv run ruff check . && uv run pyright
```

The tests rebuild the schema in `memagent_test` on every run. Override the target with `TEST_DATABASE_URL`.

## Configuration

| Variable | Required | Default |
|---|---|---|
| `OPENAI_API_KEY` | to run the server | none |
| `AGENT_MODEL` | to run the server | none (never hardcoded) |
| `SEARCH_MODEL` | to run the server | none (never hardcoded) |
| `DATABASE_URL` | no | `postgresql+asyncpg://memagent:memagent@localhost:5433/memagent` |
| `MAX_STEPS` | no | `10` |
| `WORKSPACE_ROOT` | no | `data/workspaces` |
| `PRICES_PATH` | no | `config/prices.toml` |
| `TEST_DATABASE_URL` | no | the `memagent_test` database above |
| `MEMAGENT_API_URL` / `MEMAGENT_API_KEY` | for `memagent chat` | `http://127.0.0.1:8000` / none |

## Seeing what it cost

```sql
SELECT module, model, count(*) AS calls, sum(input_tokens) AS input, sum(output_tokens) AS output,
       sum(search_calls) AS searches, sum(cost_usd) AS usd
FROM llm_calls GROUP BY module, model ORDER BY module;
```

Run it with `docker compose exec db psql -U memagent -c "<query>"`.
