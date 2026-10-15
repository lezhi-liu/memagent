# memagent

A multi-user, general-purpose AI agent with long-term memory. This is a learning and job-search project: the author must be able to read every line and explain it in an interview.

## How to work

- One slice at a time. The current slice spec lives in `docs/slices/S<n>.md`; implement only what it says. If something from a later slice seems necessary, stop and ask instead of building it early.
- Plan before code. The plan lists: files to create or change, table schemas, key function signatures, and the test list.
- Readability over cleverness: small functions, explicit types, comments that explain "why", not "what".
- A slice is done only when it delivers:
  - all tests passing, with no ruff or pyright errors
  - `docs/walkthrough/S<n>.md`: suggested reading order, the full code path of one request, and the reason behind each key design choice
  - `docs/decisions/NNNN-<title>.md`: one record per key decision in the slice (context, options, choice, cost)
  - run instructions in the README
- The full design is in `docs/design.md`. Consult it only when you need background; do not load the whole file.

## Hard constraints (no slice may break these)

- Multi-tenancy: every business table has `user_id`; every query filters by `user_id`; accessing another user's resource always returns 404.
- Scrub before persisting: everything that stores user content (events, workspace files, logs, traces) goes through the single scrubbing gateway `memagent.sanitize`. Nothing writes to the database or disk around it.
- Append-only raw log: UPDATE and DELETE on `events` are forbidden (enforced by a database trigger); the only exception is the redaction function used for deletion.
- Tests cost nothing: tests use FakeLLM and a fake search provider only, never a paid API. Real APIs are for manual smoke tests only.
- Visible spend: every LLM call writes one `llm_calls` row (module, model, input / output / cached tokens, USD cost, latency).
- Budget: the project API budget is $50/month. Before any command that runs evals or batch jobs against a real API, estimate the cost and get approval.
- No secrets in code or logs; configuration comes from environment variables (`.env`, listed in .gitignore).

## Stack

Python 3.12 · uv · FastAPI · Pydantic v2 · SQLAlchemy 2.0 (async) + Alembic · PostgreSQL 16 + pgvector (Docker Compose) · httpx · Typer · pytest + pytest-asyncio · ruff · pyright (strict)

## Common commands

- `docker compose up -d db`: start the database
- `uv run alembic upgrade head`: run migrations
- `uv run pytest`: run tests
- `uv run ruff check . && uv run pyright`: static checks
- `uv run memagent --help`: CLI entry point
