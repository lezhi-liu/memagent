"""Visible spend: every model call, agent or search, writes exactly one llm_calls row."""

import uuid
from decimal import Decimal

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncEngine

from memagent.agent.loop import run_turn
from memagent.db.engine import SessionMaker
from memagent.db.models import LLMCall
from memagent.llm.fake import FAKE_MODEL, FakeLLM, call_tool, reply
from memagent.llm.types import LLMResponse, Usage
from memagent.tools.web_search import FakeSearchProvider, WebSearchTool
from tests.conftest import TEST_PRICES, MakeUser, make_deps, new_session


async def _rows(sessionmaker: SessionMaker, session_id: uuid.UUID) -> list[LLMCall]:
    async with sessionmaker() as db:
        result = await db.execute(
            select(LLMCall).where(LLMCall.session_id == session_id).order_by(LLMCall.id)
        )
        return list(result.scalars())


async def test_each_fake_llm_call_writes_one_agent_row(
    make_user: MakeUser, engine: AsyncEngine, sessionmaker: SessionMaker
) -> None:
    user, _ = await make_user()
    session_id = await new_session(sessionmaker, user)
    with_cache = LLMResponse(
        text="done",
        usage=Usage(input_tokens=1000, output_tokens=200, cached_input_tokens=400),
        model=FAKE_MODEL,
    )
    llm = FakeLLM([call_tool("workspace_list"), call_tool("workspace_list"), with_cache])

    result = await run_turn(make_deps(engine, sessionmaker, llm), user.id, session_id, "go")

    rows = await _rows(sessionmaker, session_id)
    assert len(rows) == len(llm.calls) == 3
    assert {r.module for r in rows} == {"agent"}
    assert all(r.model == FAKE_MODEL and r.user_id == user.id for r in rows)
    last = rows[-1]
    assert (last.input_tokens, last.output_tokens, last.cached_input_tokens) == (1000, 200, 400)
    # 600 uncached * $1 + 400 cached * $0.10 + 200 output * $4, per million tokens
    assert last.cost_usd == Decimal("0.001440")
    assert result.cost_usd == sum((r.cost_usd for r in rows), Decimal(0))


async def test_web_search_writes_a_search_row(
    make_user: MakeUser, engine: AsyncEngine, sessionmaker: SessionMaker
) -> None:
    user, _ = await make_user()
    session_id = await new_session(sessionmaker, user)
    search = FakeSearchProvider(
        usage=Usage(input_tokens=500, output_tokens=100, search_calls=1), model=FAKE_MODEL
    )
    llm = FakeLLM([call_tool("web_search", query="postgres advisory locks"), reply("found it")])
    deps = make_deps(engine, sessionmaker, llm, tools=[WebSearchTool(search, TEST_PRICES)])

    result = await run_turn(deps, user.id, session_id, "search please")

    assert search.queries == ["postgres advisory locks"]
    rows = await _rows(sessionmaker, session_id)
    assert [r.module for r in rows] == ["agent", "search", "agent"]
    search_row = rows[1]
    assert search_row.search_calls == 1
    # 500 * $1 + 100 * $4 per million tokens, plus one $0.01 search
    assert search_row.cost_usd == Decimal("0.010900")
    # The turn's reported cost includes the search.
    assert result.cost_usd == sum((r.cost_usd for r in rows), Decimal(0))
