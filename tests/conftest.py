"""Shared fixtures.

Tests run against a separate `memagent_test` database (created by docker/initdb) and
never read `.env`: no API keys, no paid calls. The schema is rebuilt once per run with
`alembic downgrade base` + `upgrade head`. Tests don't clean up rows (events can't be
deleted anyway); each test creates its own users, so tests never see each other's data.
"""

import argparse
import os
import uuid
from collections.abc import AsyncGenerator, Awaitable, Callable, Sequence
from decimal import Decimal
from pathlib import Path

import httpx
import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy.exc import DBAPIError
from sqlalchemy.ext.asyncio import AsyncEngine

from memagent import store
from memagent.agent.loop import AgentDeps
from memagent.api.app import create_app
from memagent.auth import generate_api_key, hash_api_key
from memagent.config import Settings
from memagent.db.engine import SessionMaker, make_engine, make_sessionmaker
from memagent.db.models import User
from memagent.llm.base import LLMClient
from memagent.llm.fake import FAKE_MODEL
from memagent.pricing import ModelPrice, PriceTable
from memagent.tools.base import Tool, ToolContext, ToolRegistry
from memagent.tools.web_search import FakeSearchProvider, SearchProvider

TEST_DATABASE_URL = os.environ.get(
    "TEST_DATABASE_URL",
    "postgresql+asyncpg://memagent:memagent@localhost:5433/memagent_test",
)
REPO_ROOT = Path(__file__).resolve().parent.parent

# Non-zero fake prices so cost arithmetic is actually exercised.
TEST_PRICES = PriceTable(
    models={
        FAKE_MODEL: ModelPrice(
            input_per_mtok=Decimal("1.00"),
            cached_input_per_mtok=Decimal("0.10"),
            output_per_mtok=Decimal("4.00"),
        )
    },
    search_per_call_usd=Decimal("0.01"),
)


@pytest.fixture(scope="session", autouse=True)
def migrated_database() -> None:
    """Rebuild the test schema from the migrations, so the trigger under test is the real one."""
    cfg = Config(str(REPO_ROOT / "alembic.ini"))
    cfg.cmd_opts = argparse.Namespace(x=[f"url={TEST_DATABASE_URL}"])
    try:
        command.downgrade(cfg, "base")
        command.upgrade(cfg, "head")
    except (OSError, DBAPIError) as exc:
        pytest.exit(
            f"cannot reach the test database ({exc}). Run `docker compose up -d db` first.",
            returncode=2,
        )


@pytest.fixture(scope="session")
async def engine() -> AsyncGenerator[AsyncEngine]:
    eng = make_engine(TEST_DATABASE_URL)
    yield eng
    await eng.dispose()


@pytest.fixture(scope="session")
def sessionmaker(engine: AsyncEngine) -> SessionMaker:
    return make_sessionmaker(engine)


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    # _env_file=None: prove the tests need no .env.
    return Settings(
        _env_file=None,  # pyright: ignore[reportCallIssue]
        database_url=TEST_DATABASE_URL,
        workspace_root=tmp_path / "workspaces",
        max_steps=10,
    )


type MakeUser = Callable[[], Awaitable[tuple[User, str]]]


@pytest.fixture
def make_user(sessionmaker: SessionMaker) -> MakeUser:
    """Create a fresh user; returns (user, raw api key)."""

    async def _make() -> tuple[User, str]:
        api_key = generate_api_key()
        async with sessionmaker() as db:
            user = await store.create_user(
                db, f"user-{uuid.uuid4().hex[:8]}", hash_api_key(api_key)
            )
        return user, api_key

    return _make


async def new_session(sessionmaker: SessionMaker, user: User) -> uuid.UUID:
    async with sessionmaker() as db:
        return (await store.create_session(db, user.id)).id


def make_deps(
    engine: AsyncEngine,
    sessionmaker: SessionMaker,
    llm: LLMClient,
    tools: Sequence[Tool] = (),
    max_steps: int = 10,
) -> AgentDeps:
    return AgentDeps(
        engine=engine,
        sessionmaker=sessionmaker,
        llm=llm,
        tools=ToolRegistry(tools),
        prices=TEST_PRICES,
        agent_model=FAKE_MODEL,
        max_steps=max_steps,
    )


def tool_context(sessionmaker: SessionMaker, user_id: uuid.UUID | None = None) -> ToolContext:
    return ToolContext(
        user_id=user_id or uuid.uuid4(), session_id=uuid.uuid4(), sessionmaker=sessionmaker
    )


type MakeClient = Callable[..., Awaitable[httpx.AsyncClient]]


@pytest.fixture
async def make_client(settings: Settings) -> AsyncGenerator[MakeClient]:
    """Build the real FastAPI app around a FakeLLM and return an in-process HTTP client."""
    clients: list[httpx.AsyncClient] = []
    engines: list[AsyncEngine] = []

    async def _make(
        llm: LLMClient, api_key: str | None = None, search: SearchProvider | None = None
    ) -> httpx.AsyncClient:
        app = create_app(
            settings,
            llm=llm,
            search=search or FakeSearchProvider(),
            prices=TEST_PRICES,
            agent_model=FAKE_MODEL,
        )
        engines.append(app.state.agent_deps.engine)
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test", headers=headers
        )
        clients.append(client)
        return client

    yield _make
    for client in clients:
        await client.aclose()
    for eng in engines:
        await eng.dispose()
