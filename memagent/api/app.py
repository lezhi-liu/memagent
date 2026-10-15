"""App factory.

`create_app` takes its collaborators as arguments so tests can pass FakeLLM and a fake
search provider. `create_default_app` builds the real OpenAI-backed app from the
environment; it is what uvicorn runs (with `--factory`), and it fails fast if a key, a
model name or a price entry is missing, before any money is spent.
"""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from openai import AsyncOpenAI

from memagent.agent.loop import AgentDeps
from memagent.api.routes import router
from memagent.config import Settings
from memagent.db.engine import make_engine, make_sessionmaker
from memagent.llm.base import LLMClient
from memagent.llm.openai_provider import OpenAIResponsesClient
from memagent.pricing import PriceTable, load_price_table
from memagent.tools.base import ToolRegistry
from memagent.tools.web_fetch import WebFetchTool
from memagent.tools.web_search import OpenAISearchProvider, SearchProvider, WebSearchTool
from memagent.tools.workspace import WorkspaceListTool, WorkspaceReadTool, WorkspaceWriteTool


def build_tools(settings: Settings, search: SearchProvider, prices: PriceTable) -> ToolRegistry:
    return ToolRegistry(
        [
            WebSearchTool(search, prices),
            WebFetchTool(),
            WorkspaceListTool(settings.workspace_root),
            WorkspaceReadTool(settings.workspace_root),
            WorkspaceWriteTool(settings.workspace_root),
        ]
    )


def create_app(
    settings: Settings,
    llm: LLMClient,
    search: SearchProvider,
    prices: PriceTable,
    agent_model: str,
) -> FastAPI:
    engine = make_engine(settings.database_url)

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncGenerator[None]:
        yield
        await engine.dispose()

    app = FastAPI(title="memagent", lifespan=lifespan)
    app.state.agent_deps = AgentDeps(
        engine=engine,
        sessionmaker=make_sessionmaker(engine),
        llm=llm,
        tools=build_tools(settings, search, prices),
        prices=prices,
        agent_model=agent_model,
        max_steps=settings.max_steps,
    )
    app.include_router(router)
    return app


def create_default_app() -> FastAPI:
    settings = Settings()
    api_key, agent_model, search_model = settings.require_openai()
    prices = load_price_table(settings.prices_path)
    prices.require(agent_model)  # fail now, not after the first paid call
    prices.require(search_model)
    client = AsyncOpenAI(api_key=api_key, max_retries=2)
    return create_app(
        settings,
        llm=OpenAIResponsesClient(client),
        search=OpenAISearchProvider(client, search_model),
        prices=prices,
        agent_model=agent_model,
    )
