"""The agent loop: tool round-trips, the step limit, and tool/LLM failures."""

from collections.abc import Sequence
from typing import Any

import pytest
from sqlalchemy.ext.asyncio import AsyncEngine

from memagent import store
from memagent.agent.loop import TurnFailed, run_turn
from memagent.db.engine import SessionMaker
from memagent.db.models import Event
from memagent.llm.base import LLMError
from memagent.llm.fake import FakeLLM, call_tool, reply
from memagent.llm.types import LLMResponse, Message, ToolCall, ToolResultMessage, ToolSpec, Usage
from memagent.tools.base import ToolContext, ToolOutput
from tests.conftest import MakeUser, make_deps, new_session


class EchoTool:
    spec = ToolSpec("echo", "Echo the text back.", {"type": "object", "properties": {}})

    async def run(self, ctx: ToolContext, args: dict[str, Any]) -> ToolOutput:
        return ToolOutput(f"echo: {args.get('text', '')}")


class BoomTool:
    spec = ToolSpec("boom", "Always crashes.", {"type": "object", "properties": {}})

    async def run(self, ctx: ToolContext, args: dict[str, Any]) -> ToolOutput:
        raise RuntimeError("unexpected bug inside a tool")


async def _events(sessionmaker: SessionMaker, user_id: Any, session_id: Any) -> list[Event]:
    async with sessionmaker() as db:
        return await store.list_events(db, user_id, session_id)


def _tool_results(messages: Sequence[Message]) -> list[ToolResultMessage]:
    return [m for m in messages if isinstance(m, ToolResultMessage)]


async def test_plain_reply_without_tools(
    make_user: MakeUser, engine: AsyncEngine, sessionmaker: SessionMaker
) -> None:
    user, _ = await make_user()
    session_id = await new_session(sessionmaker, user)
    deps = make_deps(engine, sessionmaker, FakeLLM([reply("hi there")]))

    result = await run_turn(deps, user.id, session_id, "hello")

    assert result.reply == "hi there"
    events = await _events(sessionmaker, user.id, session_id)
    assert [(e.seq, e.type) for e in events] == [(1, "user_message"), (2, "assistant_message")]


async def test_tool_call_then_answer(
    make_user: MakeUser, engine: AsyncEngine, sessionmaker: SessionMaker
) -> None:
    user, _ = await make_user()
    session_id = await new_session(sessionmaker, user)
    llm = FakeLLM([call_tool("echo", text="ping"), reply("done")])
    deps = make_deps(engine, sessionmaker, llm, tools=[EchoTool()])

    result = await run_turn(deps, user.id, session_id, "use the tool")

    assert result.reply == "done"
    assert [s.tool for s in result.steps] == ["echo"]
    events = await _events(sessionmaker, user.id, session_id)
    assert [e.type for e in events] == [
        "user_message",
        "tool_call",
        "tool_result",
        "assistant_message",
    ]
    # The second model call was shown the tool's result, rebuilt from the database.
    assert _tool_results(llm.calls[1]) == [
        ToolResultMessage("call_1", "echo", "echo: ping", is_error=False)
    ]


async def test_stops_at_max_steps_with_error_event(
    make_user: MakeUser, engine: AsyncEngine, sessionmaker: SessionMaker
) -> None:
    user, _ = await make_user()
    session_id = await new_session(sessionmaker, user)
    # A model that never stops asking for tools.
    llm = FakeLLM(lambda _messages: call_tool("echo", text="again"))
    deps = make_deps(engine, sessionmaker, llm, tools=[EchoTool()], max_steps=3)

    result = await run_turn(deps, user.id, session_id, "loop forever")

    assert len(llm.calls) == 3  # exactly max_steps model calls
    assert result.stopped_by_limit
    assert "3 steps" in result.reply
    events = await _events(sessionmaker, user.id, session_id)
    assert events[-1].type == "error"
    assert events[-1].payload["kind"] == "max_steps"
    # Steps 1 and 2 ran their tool; step 3's request was not stored or executed.
    assert [e.type for e in events].count("tool_call") == 2
    assert [e.type for e in events].count("tool_result") == 2


async def test_session_still_usable_after_step_limit(
    make_user: MakeUser, engine: AsyncEngine, sessionmaker: SessionMaker
) -> None:
    user, _ = await make_user()
    session_id = await new_session(sessionmaker, user)
    looping = FakeLLM(lambda _messages: call_tool("echo"))
    await run_turn(
        make_deps(engine, sessionmaker, looping, tools=[EchoTool()], max_steps=2),
        user.id,
        session_id,
        "first",
    )

    llm = FakeLLM([reply("fine")])
    result = await run_turn(make_deps(engine, sessionmaker, llm), user.id, session_id, "second")

    assert result.reply == "fine"
    # The error event is not shown to the model.
    assert all("I stopped after" not in getattr(m, "text", "") for m in llm.calls[0])


async def test_tool_exception_is_fed_back_not_raised(
    make_user: MakeUser, engine: AsyncEngine, sessionmaker: SessionMaker
) -> None:
    user, _ = await make_user()
    session_id = await new_session(sessionmaker, user)
    llm = FakeLLM([call_tool("boom"), reply("sorry, the tool failed")])
    deps = make_deps(engine, sessionmaker, llm, tools=[BoomTool()])

    result = await run_turn(deps, user.id, session_id, "try it")

    assert result.reply == "sorry, the tool failed"
    assert result.steps[0].is_error
    fed_back = _tool_results(llm.calls[1])
    assert len(fed_back) == 1 and fed_back[0].is_error
    assert "RuntimeError" in fed_back[0].output
    events = await _events(sessionmaker, user.id, session_id)
    assert events[2].type == "tool_result" and events[2].payload["is_error"] is True


async def test_unknown_tool_is_an_error_result(
    make_user: MakeUser, engine: AsyncEngine, sessionmaker: SessionMaker
) -> None:
    user, _ = await make_user()
    session_id = await new_session(sessionmaker, user)
    llm = FakeLLM([call_tool("no_such_tool"), reply("ok")])

    await run_turn(make_deps(engine, sessionmaker, llm), user.id, session_id, "x")

    (result,) = _tool_results(llm.calls[1])
    assert result.is_error and "unknown tool" in result.output


async def test_malformed_arguments_are_an_error_result(
    make_user: MakeUser, engine: AsyncEngine, sessionmaker: SessionMaker
) -> None:
    user, _ = await make_user()
    session_id = await new_session(sessionmaker, user)
    bad_call = LLMResponse(
        text="",
        usage=Usage(1, 1),
        model="fake",
        tool_calls=[ToolCall("call_1", "echo", {}, arguments_error="arguments are not valid JSON")],
    )
    llm = FakeLLM([bad_call, reply("ok")])

    await run_turn(
        make_deps(engine, sessionmaker, llm, tools=[EchoTool()]), user.id, session_id, "x"
    )

    (result,) = _tool_results(llm.calls[1])
    assert result.is_error and "invalid arguments" in result.output


async def test_llm_failure_writes_error_event(
    make_user: MakeUser, engine: AsyncEngine, sessionmaker: SessionMaker
) -> None:
    user, _ = await make_user()
    session_id = await new_session(sessionmaker, user)

    def fail(_messages: Sequence[Message]) -> LLMResponse:
        raise LLMError("OpenAI request failed: APIConnectionError")

    with pytest.raises(TurnFailed):
        await run_turn(make_deps(engine, sessionmaker, FakeLLM(fail)), user.id, session_id, "x")

    events = await _events(sessionmaker, user.id, session_id)
    assert [e.type for e in events] == ["user_message", "error"]
    assert events[1].payload["kind"] == "llm_error"
