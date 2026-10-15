"""The agent loop, written by hand (decision 0001).

One turn:
    lock the session
    append the user's message
    repeat up to max_steps times:
        rebuild history from events -> call the model -> record cost
        no tool calls?  -> store the answer, done
        last step?      -> store an error event, tell the user, done
        otherwise       -> store the calls, run each tool, store each result, go again

A "step" is one model call. Every write goes through `store.append_event`, so everything
is scrubbed and logged in order.
"""

import time
import uuid
from dataclasses import dataclass, field
from decimal import Decimal

from sqlalchemy.ext.asyncio import AsyncEngine

from memagent import store
from memagent.agent.history import events_to_messages
from memagent.agent.prompts import LLM_ERROR_MESSAGE, MAX_STEPS_MESSAGE, SYSTEM_PROMPT
from memagent.db.engine import SessionMaker
from memagent.db.models import EventType
from memagent.llm.base import LLMClient, LLMError
from memagent.llm.types import LLMResponse
from memagent.pricing import PriceTable, cost_usd
from memagent.tools.base import ToolContext, ToolRegistry


@dataclass(frozen=True)
class StepSummary:
    tool: str
    is_error: bool


@dataclass
class TurnResult:
    reply: str
    steps: list[StepSummary] = field(default_factory=list[StepSummary])
    cost_usd: Decimal = Decimal(0)
    stopped_by_limit: bool = False


@dataclass(frozen=True)
class AgentDeps:
    engine: AsyncEngine
    sessionmaker: SessionMaker
    llm: LLMClient
    tools: ToolRegistry
    prices: PriceTable
    agent_model: str
    max_steps: int


class TurnFailed(RuntimeError):
    """The model could not be reached. An error event has already been stored."""


async def run_turn(
    deps: AgentDeps, user_id: uuid.UUID, session_id: uuid.UUID, user_text: str
) -> TurnResult:
    """Run one user turn. The caller must already have checked the session belongs to user_id."""
    async with store.session_lock(deps.engine, session_id):
        async with deps.sessionmaker() as db:
            await store.append_event(db, user_id, session_id, "user_message", {"text": user_text})
        return await _run_steps(deps, user_id, session_id)


async def _run_steps(deps: AgentDeps, user_id: uuid.UUID, session_id: uuid.UUID) -> TurnResult:
    result = TurnResult(reply="")
    ctx = ToolContext(user_id=user_id, session_id=session_id, sessionmaker=deps.sessionmaker)

    for step in range(1, deps.max_steps + 1):
        response = await _call_model(deps, user_id, session_id, result)

        if not response.tool_calls:
            await _append(deps, user_id, session_id, "assistant_message", {"text": response.text})
            result.reply = response.text
            return result

        if step == deps.max_steps:
            # The model still wants tools but we are out of steps. Those calls are not
            # stored: a call without a result would make the next request invalid.
            break

        if response.text:
            await _append(deps, user_id, session_id, "assistant_message", {"text": response.text})
        for call in response.tool_calls:
            await _append(
                deps,
                user_id,
                session_id,
                "tool_call",
                {"call_id": call.call_id, "name": call.name, "arguments": call.arguments},
            )
            output = await deps.tools.execute(ctx, call)
            await _append(
                deps,
                user_id,
                session_id,
                "tool_result",
                {
                    "call_id": call.call_id,
                    "name": call.name,
                    "output": output.text,
                    "is_error": output.is_error,
                },
            )
            result.steps.append(StepSummary(tool=call.name, is_error=output.is_error))
            result.cost_usd += output.cost_usd

    message = MAX_STEPS_MESSAGE.format(max_steps=deps.max_steps)
    await _append(deps, user_id, session_id, "error", {"kind": "max_steps", "message": message})
    result.reply = message
    result.stopped_by_limit = True
    return result


async def _call_model(
    deps: AgentDeps, user_id: uuid.UUID, session_id: uuid.UUID, result: TurnResult
) -> LLMResponse:
    """Rebuild history from the database, call the model, and write the llm_calls row."""
    async with deps.sessionmaker() as db:
        events = await store.list_events(db, user_id, session_id)
    messages = events_to_messages(events, SYSTEM_PROMPT)

    started = time.monotonic()
    try:
        response = await deps.llm.complete(messages, deps.tools.specs(), deps.agent_model)
    except LLMError as exc:
        await _append(
            deps, user_id, session_id, "error", {"kind": "llm_error", "message": str(exc)}
        )
        raise TurnFailed(LLM_ERROR_MESSAGE) from exc
    latency_ms = int((time.monotonic() - started) * 1000)

    # Price by the model we asked for: the response may name a dated snapshot
    # (e.g. "<model>-2025-01-01") that is not a key in the price table.
    cost = cost_usd(deps.prices, deps.agent_model, response.usage)
    async with deps.sessionmaker() as db:
        await store.record_llm_call(
            db,
            user_id=user_id,
            session_id=session_id,
            module="agent",
            model=deps.agent_model,
            usage=response.usage,
            cost_usd=cost,
            latency_ms=latency_ms,
        )
    result.cost_usd += cost
    return response


async def _append(
    deps: AgentDeps,
    user_id: uuid.UUID,
    session_id: uuid.UUID,
    type: EventType,
    payload: dict[str, object],
) -> None:
    async with deps.sessionmaker() as db:
        await store.append_event(db, user_id, session_id, type, payload)
