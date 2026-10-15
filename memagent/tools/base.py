"""Tool interface and registry.

A tool is a name, a JSON Schema for its arguments, and an async `run`. The registry is the
single place that turns any failure into an error result, which is what guarantees
"tool exceptions are never raised" for the agent loop.
"""

import logging
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Protocol

from memagent.db.engine import SessionMaker
from memagent.llm.types import ToolCall, ToolSpec

logger = logging.getLogger(__name__)


class ToolError(Exception):
    """An expected failure with a message that is safe and useful to show the model."""


@dataclass(frozen=True)
class ToolContext:
    """What a tool may know about the request it serves. Tools get the user from here,
    never from model-supplied arguments, so the model can't act as another user."""

    user_id: uuid.UUID
    session_id: uuid.UUID
    sessionmaker: SessionMaker


@dataclass(frozen=True)
class ToolOutput:
    text: str
    is_error: bool = False
    # Spend incurred inside the tool (web_search's own model call), so the turn's
    # reported cost includes it.
    cost_usd: Decimal = Decimal(0)


class Tool(Protocol):
    @property
    def spec(self) -> ToolSpec: ...

    async def run(self, ctx: ToolContext, args: dict[str, Any]) -> ToolOutput: ...


def require_str(args: dict[str, Any], name: str) -> str:
    """Fetch a required string argument; the model sometimes omits or mistypes them."""
    value = args.get(name)
    if not isinstance(value, str) or not value:
        raise ToolError(f"argument {name!r} is required and must be a non-empty string")
    return value


class ToolRegistry:
    def __init__(self, tools: Sequence[Tool]) -> None:
        self._tools = {tool.spec.name: tool for tool in tools}

    def specs(self) -> list[ToolSpec]:
        return [tool.spec for tool in self._tools.values()]

    async def execute(self, ctx: ToolContext, call: ToolCall) -> ToolOutput:
        """Never raises: every failure becomes an error result the model can react to."""
        if call.arguments_error is not None:
            return ToolOutput(f"invalid arguments: {call.arguments_error}", is_error=True)
        tool = self._tools.get(call.name)
        if tool is None:
            return ToolOutput(f"unknown tool {call.name!r}", is_error=True)
        try:
            return await tool.run(ctx, call.arguments)
        except ToolError as exc:
            return ToolOutput(str(exc), is_error=True)
        except Exception as exc:
            # Unexpected bug. Log only the exception type: messages and tracebacks can carry
            # user content, and logs must not. Give the model something it can act on.
            logger.error("tool %s crashed with %s", call.name, type(exc).__name__)
            return ToolOutput(f"tool {call.name} failed: {type(exc).__name__}", is_error=True)
