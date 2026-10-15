"""Rebuild the model's message list from stored events.

The database is the only source of history: the model sees exactly what was persisted,
which means already-scrubbed content (decision 0003). Nothing is kept in process memory
between requests, so any API process can serve any turn.
"""

from collections.abc import Sequence

from memagent.db.models import Event
from memagent.llm.types import (
    AssistantMessage,
    Message,
    SystemMessage,
    ToolCallMessage,
    ToolResultMessage,
    UserMessage,
)

REDACTED_TEXT = "[this message was deleted]"


def event_to_message(event: Event) -> Message | None:
    payload = event.payload
    if event.redacted_at is not None:
        # Keep the slot so the conversation still reads in order, but say nothing about
        # what was there. Tool pairs are dropped: a redacted call without its arguments
        # (or a result without its call) is not a valid tool exchange for the model.
        if event.type in ("user_message", "assistant_message"):
            cls = UserMessage if event.type == "user_message" else AssistantMessage
            return cls(REDACTED_TEXT)
        return None
    match event.type:
        case "user_message":
            return UserMessage(payload["text"])
        case "assistant_message":
            return AssistantMessage(payload["text"])
        case "tool_call":
            return ToolCallMessage(payload["call_id"], payload["name"], payload["arguments"])
        case "tool_result":
            return ToolResultMessage(
                payload["call_id"], payload["name"], payload["output"], payload["is_error"]
            )
        case _:
            # "error" events are for people and audits, not for the model.
            return None


def events_to_messages(events: Sequence[Event], system_prompt: str) -> list[Message]:
    messages: list[Message] = [SystemMessage(system_prompt)]
    for event in events:
        message = event_to_message(event)
        if message is not None:
            messages.append(message)
    return _drop_unpaired_tool_messages(messages)


def _drop_unpaired_tool_messages(messages: list[Message]) -> list[Message]:
    """Providers reject a tool call without a result (or the reverse).

    That can happen if the process died between writing a tool_call and its tool_result,
    or after a redaction. Dropping the orphan keeps the session usable.
    """
    call_ids = {m.call_id for m in messages if isinstance(m, ToolCallMessage)}
    result_ids = {m.call_id for m in messages if isinstance(m, ToolResultMessage)}
    paired = call_ids & result_ids
    return [
        m
        for m in messages
        if not isinstance(m, ToolCallMessage | ToolResultMessage) or m.call_id in paired
    ]
