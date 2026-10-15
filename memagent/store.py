"""All database reads and writes.

Rules this module enforces for the rest of the code:
- Every function that touches a business table takes `user_id` and filters by it.
  Code outside this module never writes SQL, so there is one place to audit.
- `append_event` is the only way to write an event, and it always scrubs first.
"""

import uuid
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from decimal import Decimal
from typing import Any

from sqlalchemy import func, select, text
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession

from memagent.db.models import ChatSession, Event, EventType, LLMCall, Module, User
from memagent.llm.types import Usage
from memagent.sanitize import scrub_json

# ---- users -----------------------------------------------------------------------------


async def create_user(db: AsyncSession, name: str, api_key_hash: str) -> User:
    user = User(name=name, api_key_hash=api_key_hash)
    db.add(user)
    await db.commit()
    return user


async def get_user_by_key_hash(db: AsyncSession, api_key_hash: str) -> User | None:
    result = await db.execute(select(User).where(User.api_key_hash == api_key_hash))
    return result.scalar_one_or_none()


# ---- sessions --------------------------------------------------------------------------


async def create_session(db: AsyncSession, user_id: uuid.UUID) -> ChatSession:
    chat = ChatSession(user_id=user_id)
    db.add(chat)
    await db.commit()
    return chat


async def get_session(
    db: AsyncSession, user_id: uuid.UUID, session_id: uuid.UUID
) -> ChatSession | None:
    """None both when the session doesn't exist and when it belongs to someone else.

    Callers turn None into 404, so another user's session is indistinguishable from a
    missing one (no existence leak).
    """
    result = await db.execute(
        select(ChatSession).where(ChatSession.id == session_id, ChatSession.user_id == user_id)
    )
    return result.scalar_one_or_none()


# ---- events ----------------------------------------------------------------------------


async def append_event(
    db: AsyncSession,
    user_id: uuid.UUID,
    session_id: uuid.UUID,
    type: EventType,
    payload: dict[str, Any],
) -> Event:
    """Scrub, assign the next seq, insert, commit.

    seq = max + 1 is only safe because the caller holds `session_lock` for this session.
    The UNIQUE(session_id, seq) constraint is the backstop if that rule is ever broken:
    a duplicate fails loudly instead of corrupting the order.
    Committing per event means a crash mid-turn still leaves an accurate log.
    """
    clean_payload, _counts = scrub_json(payload)
    next_seq = await db.scalar(
        select(func.coalesce(func.max(Event.seq), 0) + 1).where(
            Event.session_id == session_id, Event.user_id == user_id
        )
    )
    event = Event(
        user_id=user_id,
        session_id=session_id,
        seq=int(next_seq or 1),
        type=type,
        payload=clean_payload,
    )
    db.add(event)
    await db.commit()
    return event


async def list_events(db: AsyncSession, user_id: uuid.UUID, session_id: uuid.UUID) -> list[Event]:
    result = await db.execute(
        select(Event)
        .where(Event.user_id == user_id, Event.session_id == session_id)
        .order_by(Event.seq)
    )
    return list(result.scalars())


async def redact_event(db: AsyncSession, user_id: uuid.UUID, event_id: int) -> bool:
    """Calls the database function, the only sanctioned way to change an event (used in S7)."""
    redacted = await db.scalar(
        text("SELECT redact_event(:event_id, :user_id)"),
        {"event_id": event_id, "user_id": user_id},
    )
    await db.commit()
    return bool(redacted)


# ---- llm_calls -------------------------------------------------------------------------


async def record_llm_call(
    db: AsyncSession,
    user_id: uuid.UUID,
    session_id: uuid.UUID,
    module: Module,
    model: str,
    usage: Usage,
    cost_usd: Decimal,
    latency_ms: int,
) -> None:
    db.add(
        LLMCall(
            user_id=user_id,
            session_id=session_id,
            module=module,
            model=model,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            cached_input_tokens=usage.cached_input_tokens,
            search_calls=usage.search_calls,
            cost_usd=cost_usd,
            latency_ms=latency_ms,
        )
    )
    await db.commit()


# ---- per-session lock ------------------------------------------------------------------


def _lock_key(session_id: uuid.UUID) -> int:
    """Advisory locks take a signed 64-bit key; use the first 8 bytes of the UUID.

    Two different sessions could in theory share a key; the only effect would be that
    their turns run one after the other, which is harmless.
    """
    return int.from_bytes(session_id.bytes[:8], "big", signed=True)


@asynccontextmanager
async def session_lock(engine: AsyncEngine, session_id: uuid.UUID) -> AsyncGenerator[None]:
    """Serialize turns within one session, across processes.

    A session-level advisory lock (not a transaction-level one) on its own connection,
    so the turn's events can be committed one by one while the lock is held. If the
    process dies, Postgres drops the connection and releases the lock automatically.
    """
    key = _lock_key(session_id)
    async with engine.connect() as conn:
        await conn.execute(text("SELECT pg_advisory_lock(:key)"), {"key": key})
        # The SELECT opened an implicit transaction; end it so this connection doesn't
        # sit "idle in transaction" for the whole turn. The lock survives the commit.
        await conn.commit()
        try:
            yield
        finally:
            try:
                await conn.execute(text("SELECT pg_advisory_unlock(:key)"), {"key": key})
                await conn.commit()
            except Exception:
                # A pooled connection must never go back still holding the lock:
                # discard it, and closing the connection releases the lock.
                await conn.invalidate()
                raise
