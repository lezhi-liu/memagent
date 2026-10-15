"""The events table is append-only, enforced by the database trigger itself.

These tests use raw SQL on purpose: the guarantee must hold even for code (or a person
with psql) that bypasses store.py.
"""

import pytest
from sqlalchemy import text
from sqlalchemy.exc import DBAPIError
from sqlalchemy.ext.asyncio import AsyncEngine

from memagent import store
from memagent.db.engine import SessionMaker
from memagent.db.models import Event
from tests.conftest import MakeUser


async def _one_event(make_user: MakeUser, sessionmaker: SessionMaker) -> Event:
    user, _ = await make_user()
    async with sessionmaker() as db:
        chat = await store.create_session(db, user.id)
        return await store.append_event(db, user.id, chat.id, "user_message", {"text": "original"})


async def _assert_rejected(engine: AsyncEngine, sql: str, event_id: int) -> None:
    async with engine.connect() as conn:
        with pytest.raises(DBAPIError, match="append-only"):
            await conn.execute(text(sql), {"id": event_id})


async def test_update_payload_is_rejected(
    make_user: MakeUser, sessionmaker: SessionMaker, engine: AsyncEngine
) -> None:
    event = await _one_event(make_user, sessionmaker)
    await _assert_rejected(
        engine, """UPDATE events SET payload = '{"text": "edited"}' WHERE id = :id""", event.id
    )


async def test_redacted_at_without_marker_payload_is_rejected(
    make_user: MakeUser, sessionmaker: SessionMaker, engine: AsyncEngine
) -> None:
    event = await _one_event(make_user, sessionmaker)
    await _assert_rejected(engine, "UPDATE events SET redacted_at = now() WHERE id = :id", event.id)


async def test_redaction_that_also_changes_seq_is_rejected(
    make_user: MakeUser, sessionmaker: SessionMaker, engine: AsyncEngine
) -> None:
    event = await _one_event(make_user, sessionmaker)
    await _assert_rejected(
        engine,
        """UPDATE events SET payload = '{"redacted": true}', redacted_at = now(), seq = 99
           WHERE id = :id""",
        event.id,
    )


async def test_delete_is_rejected(
    make_user: MakeUser, sessionmaker: SessionMaker, engine: AsyncEngine
) -> None:
    event = await _one_event(make_user, sessionmaker)
    await _assert_rejected(engine, "DELETE FROM events WHERE id = :id", event.id)


async def test_truncate_is_rejected(engine: AsyncEngine) -> None:
    async with engine.connect() as conn:
        with pytest.raises(DBAPIError, match="append-only"):
            await conn.execute(text("TRUNCATE events"))


async def test_redact_event_succeeds_once(
    make_user: MakeUser, sessionmaker: SessionMaker, engine: AsyncEngine
) -> None:
    event = await _one_event(make_user, sessionmaker)
    async with sessionmaker() as db:
        assert await store.redact_event(db, event.user_id, event.id) is True
        # Already redacted: the function finds nothing to do...
        assert await store.redact_event(db, event.user_id, event.id) is False

    async with engine.connect() as conn:
        row = (
            await conn.execute(
                text("SELECT payload, redacted_at, seq FROM events WHERE id = :id"),
                {"id": event.id},
            )
        ).one()
    assert row.payload == {"redacted": True}
    assert row.redacted_at is not None
    assert row.seq == event.seq

    # ...and a manual second redaction is rejected by the trigger (redacted_at is set).
    await _assert_rejected(
        engine,
        """UPDATE events SET payload = '{"redacted": true}', redacted_at = now() WHERE id = :id""",
        event.id,
    )
