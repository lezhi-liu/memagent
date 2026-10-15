"""Multi-tenancy: user B can never see or touch user A's sessions or events."""

import uuid

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from memagent import store
from memagent.db.engine import SessionMaker
from memagent.llm.fake import FakeLLM, reply
from tests.conftest import MakeClient, MakeUser


async def _session_with_one_turn(make_client: MakeClient, api_key: str) -> str:
    client = await make_client(FakeLLM([reply("hello A")]), api_key)
    session_id = (await client.post("/sessions")).json()["id"]
    assert (
        await client.post(f"/sessions/{session_id}/messages", json={"text": "hi"})
    ).status_code == 200
    return session_id


async def test_other_users_session_is_404(make_user: MakeUser, make_client: MakeClient) -> None:
    _, key_a = await make_user()
    _, key_b = await make_user()
    session_a = await _session_with_one_turn(make_client, key_a)

    llm_b = FakeLLM([reply("should never run")])
    client_b = await make_client(llm_b, key_b)
    assert (await client_b.get(f"/sessions/{session_a}/events")).status_code == 404
    post = await client_b.post(f"/sessions/{session_a}/messages", json={"text": "hijack"})
    assert post.status_code == 404
    assert llm_b.calls == []  # B's request never reached the model

    # A still sees only its own two events, untouched.
    client_a = await make_client(FakeLLM([]), key_a)
    events = (await client_a.get(f"/sessions/{session_a}/events")).json()
    assert [e["type"] for e in events] == ["user_message", "assistant_message"]


async def test_missing_session_is_also_404(make_user: MakeUser, make_client: MakeClient) -> None:
    """Same status as another user's session, so ids can't be probed for existence."""
    _, key = await make_user()
    client = await make_client(FakeLLM([]), key)
    assert (await client.get(f"/sessions/{uuid.uuid4()}/events")).status_code == 404


async def test_missing_or_bad_key_is_401(make_client: MakeClient) -> None:
    anonymous = await make_client(FakeLLM([]))
    assert (await anonymous.post("/sessions")).status_code == 401
    bad = await make_client(FakeLLM([]), "mk_not-a-real-key")
    assert (await bad.post("/sessions")).status_code == 401


async def test_store_get_session_filters_by_user(
    make_user: MakeUser, sessionmaker: SessionMaker
) -> None:
    user_a, _ = await make_user()
    user_b, _ = await make_user()
    async with sessionmaker() as db:
        chat = await store.create_session(db, user_a.id)
        assert await store.get_session(db, user_b.id, chat.id) is None
        assert await store.get_session(db, user_a.id, chat.id) is not None


async def test_redact_event_cannot_touch_other_users_event(
    make_user: MakeUser, sessionmaker: SessionMaker, engine: AsyncEngine
) -> None:
    user_a, _ = await make_user()
    user_b, _ = await make_user()
    async with sessionmaker() as db:
        chat = await store.create_session(db, user_a.id)
        event = await store.append_event(db, user_a.id, chat.id, "user_message", {"text": "mine"})
        assert await store.redact_event(db, user_b.id, event.id) is False
    async with engine.connect() as conn:
        payload = await conn.scalar(
            text("SELECT payload->>'text' FROM events WHERE id = :id"), {"id": event.id}
        )
    assert payload == "mine"
