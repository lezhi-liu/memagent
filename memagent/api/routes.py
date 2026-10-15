"""HTTP routes. Every session lookup is scoped to the authenticated user; a session that
belongs to someone else is reported as 404, exactly like one that doesn't exist."""

import uuid

from fastapi import APIRouter, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from memagent import store
from memagent.agent.loop import TurnFailed, run_turn
from memagent.api.deps import DbDep, DepsDep, UserDep
from memagent.api.schemas import EventOut, MessageIn, SessionOut, StepOut, TurnOut
from memagent.db.models import ChatSession

router = APIRouter()


async def _own_session_or_404(
    db: AsyncSession, user_id: uuid.UUID, session_id: uuid.UUID
) -> ChatSession:
    chat = await store.get_session(db, user_id, session_id)
    if chat is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "session not found")
    return chat


@router.post("/sessions", status_code=status.HTTP_201_CREATED)
async def create_session(user: UserDep, db: DbDep) -> SessionOut:
    chat = await store.create_session(db, user.id)
    return SessionOut(id=chat.id, created_at=chat.created_at)


@router.post("/sessions/{session_id}/messages")
async def post_message(
    session_id: uuid.UUID, body: MessageIn, user: UserDep, db: DbDep, deps: DepsDep
) -> TurnOut:
    await _own_session_or_404(db, user.id, session_id)
    # End this request's read transaction now: a turn can take minutes, and the loop uses
    # its own short sessions. Otherwise this connection sits "idle in transaction".
    await db.commit()
    try:
        result = await run_turn(deps, user.id, session_id, body.text)
    except TurnFailed as exc:
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, str(exc)) from exc
    return TurnOut(
        reply=result.reply,
        steps=[StepOut(tool=s.tool, is_error=s.is_error) for s in result.steps],
        cost_usd=result.cost_usd,
        stopped_by_limit=result.stopped_by_limit,
    )


@router.get("/sessions/{session_id}/events")
async def get_events(session_id: uuid.UUID, user: UserDep, db: DbDep) -> list[EventOut]:
    await _own_session_or_404(db, user.id, session_id)
    events = await store.list_events(db, user.id, session_id)
    return [
        EventOut(
            id=e.id,
            seq=e.seq,
            type=e.type,
            payload=e.payload,
            created_at=e.created_at,
            redacted_at=e.redacted_at,
        )
        for e in events
    ]
