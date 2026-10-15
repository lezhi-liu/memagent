"""FastAPI dependencies: the agent wiring, a DB session, and the authenticated user."""

from collections.abc import AsyncIterator
from typing import Annotated, cast

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.ext.asyncio import AsyncSession

from memagent import store
from memagent.agent.loop import AgentDeps
from memagent.auth import hash_api_key
from memagent.db.models import User

_bearer = HTTPBearer(auto_error=False)


def get_agent_deps(request: Request) -> AgentDeps:
    return cast(AgentDeps, request.app.state.agent_deps)


async def get_db(
    deps: Annotated[AgentDeps, Depends(get_agent_deps)],
) -> AsyncIterator[AsyncSession]:
    async with deps.sessionmaker() as db:
        yield db


async def current_user(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> User:
    """Look the user up by the hash of their key. The raw key is never stored or logged."""
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "missing bearer API key")
    user = await store.get_user_by_key_hash(db, hash_api_key(credentials.credentials))
    if user is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "invalid API key")
    return user


DepsDep = Annotated[AgentDeps, Depends(get_agent_deps)]
DbDep = Annotated[AsyncSession, Depends(get_db)]
UserDep = Annotated[User, Depends(current_user)]
