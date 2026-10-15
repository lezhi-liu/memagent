"""Engine and session factory. One engine per process; one AsyncSession per unit of work."""

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

type SessionMaker = async_sessionmaker[AsyncSession]


def make_engine(database_url: str) -> AsyncEngine:
    # pool_pre_ping: survive a restarted Postgres container without a confusing first error.
    return create_async_engine(database_url, pool_pre_ping=True)


def make_sessionmaker(engine: AsyncEngine) -> SessionMaker:
    # expire_on_commit=False: we commit after each event and keep using the returned object.
    return async_sessionmaker(engine, expire_on_commit=False)
