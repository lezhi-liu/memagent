"""Alembic environment: async engine, URL from Settings unless overridden with `-x url=`."""

import asyncio
from logging.config import fileConfig

from alembic import context
from sqlalchemy.engine import Connection

from memagent.config import Settings
from memagent.db.engine import make_engine

config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name, disable_existing_loggers=False)


def _database_url() -> str:
    override = context.get_x_argument(as_dictionary=True).get("url")
    return override or Settings().database_url


def _run(connection: Connection) -> None:
    context.configure(connection=connection, target_metadata=None)
    with context.begin_transaction():
        context.run_migrations()


async def _run_async() -> None:
    engine = make_engine(_database_url())
    async with engine.connect() as connection:
        await connection.run_sync(_run)
    await engine.dispose()


if context.is_offline_mode():
    raise SystemExit("offline mode is not supported; run against a database")
asyncio.run(_run_async())
