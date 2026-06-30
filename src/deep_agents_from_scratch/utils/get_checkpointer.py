"""Factory for LangGraph checkpointers."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from enum import Enum
from typing import TypeVar
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

MOCK_SQLITE_CONN_STRING = "deep_agents.db"
MOCK_POSTGRES_CONN_STRING = "postgresql://user:password@localhost:5432/deep_agents"

_T = TypeVar("_T", bound=BaseCheckpointSaver)


class CheckpointerType(str, Enum):
    """Supported checkpointer backends per LangGraph checkpoint savers."""

    IN_MEMORY = "in_memory"
    ASYNC_SQLITE = "async_sqlite"
    ASYNC_POSTGRESQL = "async_postgresql"


@asynccontextmanager
async def async_sqlite_checkpointer(
    conn_string: str = MOCK_SQLITE_CONN_STRING,
) -> AsyncIterator[AsyncSqliteSaver]:
    """Open an async SQLite checkpointer with tables initialized.

    Usage::

        async with async_sqlite_checkpointer() as checkpointer:
            graph = builder.compile(checkpointer=checkpointer)
    """
    cm = AsyncSqliteSaver.from_conn_string(conn_string)
    checkpointer = await cm.__aenter__()
    try:
        await checkpointer.setup()
        yield checkpointer
    finally:
        await cm.__aexit__(None, None, None)


@asynccontextmanager
async def async_postgres_checkpointer(
    conn_string: str = MOCK_POSTGRES_CONN_STRING,
) -> AsyncIterator[AsyncPostgresSaver]:
    """Open an async Postgres checkpointer with tables initialized.

    Usage::

        async with async_postgres_checkpointer() as checkpointer:
            graph = builder.compile(checkpointer=checkpointer)
    """
    cm = AsyncPostgresSaver.from_conn_string(conn_string)
    checkpointer = await cm.__aenter__()
    try:
        await checkpointer.setup()
        yield checkpointer
    finally:
        await cm.__aexit__(None, None, None)


def _run_async(coro):
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    raise RuntimeError(
        "Cannot create an async checkpointer while an event loop is running. "
        "Use AsyncSqliteSaver.from_conn_string() or "
        "AsyncPostgresSaver.from_conn_string() directly in async code."
    )


async def _enter_async_context(
    context: AbstractAsyncContextManager[_T],
) -> _T:
    return await context.__aenter__()


def _open_async_checkpointer(
    context: AbstractAsyncContextManager[_T],
) -> _T:
    return _run_async(_enter_async_context(context))


def get_checkpointer(checkpointer_type: CheckpointerType) -> BaseCheckpointSaver:
    """Return a checkpointer for the given backend type.

    ``IN_MEMORY`` is ready for ``graph.compile(checkpointer=...)``. For async savers, prefer
    ``async with async_sqlite_checkpointer()`` or
    ``async with async_postgres_checkpointer()`` directly. This factory
    uses ``asyncio.run`` when no event loop is active.
    """
    if checkpointer_type is CheckpointerType.IN_MEMORY:
        return MemorySaver()

    if checkpointer_type is CheckpointerType.ASYNC_SQLITE:
        return _open_async_checkpointer(async_sqlite_checkpointer())

    if checkpointer_type is CheckpointerType.ASYNC_POSTGRESQL:
        return _open_async_checkpointer(async_postgres_checkpointer())